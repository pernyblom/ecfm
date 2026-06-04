from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
from PIL import Image


CHANNELS = {
    "red": 0,
    "r": 0,
    "green": 1,
    "g": 1,
    "blue": 2,
    "b": 2,
    "alpha": 3,
    "a": 3,
}

PRESETS = {
    "xt_my_to_xt": ("xt_my", "xt", "green"),
    "yt_mx_to_yt": ("yt_mx", "yt", "green"),
    "cstr3_to_cstr2": ("cstr3", "cstr2", "green"),
}


@dataclass
class ConvertStats:
    scanned: int = 0
    converted: int = 0
    skipped_existing: int = 0
    skipped_channel: int = 0
    failed: int = 0


@dataclass(frozen=True)
class Conversion:
    source_rep: str
    target_rep: str
    zero_channel: str


def _parse_conversion(value: str) -> Conversion:
    parts = value.split(":")
    if len(parts) not in {2, 3}:
        raise argparse.ArgumentTypeError(
            "Conversions must be source_rep:target_rep or source_rep:target_rep:zero_channel."
        )
    source_rep, target_rep = parts[0].strip(), parts[1].strip()
    zero_channel = parts[2].strip() if len(parts) == 3 else "green"
    if not source_rep or not target_rep:
        raise argparse.ArgumentTypeError("Conversion source and target representations must be non-empty.")
    if source_rep == target_rep:
        raise argparse.ArgumentTypeError("Conversion source and target representations must be different.")
    if zero_channel not in CHANNELS:
        raise argparse.ArgumentTypeError(f"Unknown zero channel '{zero_channel}'.")
    return Conversion(source_rep=source_rep, target_rep=target_rep, zero_channel=zero_channel)


def _resolve_conversions(args: argparse.Namespace) -> list[Conversion]:
    if args.conversion:
        return list(args.conversion)
    if args.preset:
        presets = list(PRESETS) if "all" in args.preset else args.preset
        out = []
        for preset in presets:
            source_rep, target_rep, zero_channel = PRESETS[preset]
            out.append(Conversion(source_rep=source_rep, target_rep=target_rep, zero_channel=zero_channel))
        return out
    if args.source_rep == args.target_rep:
        raise ValueError("--source-rep and --target-rep must be different.")
    return [Conversion(source_rep=args.source_rep, target_rep=args.target_rep, zero_channel=args.zero_channel)]


def _iter_source_files(root: Path, *, source_rep: str, folders: Optional[set[str]]) -> Iterable[Path]:
    suffix = f"_{source_rep}.png"
    for path in sorted(root.rglob(f"*{suffix}")):
        if not path.is_file():
            continue
        if folders is not None:
            rel_parts = path.relative_to(root).parts
            if not rel_parts or rel_parts[0] not in folders:
                continue
        yield path


def _target_path(source_path: Path, *, source_rep: str, target_rep: str) -> Path:
    source_suffix = f"_{source_rep}.png"
    if not source_path.name.endswith(source_suffix):
        raise ValueError(f"Source filename does not end with {source_suffix}: {source_path}")
    target_name = source_path.name[: -len(source_suffix)] + f"_{target_rep}.png"
    return source_path.with_name(target_name)


def _zero_channel(source_path: Path, target_path: Path, *, channel_idx: int) -> bool:
    with Image.open(source_path) as image:
        image = image.convert("RGBA") if image.mode in {"LA", "P"} else image
        arr = np.asarray(image).copy()
        if arr.ndim != 3 or arr.shape[2] <= channel_idx:
            return False
        arr[..., channel_idx] = 0
        out = Image.fromarray(arr, mode=image.mode)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        out.save(target_path)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create a target event representation by copying source PNGs and zeroing one channel. "
            "By default this converts *_xt_my.png to *_xt.png by setting green to zero."
        )
    )
    parser.add_argument(
        "root",
        type=Path,
        help="Representation output root to scan recursively, for example datasets/FRED or a rendered images root.",
    )
    parser.add_argument(
        "--preset",
        choices=["all", *sorted(PRESETS)],
        action="append",
        default=None,
        help=(
            "Named conversion to run. Can be repeated. Use all for xt_my->xt, "
            "yt_mx->yt, and cstr3->cstr2."
        ),
    )
    parser.add_argument(
        "--conversion",
        type=_parse_conversion,
        action="append",
        default=None,
        help=(
            "Custom conversion as source_rep:target_rep or source_rep:target_rep:zero_channel. "
            "Can be repeated."
        ),
    )
    parser.add_argument("--source-rep", type=str, default="xt_my", help="Source representation suffix.")
    parser.add_argument("--target-rep", type=str, default="xt", help="Target representation suffix.")
    parser.add_argument(
        "--zero-channel",
        choices=sorted(CHANNELS),
        default="green",
        help="Channel to set to zero in the copied image.",
    )
    parser.add_argument(
        "--folder",
        action="append",
        default=None,
        help=(
            "Optional top-level subfolder under root to process. Can be passed multiple times, "
            "for example --folder 0 --folder 1."
        ),
    )
    parser.add_argument("--max-count", type=int, default=None, help="Maximum number of files to convert.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing target files.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned conversions without writing files.")
    parser.add_argument("--quiet", action="store_true", help="Only print the final summary.")
    args = parser.parse_args()

    root = args.root
    if not root.exists():
        raise FileNotFoundError(f"Root does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Root is not a directory: {root}")
    if args.max_count is not None and args.max_count < 0:
        raise ValueError("--max-count must be non-negative.")
    if args.conversion and args.preset:
        raise ValueError("Use either --conversion or --preset, not both.")

    folders = {str(folder).strip("/\\") for folder in args.folder} if args.folder else None
    conversions = _resolve_conversions(args)
    stats = ConvertStats()

    for conversion in conversions:
        channel_idx = CHANNELS[conversion.zero_channel]
        if not args.quiet:
            print(
                f"Conversion: {conversion.source_rep} -> {conversion.target_rep} "
                f"(zero {conversion.zero_channel})"
            )
        for source_path in _iter_source_files(root, source_rep=conversion.source_rep, folders=folders):
            if args.max_count is not None and stats.converted >= args.max_count:
                break
            stats.scanned += 1
            target_path = _target_path(
                source_path,
                source_rep=conversion.source_rep,
                target_rep=conversion.target_rep,
            )
            if target_path.exists() and not args.overwrite:
                stats.skipped_existing += 1
                if not args.quiet:
                    print(f"SKIP existing: {target_path}")
                continue
            if args.dry_run:
                stats.converted += 1
                if not args.quiet:
                    print(f"DRY RUN: {source_path} -> {target_path} (zero {conversion.zero_channel})")
                continue
            try:
                ok = _zero_channel(source_path, target_path, channel_idx=channel_idx)
            except Exception as exc:
                stats.failed += 1
                print(f"FAILED: {source_path} -> {target_path}: {exc}")
                continue
            if not ok:
                stats.skipped_channel += 1
                if not args.quiet:
                    print(f"SKIP channel missing: {source_path}")
                continue
            stats.converted += 1
            if not args.quiet:
                print(f"WROTE: {target_path}")

    action = "Would write" if args.dry_run else "Wrote"
    print(
        f"{action} {stats.converted} files from {stats.scanned} scanned "
        f"({stats.skipped_existing} existing, {stats.skipped_channel} missing channel, {stats.failed} failed)."
    )


if __name__ == "__main__":
    main()
