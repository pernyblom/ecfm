import argparse
import shutil
from pathlib import Path


def _folder_size(path: Path) -> tuple[int, int]:
    total_bytes = 0
    total_files = 0
    for child in path.rglob("*"):
        if not child.is_file():
            continue
        try:
            stat = child.stat()
        except OSError:
            continue
        total_bytes += stat.st_size
        total_files += 1
    return total_bytes, total_files


def _file_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def _format_bytes(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or unit == "TiB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} {unit}"
        value /= 1024.0
    return f"{num_bytes} B"


def _iter_fred_ids(fred_root: Path) -> list[Path]:
    folders = [path for path in fred_root.iterdir() if path.is_dir()]
    return sorted(folders, key=lambda p: (not p.name.isdigit(), int(p.name) if p.name.isdigit() else p.name))


def _remove_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Free FRED dataset storage by removing Event/Frames folders and removing "
            "Event/events.raw only when Event/output_events.npz exists."
        )
    )
    parser.add_argument(
        "--fred-root",
        type=Path,
        default=Path("datasets/FRED"),
        help="Root path of FRED dataset (default: datasets/FRED).",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually delete the listed paths. Without this flag, the script is a dry run.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only print the summary.",
    )
    args = parser.parse_args()

    fred_root = args.fred_root
    if not fred_root.exists():
        raise FileNotFoundError(f"FRED root does not exist: {fred_root}")
    if not fred_root.is_dir():
        raise NotADirectoryError(f"FRED root is not a directory: {fred_root}")

    mode = "DELETE" if args.execute else "DRY RUN"
    print(f"{mode}: scanning {fred_root}")

    frames_dirs = 0
    raw_files = 0
    frame_files = 0
    bytes_to_free = 0

    for folder in _iter_fred_ids(fred_root):
        event_dir = folder / "Event"
        frames_dir = event_dir / "Frames"
        raw_path = event_dir / "events.raw"
        npz_path = event_dir / "output_events.npz"

        targets: list[tuple[Path, int, int, str]] = []
        if frames_dir.exists():
            size, count = _folder_size(frames_dir)
            targets.append((frames_dir, size, count, "frames"))
            frames_dirs += 1
            frame_files += count
            bytes_to_free += size

        if raw_path.exists() and npz_path.exists():
            size = _file_size(raw_path)
            targets.append((raw_path, size, 1, "raw-with-npz"))
            raw_files += 1
            bytes_to_free += size

        for path, size, count, reason in targets:
            if not args.quiet:
                noun = "files" if count != 1 else "file"
                print(f"{mode}: {path} ({reason}, {count} {noun}, {_format_bytes(size)})")
            if args.execute:
                _remove_path(path)

    action = "Freed" if args.execute else "Would free"
    print(
        f"{action} {_format_bytes(bytes_to_free)} from {frames_dirs} Frames folders "
        f"({frame_files} files) and {raw_files} raw files."
    )
    if not args.execute:
        print("Dry run only. Re-run with --execute to delete these paths.")


if __name__ == "__main__":
    main()
