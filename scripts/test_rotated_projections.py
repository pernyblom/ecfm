from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from ecfm.data.tokenizer import Region, build_patch
from ecfm.training.linear_probe import load_events


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=["thu-eact", "dvs-lip"], required=True)
    parser.add_argument("--dataset-path", type=str, required=True)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--plane-types", type=str, nargs="+", default=["xy_p45", "xy_m45"])
    parser.add_argument("--image-width", type=int, default=128)
    parser.add_argument("--image-height", type=int, default=128)
    parser.add_argument("--time-unit", type=float, default=1e-6)
    parser.add_argument("--time-bins", type=int, default=64)
    parser.add_argument("--patch-size", type=int, default=32)
    parser.add_argument("--region-x", type=int, default=0)
    parser.add_argument("--region-y", type=int, default=0)
    parser.add_argument("--region-t", type=float, default=0.0)
    parser.add_argument("--region-dx", type=int, default=0)
    parser.add_argument("--region-dy", type=int, default=0)
    parser.add_argument("--region-dt", type=float, default=0.0)
    parser.add_argument("--patch-divider", type=float, default=0.0)
    parser.add_argument("--patch-norm", type=str, default="region_max")
    parser.add_argument("--patch-norm-eps", type=float, default=1e-6)
    parser.add_argument("--out-dir", type=str, default="outputs/rotated_projection_test")
    return parser.parse_args()


def select_file(dataset: str, root: Path, split: str, index: int) -> tuple[Path, int]:
    if dataset == "thu-eact":
        list_path = root / f"{split}.txt"
        if not list_path.exists():
            raise FileNotFoundError(f"Missing split list: {list_path}")
        entries = []
        for line in list_path.read_text().splitlines():
            parts = line.split()
            if not parts:
                continue
            name = Path(parts[0]).name
            label = int(parts[1]) if len(parts) > 1 else -1
            candidate = root / name
            if candidate.exists():
                entries.append((candidate, label))
        if not entries:
            raise FileNotFoundError("No dataset files found for split list")
        idx = max(0, min(index, len(entries) - 1))
        return entries[idx]

    split_dir = root / split
    class_dirs = sorted([p for p in split_dir.iterdir() if p.is_dir()])
    if not class_dirs:
        raise FileNotFoundError(f"No class folders found in {split_dir}")
    files = []
    labels = []
    for label, class_dir in enumerate(class_dirs):
        for path in sorted(class_dir.glob("*.npy")):
            files.append(path)
            labels.append(label)
    if not files:
        raise FileNotFoundError(f"No .npy files found in {split_dir}")
    idx = max(0, min(index, len(files) - 1))
    return files[idx], labels[idx]


def main() -> None:
    args = parse_args()
    root = Path(args.dataset_path)
    sample_path, label = select_file(args.dataset, root, args.split, args.index)
    events, seq_len_sec = load_events(sample_path, args.time_unit)

    dx = args.region_dx if args.region_dx > 0 else args.image_width
    dy = args.region_dy if args.region_dy > 0 else args.image_height
    dt = args.region_dt if args.region_dt > 0 else 1.0
    region = {
        "x": args.region_x,
        "y": args.region_y,
        "t": args.region_t,
        "dx": dx,
        "dy": dy,
        "dt": dt,
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for plane in args.plane_types:
        reg = Region(
            x=region["x"],
            y=region["y"],
            t=region["t"],
            dx=region["dx"],
            dy=region["dy"],
            dt=region["dt"],
            plane=plane,
        )
        patch, total_events = build_patch(
            events,
            reg,
            args.patch_size,
            args.time_bins,
            patch_divider=args.patch_divider,
            norm_mode=args.patch_norm,
            norm_eps=args.patch_norm_eps,
        )
        patch_np = patch.numpy()
        for ch in range(patch_np.shape[0]):
            img = np.clip(patch_np[ch] * 255.0, 0, 255).astype(np.uint8)
            Image.fromarray(img, mode="L").save(out_dir / f"{plane}_p{ch}.png")
        meta = {
            "dataset": args.dataset,
            "file": str(sample_path),
            "label": int(label),
            "region": region,
            "plane": plane,
            "time_bins": args.time_bins,
            "patch_size": args.patch_size,
            "seq_len_sec": float(seq_len_sec),
            "total_events": float(total_events),
            "patch_norm": args.patch_norm,
            "patch_norm_eps": args.patch_norm_eps,
            "patch_divider": args.patch_divider,
        }
        (out_dir / f"{plane}_meta.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
