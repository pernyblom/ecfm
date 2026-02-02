import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.render_evt3_yolo_frames import render_yolo_frames


def _read_split_file(path: Path) -> list[str]:
    items = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        items.append(line.strip("/"))
    return items


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch render EVT3 representations for multiple FRED folders."
    )
    parser.add_argument(
        "--fred-root",
        type=Path,
        default=Path("datasets/FRED"),
        help="Root path of FRED dataset (default: datasets/FRED)",
    )
    parser.add_argument(
        "--split-file",
        type=Path,
        required=True,
        help="Split file with folder IDs (e.g., datasets/FRED/dataset_splits/canonical/test_split.txt)",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/fred_reps"),
        help="Output root directory (default: outputs/fred_reps)",
    )
    parser.add_argument(
        "--representation",
        type=str,
        default="events",
        help="Representation(s) to render, separated by ';' (default: events).",
    )
    parser.add_argument(
        "--crop-representations",
        type=str,
        default="",
        help="Representations to crop to YOLO boxes, separated by ';' (default: none).",
    )
    parser.add_argument(
        "--rgb-dir",
        type=Path,
        default=None,
        help="Optional RGB image directory (defaults to sibling RGB or PADDED_RGB)",
    )
    parser.add_argument("--window", type=float, default=33333.0)
    parser.add_argument("--label-unit", type=float, default=1.0)
    parser.add_argument("--event-unit", type=float, default=1.0)
    parser.add_argument("--ts-shift-us", type=float, default=None)
    parser.add_argument("--center", action="store_true")
    parser.add_argument("--endian", choices=["little", "big"], default="little")
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--pixel-size", type=int, default=1)
    parser.add_argument("--temporal-bins", type=int, default=224)
    parser.add_argument("--spatial-bins", type=int, default=224)
    parser.add_argument("--output-size", type=int, nargs=2, default=None)
    parser.add_argument("--grid-x", type=int, default=1)
    parser.add_argument("--grid-y", type=int, default=1)
    parser.add_argument("--rect-color", type=int, nargs=3, default=(0, 255, 0))
    parser.add_argument("--draw-rectangles", action="store_true", default=False)
    parser.add_argument("--no-rectangles", action="store_false", dest="draw_rectangles")
    parser.add_argument("--rect-thickness", type=int, default=1)
    parser.add_argument("--only-with-rects", action="store_true", default=True)
    parser.add_argument("--include-empty", action="store_false", dest="only_with_rects")
    parser.add_argument(
        "--transform",
        type=str,
        default="none",
        choices=["none", "spectrogram", "spectrum2d", "spectrogram_bin", "spectrum2d_bin"],
    )
    parser.add_argument(
        "--transform-scale",
        type=str,
        default="linear",
        choices=["linear", "log", "db"],
    )
    parser.add_argument("--transform-eps", type=float, default=1e-6)
    args = parser.parse_args()

    folders = _read_split_file(args.split_file)
    for folder in folders:
        base = args.fred_root / folder
        raw_path = base / "Event" / "events.raw"
        yolo_dir = base / "Event_YOLO"
        if not raw_path.exists() or not yolo_dir.exists():
            print(f"Skipping {folder}: missing raw or labels.")
            continue
        out_dir = args.output_root / folder
        per_args = argparse.Namespace(**vars(args))
        per_args.raw = raw_path
        per_args.yolo_dir = yolo_dir
        per_args.output_dir = out_dir
        print(f"Rendering {folder} -> {out_dir}")
        render_yolo_frames(per_args)


if __name__ == "__main__":
    main()
