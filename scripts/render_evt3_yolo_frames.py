import argparse
import re
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from ecfm.utils.evt3 import decode_evt3_raw, read_raw_header
from ecfm.utils.evt3_vis import draw_rectangles, events_to_image, write_image


_FRAME_RE = re.compile(r"_frame_(\d+)", re.IGNORECASE)


def _parse_geometry(meta: dict) -> Tuple[Optional[int], Optional[int]]:
    width = meta.get("width")
    height = meta.get("height")
    if width is not None and height is not None:
        try:
            return int(width), int(height)
        except ValueError:
            pass
    geometry = meta.get("geometry")
    if geometry and "x" in geometry:
        parts = geometry.lower().split("x", 1)
        try:
            return int(parts[0]), int(parts[1])
        except ValueError:
            return None, None
    return None, None


def _parse_label_time(path: Path) -> Optional[int]:
    match = _FRAME_RE.search(path.stem)
    if not match:
        return None
    return int(match.group(1))


def _read_ts_shift_us(raw_path: Path) -> Optional[int]:
    tmp_index = raw_path.with_suffix(raw_path.suffix + ".tmp_index")
    if not tmp_index.exists():
        return None
    _, _, meta = read_raw_header(tmp_index)
    shift = meta.get("ts_shift_us")
    if shift is None:
        return None
    try:
        return int(shift)
    except ValueError:
        return None


def _read_yolo_boxes(
    path: Path, width: int, height: int
) -> List[Tuple[int, int, int, int]]:
    boxes: List[Tuple[int, int, int, int]] = []
    if path.stat().st_size == 0:
        return boxes
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        _, cx, cy, bw, bh = parts[:5]
        try:
            cx = float(cx)
            cy = float(cy)
            bw = float(bw)
            bh = float(bh)
        except ValueError:
            continue
        x0 = (cx - bw / 2.0) * width
        y0 = (cy - bh / 2.0) * height
        x1 = (cx + bw / 2.0) * width
        y1 = (cy + bh / 2.0) * height
        boxes.append((int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))))
    return boxes


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Decode EVT3 .raw, render images only for non-empty YOLO label files, "
            "and draw target rectangles."
        )
    )
    parser.add_argument("raw", type=Path, help="Path to .raw file")
    parser.add_argument("yolo_dir", type=Path, help="Directory with YOLO txt files")
    parser.add_argument("output_dir", type=Path, help="Output directory for images")
    parser.add_argument(
        "--window",
        type=float,
        default=33333.0,
        help="Time window length in label units (default: 33333.0)",
    )
    parser.add_argument(
        "--label-unit",
        type=float,
        default=1.0,
        help="Label timestamp unit scale to seconds (default: 1.0 = microseconds)",
    )
    parser.add_argument(
        "--event-unit",
        type=float,
        default=1.0,
        help="Event timestamp unit scale to seconds (default: 1.0 = microseconds)",
    )
    parser.add_argument(
        "--ts-shift-us",
        type=float,
        default=None,
        help=(
            "Timestamp shift (microseconds) to add to event times. "
            "Defaults to ts_shift_us from .raw.tmp_index if available."
        ),
    )
    parser.add_argument(
        "--center",
        action="store_true",
        help="Center the time window on the label timestamp (default: start at label)",
    )
    parser.add_argument(
        "--endian",
        choices=["little", "big"],
        default="little",
        help="Byte order of EVT3 words (default: little)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Frame width override (defaults to header geometry)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Frame height override (defaults to header geometry)",
    )
    parser.add_argument(
        "--pixel-size",
        type=int,
        default=1,
        help="Event pixel size in output image (default: 1)",
    )
    parser.add_argument(
        "--rect-color",
        type=int,
        nargs=3,
        default=(0, 255, 0),
        help="Rectangle color as R G B (default: 0 255 0)",
    )
    parser.add_argument(
        "--rect-thickness",
        type=int,
        default=1,
        help="Rectangle thickness in pixels (default: 1)",
    )
    args = parser.parse_args()

    events, counters, meta, _ = decode_evt3_raw(args.raw, endian=args.endian)
    width, height = _parse_geometry(meta)
    if args.width is not None:
        width = args.width
    if args.height is not None:
        height = args.height
    if width is None or height is None:
        raise ValueError("Could not determine geometry; pass --width/--height.")

    t = events[:, 2].astype(np.float64) * float(args.event_unit)
    ts_shift_us = args.ts_shift_us
    if ts_shift_us is None:
        ts_shift_us = _read_ts_shift_us(args.raw)
    if ts_shift_us is not None:
        shift = float(ts_shift_us) * float(args.event_unit)
        t = t - shift
        keep = t >= 0
        if np.any(keep):
            events = events[keep]
            t = t[keep]
        else:
            events = events[:0]
            t = t[:0]
        print(f"Applied ts_shift_us={ts_shift_us} (dropped events before shift)")
    if np.any(np.diff(t) < 0):
        order = np.argsort(t, kind="stable")
        events = events[order]
        t = t[order]

    label_files = list(args.yolo_dir.glob("*.txt"))
    label_files.sort(key=lambda p: (_parse_label_time(p) is None, _parse_label_time(p) or 0, p.name))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for label_path in label_files:
        label_time_raw = _parse_label_time(label_path)
        if label_time_raw is None:
            continue
        label_time = float(label_time_raw) * float(args.label_unit)
        window = float(args.window) * float(args.label_unit)
        if args.center:
            t0 = label_time - window / 2.0
            t1 = label_time + window / 2.0
        else:
            t0 = label_time
            t1 = label_time + window

        idx0 = int(np.searchsorted(t, t0, side="left"))
        idx1 = int(np.searchsorted(t, t1, side="left"))
        ev = events[idx0:idx1]

        img = events_to_image(
            ev,
            width,
            height,
            pixel_size=args.pixel_size,
        )
        boxes = _read_yolo_boxes(label_path, width, height)
        if boxes:
            draw_rectangles(
                img,
                boxes,
                color=tuple(args.rect_color),
                thickness=args.rect_thickness,
            )
        out_path = args.output_dir / f"{label_path.stem}.png"
        final_path = write_image(out_path, img)
        print(f"Wrote {final_path}")

    if any(counters.values()):
        print("Ignored word types:")
        for k, v in counters.items():
            if v:
                print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
