import argparse
import re
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from ecfm.utils.evt3 import decode_evt3_raw, read_raw_header
from ecfm.data.tokenizer import Region, build_patch
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
    path: Path,
) -> List[Tuple[float, float, float, float]]:
    boxes: List[Tuple[float, float, float, float]] = []
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
        boxes.append((cx, cy, bw, bh))
    return boxes


def _project_boxes(
    boxes: List[Tuple[float, float, float, float]],
    *,
    dst_w: int,
    dst_h: int,
) -> List[Tuple[int, int, int, int]]:
    if not boxes:
        return []
    projected: List[Tuple[int, int, int, int]] = []
    for cx, cy, bw, bh in boxes:
        x0 = (cx - bw / 2.0) * dst_w
        y0 = (cy - bh / 2.0) * dst_h
        x1 = (cx + bw / 2.0) * dst_w
        y1 = (cy + bh / 2.0) * dst_h
        projected.append((int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))))
    return projected


def _patch_to_rgb(patch: np.ndarray, *, time_horizontal: bool = False) -> np.ndarray:
    if patch.ndim != 3 or patch.shape[0] != 2:
        raise ValueError("patch must be shaped [2, H, W]")
    # If time is horizontal, swap (T, Y) -> (Y, T) so time runs left->right.
    if time_horizontal:
        patch = patch.transpose(0, 2, 1)
    p0 = np.clip(patch[0], 0.0, 1.0)
    p1 = np.clip(patch[1], 0.0, 1.0)
    h, w = p0.shape
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :, 0] = (p1 * 255.0).astype(np.uint8)
    img[:, :, 2] = (p0 * 255.0).astype(np.uint8)
    return img


def _render_histogram_grid(
    events: np.ndarray,
    *,
    width: int,
    height: int,
    t0: float,
    dt: float,
    plane: str,
    time_bins: int,
    patch_size: int,
    grid_x: int,
    grid_y: int,
) -> np.ndarray:
    out_h = patch_size * grid_y
    out_w = patch_size * grid_x
    canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    x_edges = np.linspace(0, width, grid_x + 1, dtype=np.int64)
    y_edges = np.linspace(0, height, grid_y + 1, dtype=np.int64)

    for gy in range(grid_y):
        for gx in range(grid_x):
            x0 = int(x_edges[gx])
            x1 = int(x_edges[gx + 1])
            y0 = int(y_edges[gy])
            y1 = int(y_edges[gy + 1])
            dx = max(1, x1 - x0)
            dy = max(1, y1 - y0)
            region = Region(
                x=x0,
                y=y0,
                t=t0,
                dx=dx,
                dy=dy,
                dt=dt,
                plane=plane,
            )
            patch_t, _ = build_patch(
                events,
                region,
                patch_size=patch_size,
                time_bins=time_bins,
            )
            patch = patch_t.detach().cpu().numpy()
            time_horizontal = plane.startswith("yt")
            patch_img = _patch_to_rgb(patch, time_horizontal=time_horizontal)
            y1 = (gy + 1) * patch_size
            x1 = (gx + 1) * patch_size
            canvas[gy * patch_size : y1, gx * patch_size : x1] = patch_img
    return canvas


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
        "--representation",
        type=str,
        default="events",
        help=(
            "Representation(s) to render, separated by ';' (default: events). "
            "Options: events, xy, xt, yt, xy_p45, xy_m45, yt_p45, yt_m45."
        ),
    )
    parser.add_argument(
        "--temporal-bins",
        type=int,
        default=64,
        help="Temporal bins for histogram representations (default: 64)",
    )
    parser.add_argument(
        "--spatial-bins",
        type=int,
        default=32,
        help="Output patch size for histogram representations (default: 32)",
    )
    parser.add_argument(
        "--grid-x",
        type=int,
        default=1,
        help="Number of grid cells in x (default: 1)",
    )
    parser.add_argument(
        "--grid-y",
        type=int,
        default=1,
        help="Number of grid cells in y (default: 1)",
    )
    parser.add_argument(
        "--rect-color",
        type=int,
        nargs=3,
        default=(0, 255, 0),
        help="Rectangle color as R G B (default: 0 255 0)",
    )
    parser.add_argument(
        "--draw-rectangles",
        action="store_true",
        default=True,
        help="Draw rectangles from YOLO labels (default: True)",
    )
    parser.add_argument(
        "--no-rectangles",
        action="store_false",
        dest="draw_rectangles",
        help="Disable drawing rectangles from YOLO labels",
    )
    parser.add_argument(
        "--rect-thickness",
        type=int,
        default=1,
        help="Rectangle thickness in pixels (default: 1)",
    )
    parser.add_argument(
        "--only-with-rects",
        action="store_true",
        default=True,
        help="Only write images that have at least one rectangle (default: True)",
    )
    parser.add_argument(
        "--include-empty",
        action="store_false",
        dest="only_with_rects",
        help="Also write images with zero rectangles",
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
        ev_time = ev
        if ev.size:
            ev_time = ev.copy()
            ev_time[:, 2] = t[idx0:idx1]

        boxes = _read_yolo_boxes(label_path)
        if args.only_with_rects and not boxes:
            continue
        reps_raw = args.representation.replace(",", ";")
        representations = [r.strip() for r in reps_raw.split(";") if r.strip()]
        valid = {
            "events",
            "xy",
            "xt",
            "yt",
            "xy_p45",
            "xy_m45",
            "yt_p45",
            "yt_m45",
        }
        for rep in representations:
            if rep not in valid:
                raise ValueError(f"Unknown representation: {rep}")
            if rep == "events":
                if args.grid_x == 1 and args.grid_y == 1:
                    img = events_to_image(
                        ev,
                        width,
                        height,
                        pixel_size=args.pixel_size,
                    )
                else:
                    img = _render_histogram_grid(
                        ev_time,
                        width=width,
                        height=height,
                        t0=t0,
                        dt=t1 - t0,
                        plane="xy",
                        time_bins=1,
                        patch_size=args.spatial_bins,
                        grid_x=args.grid_x,
                        grid_y=args.grid_y,
                    )
            else:
                img = _render_histogram_grid(
                    ev_time,
                    width=width,
                    height=height,
                    t0=t0,
                    dt=t1 - t0,
                    plane=rep,
                    time_bins=args.temporal_bins,
                    patch_size=args.spatial_bins,
                    grid_x=args.grid_x,
                    grid_y=args.grid_y,
                )
        scaled_boxes = _project_boxes(
            boxes,
            dst_w=img.shape[1],
            dst_h=img.shape[0],
        )
        if args.draw_rectangles and scaled_boxes:
            draw_rectangles(
                img,
                scaled_boxes,
                color=tuple(args.rect_color),
                thickness=args.rect_thickness,
            )
        out_path = args.output_dir / f"{label_path.stem}_{rep}.png"
        final_path = write_image(out_path, img)
        print(f"Wrote {final_path}")

    if any(counters.values()):
        print("Ignored word types:")
        for k, v in counters.items():
            if v:
                print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
