import argparse
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from ecfm.utils.evt3 import decode_evt3_raw
from ecfm.utils.evt3_vis import render_event_frames, write_image


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Decode EVT3 .raw and render time-sliced event frames as images."
        )
    )
    parser.add_argument("input", type=Path, help="Path to .raw file")
    parser.add_argument("output_dir", type=Path, help="Directory for output images")
    parser.add_argument(
        "--dt",
        type=float,
        default=0.02,
        help="Frame duration in seconds (default: 0.02)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=100,
        help="Maximum number of frames to write (default: 100)",
    )
    parser.add_argument(
        "--time-unit",
        type=float,
        default=1e-6,
        help="Time unit to convert timestamps to seconds (default: 1e-6)",
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
    args = parser.parse_args()

    events, counters, meta, _ = decode_evt3_raw(
        args.input, endian=args.endian, require_evt3=True
    )

    width, height = _parse_geometry(meta)
    if args.width is not None:
        width = args.width
    if args.height is not None:
        height = args.height
    if width is None or height is None:
        raise ValueError("Could not determine geometry; pass --width/--height.")

    frames, ranges = render_event_frames(
        events,
        width,
        height,
        args.dt,
        time_unit=args.time_unit,
        max_frames=args.max_frames,
        pixel_size=args.pixel_size,
        sort_by_time=True,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for idx, img in enumerate(frames):
        out_path = args.output_dir / f"frame_{idx:05d}.png"
        final_path = write_image(out_path, img)
        t0, t1 = ranges[idx]
        print(f"Wrote {final_path}  [{t0:.6f}s, {t1:.6f}s)")

    if any(counters.values()):
        print("Ignored word types:")
        for k, v in counters.items():
            if v:
                print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
