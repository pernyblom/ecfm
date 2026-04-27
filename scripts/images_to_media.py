from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Literal

import numpy as np
from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _image_paths(image_dir: Path, extensions: Iterable[str]) -> List[Path]:
    allowed = {ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in extensions}
    return sorted(
        (
            path
            for path in image_dir.iterdir()
            if path.is_file()
            and not path.name.startswith("._")
            and path.suffix.lower() in allowed
        ),
        key=lambda path: path.name,
    )


def _select_frames(paths: List[Path], *, start_index: int, frame_count: int | None) -> List[Path]:
    if start_index < 0:
        raise ValueError(f"--start-index must be >= 0, got {start_index}")
    if frame_count is not None and frame_count < 1:
        raise ValueError(f"--frame-count must be >= 1 when set, got {frame_count}")
    selected = paths[start_index:] if frame_count is None else paths[start_index : start_index + frame_count]
    if not selected:
        raise ValueError(
            f"No frames selected from {len(paths)} image(s). "
            "Check --start-index and --frame-count."
        )
    return selected


def _infer_format(output_path: Path, requested: str) -> Literal["mp4", "gif"]:
    if requested != "auto":
        return requested  # type: ignore[return-value]
    suffix = output_path.suffix.lower()
    if suffix == ".gif":
        return "gif"
    if suffix == ".mp4":
        return "mp4"
    raise ValueError("Could not infer output format from extension. Use --format mp4 or --format gif.")


def _open_rgb(path: Path, size: tuple[int, int] | None = None) -> Image.Image:
    img = Image.open(path).convert("RGB")
    if size is not None and img.size != size:
        img = img.resize(size, resample=Image.BILINEAR)
    return img


def _write_mp4(frame_paths: List[Path], output_path: Path, fps: float) -> None:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("Writing MP4 requires opencv-python because this project uses cv2.VideoWriter.") from exc

    first = _open_rgb(frame_paths[0])
    frame_w, frame_h = first.size
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (frame_w, frame_h))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter for output {output_path}")

    try:
        for frame_path in frame_paths:
            frame = _open_rgb(frame_path, size=(frame_w, frame_h))
            frame_bgr = cv2.cvtColor(np.asarray(frame, dtype=np.uint8), cv2.COLOR_RGB2BGR)
            writer.write(frame_bgr)
    finally:
        writer.release()


def _write_gif(frame_paths: List[Path], output_path: Path, fps: float) -> None:
    first = _open_rgb(frame_paths[0])
    frame_w, frame_h = first.size
    frames = [first]
    frames.extend(_open_rgb(path, size=(frame_w, frame_h)) for path in frame_paths[1:])
    duration_ms = max(1, int(round(1000.0 / fps)))
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a name-sorted image folder slice to an MP4 video or GIF."
    )
    parser.add_argument("--image-dir", type=Path, required=True, help="Folder containing input images.")
    parser.add_argument("--output", type=Path, required=True, help="Output .mp4 or .gif path.")
    parser.add_argument(
        "--format",
        choices=["auto", "mp4", "gif"],
        default="auto",
        help="Output format. Defaults to inferring from --output extension.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Zero-based start index after sorting images by filename.",
    )
    parser.add_argument(
        "--frame-count",
        type=int,
        default=None,
        help="Number of frames to include. Defaults to all remaining images.",
    )
    parser.add_argument("--fps", type=float, default=30.0, help="Target frames per second.")
    parser.add_argument(
        "--extensions",
        type=str,
        default=",".join(sorted(IMAGE_EXTENSIONS)),
        help="Comma-separated image extensions to include.",
    )
    args = parser.parse_args()

    if args.fps <= 0:
        raise ValueError(f"--fps must be > 0, got {args.fps}")
    if not args.image_dir.exists() or not args.image_dir.is_dir():
        raise FileNotFoundError(f"Missing image directory: {args.image_dir}")

    extensions = [item.strip() for item in str(args.extensions).split(",") if item.strip()]
    paths = _image_paths(args.image_dir, extensions)
    if not paths:
        raise FileNotFoundError(f"No images found in {args.image_dir} with extensions {extensions}")
    frame_paths = _select_frames(paths, start_index=args.start_index, frame_count=args.frame_count)

    output_format = _infer_format(args.output, args.format)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if output_format == "mp4":
        _write_mp4(frame_paths, args.output, float(args.fps))
    else:
        _write_gif(frame_paths, args.output, float(args.fps))

    print(
        f"Wrote {args.output} from {len(frame_paths)} frame(s) "
        f"starting at sorted index {args.start_index} at {float(args.fps):.3f} fps"
    )


if __name__ == "__main__":
    main()
