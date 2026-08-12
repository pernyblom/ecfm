
import argparse
import re
import sys
from pathlib import Path

import cv2
import numpy as np


FRAME_RE = re.compile(r"^(\d+)\.png$", re.IGNORECASE)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Composite numbered PNG sequences from multiple folders. "
            "Folders are layered back-to-front in the order provided."
        )
    )

    parser.add_argument(
        "folders",
        nargs="+",
        type=Path,
        help=(
            "Input folders containing numbered PNG files. "
            "First folder is the back layer; last folder is the front layer."
        ),
    )

    parser.add_argument(
        "-o",
        "--output",
        required=True,
        type=Path,
        help="Output .mp4, .gif, or directory for composed PNG frames.",
    )

    parser.add_argument(
        "--fps",
        type=float,
        default=24.0,
        help="Output frame rate for MP4/GIF (default: 24).",
    )

    parser.add_argument(
        "--codec",
        default="mp4v",
        help="FourCC codec for MP4 output (default: mp4v).",
    )

    parser.add_argument(
        "--background",
        default="0,0,0",
        help=(
            "RGB background used when flattening transparency for MP4. "
            "Example: 255,255,255 for white (default: 0,0,0)."
        ),
    )

    parser.add_argument(
        "--missing",
        choices=("skip", "transparent", "error"),
        default="transparent",
        help=(
            "How to handle a frame missing from one or more folders: "
            "'transparent' ignores missing layers; "
            "'skip' skips the entire frame; "
            "'error' aborts. Default: transparent."
        ),
    )

    return parser.parse_args()


def parse_background(value):
    try:
        rgb = tuple(int(x.strip()) for x in value.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError(
            "Background must be three integers like 255,255,255"
        )

    if len(rgb) != 3 or any(x < 0 or x > 255 for x in rgb):
        raise argparse.ArgumentTypeError(
            "Background must be three integers from 0 to 255."
        )

    return rgb


def find_frames(folder):
    """Return {frame_number: path} for numbered PNGs in a folder."""
    result = {}

    if not folder.is_dir():
        raise FileNotFoundError(f"Input folder does not exist: {folder}")

    for path in folder.iterdir():
        if not path.is_file():
            continue

        match = FRAME_RE.match(path.name)
        if not match:
            continue

        frame_number = int(match.group(1))

        if frame_number in result:
            raise ValueError(
                f"Duplicate numeric frame {frame_number} in {folder}: "
                f"{result[frame_number].name} and {path.name}"
            )

        result[frame_number] = path

    return result


def load_rgba(path):
    """
    Load a PNG as BGRA uint8.

    OpenCV uses BGR/BGRA rather than RGB/RGBA.
    """
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)

    if image is None:
        raise RuntimeError(f"Could not read image: {path}")

    if image.ndim == 2:
        # Grayscale -> BGRA
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGRA)

    elif image.shape[2] == 3:
        # BGR -> BGRA, fully opaque
        image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)

    elif image.shape[2] == 4:
        pass

    else:
        raise RuntimeError(
            f"Unsupported image format for {path}: shape={image.shape}"
        )

    return image


def alpha_composite(bottom, top):
    """
    Alpha-composite BGRA uint8 arrays:
        result = top OVER bottom
    """
    if bottom.shape != top.shape:
        raise ValueError(
            f"Layer size mismatch: bottom={bottom.shape}, top={top.shape}"
        )

    # This case is very common for rendered frame sequences. In particular,
    # cv2.cvtColor gives RGB PNGs a completely opaque alpha channel. Avoiding
    # the float work below makes it essentially a pointer swap.
    top_alpha = top[..., 3]
    if cv2.countNonZero(top_alpha) == top_alpha.size:
        return top

    b = bottom.astype(np.float32) / 255.0
    t = top.astype(np.float32) / 255.0

    bottom_rgb = b[..., :3]
    bottom_a = b[..., 3:4]

    top_rgb = t[..., :3]
    top_a = t[..., 3:4]

    out_a = top_a + bottom_a * (1.0 - top_a)

    # Premultiplied calculation, then convert back to straight alpha.
    out_rgb_premult = (
        top_rgb * top_a
        + bottom_rgb * bottom_a * (1.0 - top_a)
    )

    out_rgb = np.divide(
        out_rgb_premult,
        out_a,
        out=np.zeros_like(out_rgb_premult),
        where=out_a > 0,
    )

    out = np.concatenate((out_rgb, out_a), axis=2)
    return np.clip(out * 255.0 + 0.5, 0, 255).astype(np.uint8)


def flatten_bgra(image, background_rgb):
    """Flatten BGRA image against a solid background and return BGR."""
    alpha_u8 = image[..., 3]
    if cv2.countNonZero(alpha_u8) == alpha_u8.size:
        # VideoWriter accepts this non-contiguous view; no conversion or copy
        # is needed for the overwhelmingly common opaque-frame case.
        return image[..., :3]

    rgb = np.array(background_rgb, dtype=np.float32)

    # OpenCV order.
    background_bgr = rgb[::-1]

    foreground = image[..., :3].astype(np.float32)
    alpha = image[..., 3:4].astype(np.float32) / 255.0

    result = (
        foreground * alpha
        + background_bgr.reshape(1, 1, 3) * (1.0 - alpha)
    )

    return np.clip(result + 0.5, 0, 255).astype(np.uint8)


def determine_canvas(layer_maps, frame_numbers):
    """Find the first available source image and use its dimensions."""
    for frame_number in frame_numbers:
        for layer_map in layer_maps:
            path = layer_map.get(frame_number)
            if path is not None:
                image = load_rgba(path)
                return image.shape[:2]  # height, width

    raise RuntimeError("No numbered PNG files found.")


def compose_frame(frame_number, layer_maps, height, width, missing_mode):
    paths = [layer_map.get(frame_number) for layer_map in layer_maps]
    missing_layers = [i for i, path in enumerate(paths) if path is None]

    # Decide this before decoding any PNGs.  Previously --missing=skip still
    # read and composited every available layer before throwing the result
    # away, and --missing=error did unnecessary work before failing.
    if missing_layers:
        if missing_mode == "error":
            indexes = ", ".join(str(i + 1) for i in missing_layers)
            raise RuntimeError(
                f"Frame {frame_number} is missing from folder layer(s): {indexes}"
            )

        if missing_mode == "skip":
            return None

    canvas = np.zeros((height, width, 4), dtype=np.uint8)

    for path in paths:
        if path is None:
            continue

        layer = load_rgba(path)

        if layer.shape[:2] != (height, width):
            raise ValueError(
                f"{path} has size {layer.shape[1]}x{layer.shape[0]}, "
                f"expected {width}x{height}"
            )

        canvas = alpha_composite(canvas, layer)

    return canvas


def output_kind(path):
    suffix = path.suffix.lower()

    if suffix == ".mp4":
        return "mp4"

    if suffix == ".gif":
        return "gif"

    return "frames"


def save_png_frames(
    output_dir,
    frame_numbers,
    layer_maps,
    height,
    width,
    missing_mode,
    padding,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    written = 0

    for frame_number in frame_numbers:
        frame = compose_frame(
            frame_number,
            layer_maps,
            height,
            width,
            missing_mode,
        )

        if frame is None:
            continue

        destination = output_dir / f"{frame_number:0{padding}d}.png"

        if not cv2.imwrite(str(destination), frame):
            raise RuntimeError(f"Could not write {destination}")

        written += 1

    return written


def save_mp4(
    output_path,
    frame_numbers,
    layer_maps,
    height,
    width,
    missing_mode,
    fps,
    codec,
    background_rgb,
):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if len(codec) != 4:
        raise ValueError("--codec must contain exactly four characters.")

    fourcc = cv2.VideoWriter_fourcc(*codec)

    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (width, height),
    )

    if not writer.isOpened():
        raise RuntimeError(
            f"Could not open MP4 writer for {output_path}. "
            f"Codec '{codec}' may not be available."
        )

    written = 0

    try:
        for frame_number in frame_numbers:
            frame = compose_frame(
                frame_number,
                layer_maps,
                height,
                width,
                missing_mode,
            )

            if frame is None:
                continue

            bgr = flatten_bgra(frame, background_rgb)
            writer.write(bgr)
            written += 1
    finally:
        writer.release()

    return written


def save_gif(
    output_path,
    frame_numbers,
    layer_maps,
    height,
    width,
    missing_mode,
    fps,
):
    # OpenCV does not provide reliable animated GIF encoding, so Pillow
    # is used only for this output format.
    try:
        from PIL import Image
    except ImportError:
        raise RuntimeError(
            "GIF output requires Pillow. Install it with:\n"
            "    pip install Pillow\n"
            "MP4 and PNG-sequence output only require OpenCV + NumPy."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    frames = []

    for frame_number in frame_numbers:
        bgra = compose_frame(
            frame_number,
            layer_maps,
            height,
            width,
            missing_mode,
        )

        if bgra is None:
            continue

        rgba = cv2.cvtColor(bgra, cv2.COLOR_BGRA2RGBA)
        frames.append(Image.fromarray(rgba))

    if not frames:
        raise RuntimeError("No frames were generated.")

    duration_ms = max(1, round(1000.0 / fps))

    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        disposal=2,
    )

    return len(frames)


def main():
    args = parse_args()

    if args.fps <= 0:
        print("Error: --fps must be greater than zero.", file=sys.stderr)
        return 2

    try:
        background_rgb = parse_background(args.background)

        layer_maps = [find_frames(folder) for folder in args.folders]

        # Union of all frame numbers. This permits layers to be absent on
        # individual frames when --missing=transparent.
        frame_numbers = sorted(
            set().union(*(layer_map.keys() for layer_map in layer_maps))
        )

        if not frame_numbers:
            raise RuntimeError("No numbered PNG files were found.")

        # Preserve enough zero-padding for the largest padding used by an
        # input filename, falling back to the number of digits required.
        padding = max(
            [
                len(path.stem)
                for layer_map in layer_maps
                for path in layer_map.values()
            ],
            default=1,
        )

        height, width = determine_canvas(layer_maps, frame_numbers)

        kind = output_kind(args.output)

        print(f"Layers: {len(args.folders)}")
        print(f"Candidate frames: {len(frame_numbers)}")
        print(f"Canvas: {width}x{height}")
        print(f"Output: {kind} -> {args.output}")

        if kind == "frames":
            written = save_png_frames(
                args.output,
                frame_numbers,
                layer_maps,
                height,
                width,
                args.missing,
                padding,
            )

        elif kind == "mp4":
            written = save_mp4(
                args.output,
                frame_numbers,
                layer_maps,
                height,
                width,
                args.missing,
                args.fps,
                args.codec,
                background_rgb,
            )

        elif kind == "gif":
            written = save_gif(
                args.output,
                frame_numbers,
                layer_maps,
                height,
                width,
                args.missing,
                args.fps,
            )

        else:
            raise AssertionError(kind)

        print(f"Wrote {written} frame(s).")
        return 0

    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
