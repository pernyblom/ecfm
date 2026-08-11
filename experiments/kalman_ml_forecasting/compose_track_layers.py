from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterator, Sequence

from PIL import Image


def parse_composition(value: str) -> list[str]:
    layers = [part.strip() for part in value.split(";") if part.strip()]
    if not layers:
        raise ValueError("Composition must contain at least one layer name.")
    return layers


def compose_frames(layer_frames: dict[str, Sequence[Image.Image]], order: Sequence[str]) -> list[Image.Image]:
    missing = [name for name in order if name not in layer_frames]
    if missing:
        raise ValueError(f"Composition references missing layers: {', '.join(missing)}")
    counts = {name: len(layer_frames[name]) for name in order}
    if len(set(counts.values())) != 1:
        raise ValueError(f"Composition layers have different frame counts: {counts}")
    result = []
    for index in range(next(iter(counts.values()))):
        result.append(compose_image_layers({name: layer_frames[name][index] for name in order}, order))
    return result


def compose_image_layers(layer_images: dict[str, Image.Image], order: Sequence[str]) -> Image.Image:
    missing = [name for name in order if name not in layer_images]
    if missing:
        raise ValueError(f"Composition references missing layers: {', '.join(missing)}")
    sizes = {name: layer_images[name].size for name in order}
    if len(set(sizes.values())) != 1:
        raise ValueError(
            "Cannot alpha-compose layers with different coordinate systems/resolutions: "
            f"{sizes}. Export projection layers such as xt/yt with --layer-rep and compare "
            "their matching numbered PNGs side by side."
        )
    base = layer_images[order[0]].convert("RGBA")
    for name in order[1:]:
        base.alpha_composite(layer_images[name].convert("RGBA"))
    return base


def load_layer_frames(layers_dir: Path, order: Sequence[str]) -> dict[str, list[Image.Image]]:
    loaded: dict[str, list[Image.Image]] = {}
    for name in order:
        directory = layers_dir / name
        paths = sorted(directory.glob("*.png")) if directory.is_dir() else []
        if not paths:
            raise FileNotFoundError(f"No PNG frames found for layer '{name}' in {directory}")
        loaded[name] = [Image.open(path).convert("RGBA") for path in paths]
    return loaded


def iter_composed_layer_files(layers_dir: Path, order: Sequence[str]) -> Iterator[Image.Image]:
    paths_by_layer: dict[str, list[Path]] = {}
    for name in order:
        directory = layers_dir / name
        paths = sorted(directory.glob("*.png")) if directory.is_dir() else []
        if not paths:
            raise FileNotFoundError(f"No PNG frames found for layer '{name}' in {directory}")
        paths_by_layer[name] = paths
    counts = {name: len(paths) for name, paths in paths_by_layer.items()}
    if len(set(counts.values())) != 1:
        raise ValueError(f"Composition layers have different frame counts: {counts}")
    for index in range(next(iter(counts.values()))):
        opened: dict[str, Image.Image] = {}
        try:
            for name in order:
                opened[name] = Image.open(paths_by_layer[name][index]).convert("RGBA")
            yield compose_image_layers(opened, order)
        finally:
            for image in opened.values():
                image.close()


def write_gif_iter(frames: Iterator[Image.Image], output: Path, duration_ms: int) -> None:
    import itertools
    import shutil
    import subprocess

    ffmpeg_exe = shutil.which("ffmpeg")
    if ffmpeg_exe is None:
        try:
            import imageio_ffmpeg
        except ImportError as exc:
            raise RuntimeError("Streaming GIF output requires FFmpeg or imageio-ffmpeg.") from exc
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

    output.parent.mkdir(parents=True, exist_ok=True)
    iterator = iter(frames)
    try:
        first = next(iterator)
    except StopIteration:
        return
    width, height = first.size
    fps = 1000.0 / duration_ms
    command = [
        ffmpeg_exe, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgba",
        "-s", f"{width}x{height}", "-r", str(fps), "-i", "-", "-loop", "0", str(output),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    assert process.stdin is not None
    try:
        for frame in itertools.chain((first,), iterator):
            try:
                process.stdin.write(frame.convert("RGBA").tobytes())
            finally:
                frame.close()
        process.stdin.close()
        stderr = process.stderr.read() if process.stderr is not None else b""
        return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(f"FFmpeg GIF encoding failed: {stderr.decode(errors='replace')}")
    finally:
        if process.poll() is None:
            process.kill()


def write_mp4_iter(frames: Iterator[Image.Image], output: Path, fps: float) -> None:
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("Writing MP4 requires opencv-python.") from exc
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    try:
        for frame in frames:
            try:
                rgb = np.asarray(frame.convert("RGB"))
                if writer is None:
                    height, width = rgb.shape[:2]
                    writer = cv2.VideoWriter(
                        str(output), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height)
                    )
                    if not writer.isOpened():
                        raise RuntimeError(f"Failed to open MP4 writer for {output}")
                writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            finally:
                frame.close()
    finally:
        if writer is not None:
            writer.release()


def write_gif(frames: Sequence[Image.Image], output: Path, duration_ms: int) -> None:
    if not frames:
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        output, save_all=True, append_images=list(frames[1:]), duration=duration_ms,
        loop=0, disposal=2,
    )


def write_mp4(frames: Sequence[Image.Image], output: Path, fps: float) -> None:
    if not frames:
        return
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("Writing MP4 requires opencv-python.") from exc
    output.parent.mkdir(parents=True, exist_ok=True)
    width, height = frames[0].size
    writer = cv2.VideoWriter(
        str(output), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open MP4 writer for {output}")
    try:
        for frame in frames:
            rgb = np.asarray(frame.convert("RGB"))
            writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compose previously rendered visualize_tracks.py PNG layers."
    )
    parser.add_argument("--layers-dir", type=Path, required=True)
    parser.add_argument("--composition", required=True, help="Semicolon-separated back-to-front layer names.")
    parser.add_argument("--output-gif", type=Path)
    parser.add_argument("--output-mp4", type=Path)
    parser.add_argument("--duration-ms", type=int, default=120)
    parser.add_argument("--fps", type=float, default=None)
    args = parser.parse_args()
    if args.output_gif is None and args.output_mp4 is None:
        parser.error("Pass --output-gif and/or --output-mp4.")
    order = parse_composition(args.composition)
    if args.output_gif is not None:
        write_gif_iter(iter_composed_layer_files(args.layers_dir, order), args.output_gif, args.duration_ms)
    if args.output_mp4 is not None:
        write_mp4_iter(
            iter_composed_layer_files(args.layers_dir, order),
            args.output_mp4,
            args.fps or 1000.0 / args.duration_ms,
        )


if __name__ == "__main__":
    main()
