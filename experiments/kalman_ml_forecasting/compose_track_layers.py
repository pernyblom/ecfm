from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

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
        base = layer_frames[order[0]][index].convert("RGBA")
        for name in order[1:]:
            base.alpha_composite(layer_frames[name][index].convert("RGBA"))
        result.append(base)
    return result


def load_layer_frames(layers_dir: Path, order: Sequence[str]) -> dict[str, list[Image.Image]]:
    loaded: dict[str, list[Image.Image]] = {}
    for name in order:
        directory = layers_dir / name
        paths = sorted(directory.glob("*.png")) if directory.is_dir() else []
        if not paths:
            raise FileNotFoundError(f"No PNG frames found for layer '{name}' in {directory}")
        loaded[name] = [Image.open(path).convert("RGBA") for path in paths]
    return loaded


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
    frames = compose_frames(load_layer_frames(args.layers_dir, order), order)
    if args.output_gif is not None:
        write_gif(frames, args.output_gif, args.duration_ms)
    if args.output_mp4 is not None:
        write_mp4(frames, args.output_mp4, args.fps or 1000.0 / args.duration_ms)


if __name__ == "__main__":
    main()
