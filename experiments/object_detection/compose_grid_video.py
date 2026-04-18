from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import List

import cv2
import numpy as np
from PIL import Image


def _find_frame_files(directory: Path) -> List[Path]:
    files = sorted(directory.glob("*.png"))
    if not files:
        raise FileNotFoundError(f"No PNG frames found in {directory}")
    return files


def _fit_rgb_tile(img: Image.Image, tile_w: int, tile_h: int) -> Image.Image:
    canvas = Image.new("RGB", (tile_w, tile_h), (0, 0, 0))
    scale = min(tile_w / img.width, tile_h / img.height)
    new_w = max(1, int(round(img.width * scale)))
    new_h = max(1, int(round(img.height * scale)))
    resized = img.resize((new_w, new_h), resample=Image.BILINEAR)
    x = (tile_w - new_w) // 2
    y = (tile_h - new_h) // 2
    canvas.paste(resized, (x, y))
    return canvas


def _load_tile(path: Path, tile_w: int, tile_h: int, *, keep_aspect: bool) -> Image.Image:
    img = Image.open(path).convert("RGB")
    if keep_aspect:
        return _fit_rgb_tile(img, tile_w, tile_h)
    if img.size != (tile_w, tile_h):
        img = img.resize((tile_w, tile_h), resample=Image.BILINEAR)
    return img


def _compose_frame(
    *,
    cstr3_path: Path,
    yt_path: Path,
    xt_path: Path,
    rgb_path: Path,
    tile_w: int,
    tile_h: int,
) -> np.ndarray:
    top_left = _load_tile(cstr3_path, tile_w, tile_h, keep_aspect=False)
    top_right = _load_tile(yt_path, tile_w, tile_h, keep_aspect=False)
    bottom_left = _load_tile(xt_path, tile_w, tile_h, keep_aspect=False)
    bottom_right = _load_tile(rgb_path, tile_w, tile_h, keep_aspect=True)

    canvas = Image.new("RGB", (tile_w * 2, tile_h * 2), (0, 0, 0))
    canvas.paste(top_left, (0, 0))
    canvas.paste(top_right, (tile_w, 0))
    canvas.paste(bottom_left, (0, tile_h))
    canvas.paste(bottom_right, (tile_w, tile_h))
    return np.asarray(canvas, dtype=np.uint8)


def _find_sequence_assets(sequence_dir: Path) -> dict[str, List[Path]]:
    assets = {}
    mapping = {
        "cstr3": sequence_dir / "frames_cstr3",
        "yt_mx": sequence_dir / "frames_yt_mx",
        "xt_my": sequence_dir / "frames_xt_my",
        "rgb": sequence_dir / "frames_rgb",
    }
    for key, directory in mapping.items():
        if not directory.exists():
            raise FileNotFoundError(
                f"Missing frame directory for {key}: {directory}. "
                "The sequence renderer should be run with frame saving enabled."
            )
        assets[key] = _find_frame_files(directory)
    return assets


def _require_binary(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise RuntimeError(f"Required binary '{name}' was not found on PATH.")
    return path


def _probe_video_fps(ffprobe: str, video_path: Path) -> float | None:
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=r_frame_rate,avg_frame_rate",
        "-of",
        "json",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return None
    payload = json.loads(result.stdout)
    streams = payload.get("streams", [])
    if not streams:
        return None
    for key in ("avg_frame_rate", "r_frame_rate"):
        raw = streams[0].get(key)
        if not raw:
            continue
        if "/" in raw:
            num_s, den_s = raw.split("/", 1)
            try:
                num = float(num_s)
                den = float(den_s)
            except ValueError:
                continue
            if den != 0:
                return num / den
        else:
            try:
                return float(raw)
            except ValueError:
                continue
    return None


def _infer_fps(sequence_dir: Path, ffprobe: str, default: float = 30.0) -> float:
    prefix = sequence_dir.name
    for rep in ("cstr3", "yt_mx", "xt_my", "rgb"):
        video_path = sequence_dir / f"{prefix}_{rep}.mp4"
        if not video_path.exists():
            continue
        fps = _probe_video_fps(ffprobe, video_path)
        if fps:
            return float(fps)
    return default


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sequence-dir",
        type=Path,
        required=True,
        help="Folder produced by render_sequence_video.py, e.g. outputs/object_detection_sequence_videos/8",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--tile-width",
        type=int,
        default=None,
        help="Optional tile width override. Defaults to the cstr3 frame width.",
    )
    parser.add_argument(
        "--tile-height",
        type=int,
        default=None,
        help="Optional tile height override. Defaults to the cstr3 frame height.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Optional output fps override. If omitted, infer from one of the generated MP4s when available.",
    )
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()

    sequence_dir = args.sequence_dir
    if not sequence_dir.exists():
        raise FileNotFoundError(sequence_dir)
    ffprobe = _require_binary("ffprobe")

    assets = _find_sequence_assets(sequence_dir)
    lengths = {key: len(paths) for key, paths in assets.items()}
    num_frames = min(lengths.values())
    if args.max_frames is not None:
        num_frames = min(num_frames, int(args.max_frames))
    if num_frames <= 0:
        raise RuntimeError(f"No frames available in {sequence_dir}")

    first_cstr3 = Image.open(assets["cstr3"][0]).convert("RGB")
    tile_w = int(args.tile_width) if args.tile_width is not None else first_cstr3.width
    tile_h = int(args.tile_height) if args.tile_height is not None else first_cstr3.height
    fps = float(args.fps) if args.fps is not None else _infer_fps(sequence_dir, ffprobe)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_w = tile_w * 2
    out_h = tile_h * 2
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(args.output), fourcc, fps, (out_w, out_h))
    if not writer.isOpened():
        raise RuntimeError(
            "Failed to open VideoWriter for output. "
            "If this persists, try a different output filename or container."
        )
    try:
        for idx in range(num_frames):
            frame = _compose_frame(
                cstr3_path=assets["cstr3"][idx],
                yt_path=assets["yt_mx"][idx],
                xt_path=assets["xt_my"][idx],
                rgb_path=assets["rgb"][idx],
                tile_w=tile_w,
                tile_h=tile_h,
            )
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            writer.write(frame_bgr)
            if (idx + 1) % 100 == 0:
                print(f"composed {idx + 1}/{num_frames} frames")
    finally:
        writer.release()

    print(
        f"Wrote {args.output} from {sequence_dir} with tile size {tile_w}x{tile_h} "
        f"and output size {out_w}x{out_h} at {fps:.3f} fps"
    )


if __name__ == "__main__":
    main()
