from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.object_detection.data.dataset import _parse_frame_time, _read_yolo_boxes
from experiments.object_detection.models.factory import build_model
from experiments.object_detection.utils.config import load_config

_RGB_TIME_RE = re.compile(r"_(\d{2})_(\d{2})_(\d{2})\.(\d+)$")


def _parse_rep_list(raw: str) -> List[str]:
    return [item.strip() for item in raw.replace(",", ";").split(";") if item.strip()]


def _load_input_tensor(path: Path, image_size: Tuple[int, int]) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    if img.size != (image_size[0], image_size[1]):
        img = img.resize(image_size, resample=Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def _find_frame_stems(
    *,
    images_dir: Path,
    labels_dir: Path,
    required_reps: List[str],
) -> List[str]:
    stems: List[str] = []
    for label_path in sorted(labels_dir.glob("*.txt"), key=lambda p: (_parse_frame_time(p.stem) or -1, p.name)):
        stem = label_path.stem
        if all(rep.lower() == "rgb" or (images_dir / f"{stem}_{rep}.png").exists() for rep in required_reps):
            stems.append(stem)
    return stems


def _parse_rgb_time(path: Path) -> Optional[float]:
    match = _RGB_TIME_RE.search(path.stem)
    if not match:
        return None
    hh, mm, ss, frac = match.groups()
    try:
        h = int(hh)
        m = int(mm)
        s = int(ss)
        micros = int(frac.ljust(6, "0")[:6])
    except ValueError:
        return None
    return h * 3600.0 + m * 60.0 + s + micros / 1_000_000.0


def _build_rgb_index(rgb_dir: Path, *, label_time_unit: float) -> List[Tuple[float, Path]]:
    if not rgb_dir.exists():
        return []
    files = sorted(rgb_dir.glob("*.jpg")) + sorted(rgb_dir.glob("*.png"))
    if not files:
        return []
    parsed = [_parse_rgb_time(p) for p in files]
    out: List[Tuple[float, Path]] = []
    if any(t is not None for t in parsed):
        base = next(t for t in parsed if t is not None)
        for path, t in zip(files, parsed):
            if t is None:
                continue
            rel_us = (t - base) * 1_000_000.0
            out.append((rel_us * float(label_time_unit), path))
    else:
        for idx, path in enumerate(files):
            out.append((float(idx), path))
    out.sort(key=lambda item: item[0])
    return out


def _find_rgb_frame(rgb_index: List[Tuple[float, Path]], label_time_s: float) -> Optional[Path]:
    if not rgb_index:
        return None
    times = [t for t, _ in rgb_index]
    idx = int(np.searchsorted(times, label_time_s, side="left"))
    if idx <= 0:
        return rgb_index[0][1]
    if idx >= len(rgb_index):
        return rgb_index[-1][1]
    before_t, before_p = rgb_index[idx - 1]
    after_t, after_p = rgb_index[idx]
    return before_p if abs(label_time_s - before_t) <= abs(after_t - label_time_s) else after_p


def _load_background_image(
    *,
    rep: str,
    stem: str,
    label_time_s: float,
    images_dir: Path,
    dataset_folder_dir: Path,
    label_time_unit: float,
    rgb_index: Optional[List[Tuple[float, Path]]],
) -> Image.Image:
    rendered_path = images_dir / f"{stem}_{rep}.png"
    if rendered_path.exists():
        return Image.open(rendered_path).convert("RGB")
    if rep.lower() != "rgb":
        raise FileNotFoundError(f"Missing rendered background image: {rendered_path}")
    if rgb_index:
        rgb_path = _find_rgb_frame(rgb_index, label_time_s)
        if rgb_path is not None and rgb_path.exists():
            return Image.open(rgb_path).convert("RGB")
    for rgb_dir_name in ("RGB", "PADDED_RGB"):
        candidate_dir = dataset_folder_dir / rgb_dir_name
        local_index = _build_rgb_index(candidate_dir, label_time_unit=label_time_unit)
        rgb_path = _find_rgb_frame(local_index, label_time_s)
        if rgb_path is not None and rgb_path.exists():
            return Image.open(rgb_path).convert("RGB")
    raise FileNotFoundError(
        f"Could not resolve rgb background for stem {stem}. Checked rendered output and dataset RGB directories."
    )


def _box_to_xyxy(box: torch.Tensor, size: Tuple[int, int]) -> Tuple[float, float, float, float]:
    w, h = float(size[0]), float(size[1])
    cx, cy, bw, bh = [float(v) for v in box]
    return (
        (cx - bw / 2.0) * w,
        (cy - bh / 2.0) * h,
        (cx + bw / 2.0) * w,
        (cy + bh / 2.0) * h,
    )


def _draw_label(draw: ImageDraw.ImageDraw, x: float, y: float, text: str, color: Tuple[int, int, int]) -> None:
    x = max(2.0, x)
    y = max(2.0, y)
    bbox = draw.textbbox((x, y), text)
    draw.rectangle(bbox, fill=(0, 0, 0))
    draw.text((x, y), text, fill=color)


def _draw_pred_overlay(
    img: Image.Image,
    *,
    rep: str,
    pred_box: torch.Tensor,
    pred_score: float,
    score_threshold: float,
    gt_boxes: Iterable[Tuple[float, float, float, float]] | None = None,
    draw_gt: bool = False,
) -> Image.Image:
    out = img.convert("RGB").copy()
    if rep.lower() == "rgb":
        return out
    draw = ImageDraw.Draw(out)
    w, h = out.size

    if draw_gt and gt_boxes is not None:
        for cx, cy, bw, bh in gt_boxes:
            if rep == "xt_my":
                x0 = (cx - bw / 2.0) * w
                x1 = (cx + bw / 2.0) * w
                draw.rectangle([x0, 0, x1, h - 1], outline=(0, 255, 0), width=2)
            elif rep == "yt_mx":
                y0 = (cy - bh / 2.0) * h
                y1 = (cy + bh / 2.0) * h
                draw.rectangle([0, y0, w - 1, y1], outline=(0, 255, 0), width=2)
            else:
                draw.rectangle(_box_to_xyxy(torch.tensor([cx, cy, bw, bh]), (w, h)), outline=(0, 255, 0), width=2)

    if pred_score >= score_threshold:
        if rep == "xt_my":
            x0 = (float(pred_box[0]) - float(pred_box[2]) / 2.0) * w
            x1 = (float(pred_box[0]) + float(pred_box[2]) / 2.0) * w
            draw.rectangle([x0, 0, x1, h - 1], outline=(255, 196, 0), width=2)
            _draw_label(draw, x0 + 2.0, 2.0, f"score {pred_score:.3f}", (255, 196, 0))
        elif rep == "yt_mx":
            y0 = (float(pred_box[1]) - float(pred_box[3]) / 2.0) * h
            y1 = (float(pred_box[1]) + float(pred_box[3]) / 2.0) * h
            draw.rectangle([0, y0, w - 1, y1], outline=(255, 196, 0), width=2)
            _draw_label(draw, 2.0, y0 + 2.0, f"score {pred_score:.3f}", (255, 196, 0))
        else:
            x0, y0, x1, y1 = _box_to_xyxy(pred_box, (w, h))
            draw.rectangle([x0, y0, x1, y1], outline=(255, 196, 0), width=2)
            _draw_label(draw, x0 + 2.0, y0 - 14.0, f"score {pred_score:.3f}", (255, 196, 0))

    return out


def _infer_fps(stems: List[str], fallback: float = 30.0) -> float:
    times = [t for t in (_parse_frame_time(stem) for stem in stems) if t is not None]
    if len(times) < 2:
        return fallback
    diffs = [b - a for a, b in zip(times[:-1], times[1:]) if b > a]
    if not diffs:
        return fallback
    median_us = float(np.median(np.asarray(diffs, dtype=np.float64)))
    if median_us <= 0:
        return fallback
    return 1_000_000.0 / median_us


def _write_video_ffmpeg(frames_pattern: Path, output_path: Path, fps: float) -> bool:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return False
    cmd = [
        ffmpeg,
        "-y",
        "-framerate",
        f"{fps:.6f}",
        "-i",
        str(frames_pattern),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--folder", type=str, required=True, help="FRED sequence folder, e.g. 8")
    parser.add_argument(
        "--reps",
        type=str,
        default="cstr3",
        help="Representations to render as backgrounds, separated by ';' (e.g. cstr3;xt_my;yt_mx).",
    )
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--fps", type=float, default=None, help="Override output video fps.")
    parser.add_argument("--draw-ground-truth", action="store_true", default=False)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/object_detection_sequence_videos"))
    parser.add_argument("--keep-frames", action="store_true", default=True)
    parser.add_argument("--no-keep-frames", action="store_false", dest="keep_frames")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    folder = args.folder.strip("/\\")
    reps = _parse_rep_list(args.reps)
    if not reps:
        raise ValueError("At least one representation must be requested.")

    images_dir = Path(data_cfg["images_root"]) / folder
    dataset_folder_dir = Path(data_cfg["labels_root"]) / folder
    labels_dir = dataset_folder_dir / data_cfg.get("labels_subdir", "Event_YOLO")
    if not images_dir.exists():
        raise FileNotFoundError(f"Missing rendered images directory: {images_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"Missing labels directory: {labels_dir}")

    required_reps = list(data_cfg["representations"])
    stems = _find_frame_stems(images_dir=images_dir, labels_dir=labels_dir, required_reps=required_reps)
    if args.max_frames is not None:
        stems = stems[: max(0, int(args.max_frames))]
    if not stems:
        raise RuntimeError(f"No frames found for folder {folder} with model reps {required_reps}.")

    device = torch.device(cfg["train"].get("device", "cpu") if torch.cuda.is_available() else "cpu")
    model = build_model(cfg, device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()

    image_size = tuple(data_cfg["image_size"])
    output_dir = args.output_dir / folder
    output_dir.mkdir(parents=True, exist_ok=True)

    fps = float(args.fps) if args.fps is not None else _infer_fps(stems)
    print(f"Rendering folder {folder} with {len(stems)} frames at {fps:.3f} fps")

    rgb_index: Optional[List[Tuple[float, Path]]] = None
    if any(rep.lower() == "rgb" for rep in reps):
        for rgb_dir_name in ("RGB", "PADDED_RGB"):
            candidate = dataset_folder_dir / rgb_dir_name
            rgb_index = _build_rgb_index(candidate, label_time_unit=float(data_cfg.get("label_time_unit", 1e-6)))
            if rgb_index:
                break

    rep_frame_dirs = {}
    for rep in reps:
        rep_frames_dir = output_dir / f"frames_{rep}"
        rep_frames_dir.mkdir(parents=True, exist_ok=True)
        rep_frame_dirs[rep] = rep_frames_dir

    for idx, stem in enumerate(stems):
        model_inputs: Dict[str, torch.Tensor] = {}
        for model_rep in data_cfg["representations"]:
            model_inputs[model_rep] = _load_input_tensor(images_dir / f"{stem}_{model_rep}.png", image_size).unsqueeze(0).to(device)

        with torch.no_grad():
            preds = model(model_inputs)
            pred_box = preds["boxes"][0].detach().cpu()
            pred_score = float(preds["objectness_logits"][0].sigmoid().cpu().item())

        label_path = labels_dir / f"{stem}.txt"
        gt_boxes = _read_yolo_boxes(label_path) if label_path.exists() else []
        label_time_raw = _parse_frame_time(stem)
        label_time_s = float(label_time_raw) * float(data_cfg.get("label_time_unit", 1e-6)) if label_time_raw is not None else 0.0

        for rep in reps:
            bg_img = _load_background_image(
                rep=rep,
                stem=stem,
                label_time_s=label_time_s,
                images_dir=images_dir,
                dataset_folder_dir=dataset_folder_dir,
                label_time_unit=float(data_cfg.get("label_time_unit", 1e-6)),
                rgb_index=rgb_index,
            )
            vis = _draw_pred_overlay(
                bg_img,
                rep=rep,
                pred_box=pred_box,
                pred_score=pred_score,
                score_threshold=float(args.score_threshold),
                gt_boxes=gt_boxes,
                draw_gt=bool(args.draw_ground_truth),
            )
            out_frame = rep_frame_dirs[rep] / f"{idx:06d}.png"
            vis.save(out_frame)

        if (idx + 1) % 100 == 0:
            print(f"rendered {idx + 1}/{len(stems)} frames")

    for rep in reps:
        rep_frames_dir = rep_frame_dirs[rep]
        video_path = output_dir / f"{folder}_{rep}.mp4"
        ok = _write_video_ffmpeg(rep_frames_dir / "%06d.png", video_path, fps)
        if ok:
            print(f"Wrote {video_path}")
        else:
            print(f"ffmpeg unavailable or failed; kept frames in {rep_frames_dir}")

        if not args.keep_frames and ok:
            for frame_path in rep_frames_dir.glob("*.png"):
                frame_path.unlink()
            rep_frames_dir.rmdir()


if __name__ == "__main__":
    main()
