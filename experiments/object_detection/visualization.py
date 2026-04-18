from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import torch
from PIL import Image, ImageDraw


def _tensor_to_pil(img: torch.Tensor) -> Image.Image:
    arr = img.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    return Image.fromarray((arr * 255.0).astype(np.uint8))


def _overlay_heatmap(background: Image.Image, heatmap: torch.Tensor, color: tuple[int, int, int]) -> Image.Image:
    base = background.convert("RGB")
    hm = heatmap.detach().cpu().squeeze().float().clamp(0, 1).numpy()[:, :, None]
    overlay = np.asarray(base, dtype=np.float32).copy()
    tint = np.zeros_like(overlay)
    tint[:, :, 0] = color[0]
    tint[:, :, 1] = color[1]
    tint[:, :, 2] = color[2]
    overlay = overlay * (1.0 - 0.45 * hm) + tint * (0.45 * hm)
    return Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8))


def _draw_xy_box(draw: ImageDraw.ImageDraw, box: torch.Tensor, color: tuple[int, int, int], size: tuple[int, int], width: int = 2) -> None:
    w, h = size
    cx, cy, bw, bh = [float(v) for v in box]
    x0 = (cx - bw / 2.0) * w
    y0 = (cy - bh / 2.0) * h
    x1 = (cx + bw / 2.0) * w
    y1 = (cy + bh / 2.0) * h
    draw.rectangle([x0, y0, x1, y1], outline=color, width=width)
    return None


def save_sample_visualization(
    *,
    output_dir: Path,
    stem: str,
    inputs: Dict[str, torch.Tensor],
    pred_boxes: torch.Tensor,
    pred_scores: torch.Tensor,
    target_boxes: torch.Tensor,
    pred_heatmaps: Dict[str, torch.Tensor],
    target_heatmaps: Dict[str, torch.Tensor],
    xy_backdrop_rep: str,
    score_threshold: float = 0.5,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for rep, img_t in inputs.items():
        _tensor_to_pil(img_t).save(output_dir / f"{stem}_{rep}.png")

    for rep, target in target_heatmaps.items():
        base = _tensor_to_pil(inputs[rep])
        _overlay_heatmap(base, target, (0, 255, 0)).save(output_dir / f"{stem}_{rep}_gt_heatmap.png")
        _overlay_heatmap(base, pred_heatmaps[rep].sigmoid(), (255, 196, 0)).save(
            output_dir / f"{stem}_{rep}_pred_heatmap.png"
        )

    box_rep = xy_backdrop_rep if xy_backdrop_rep in inputs else next(iter(inputs.keys()))
    base = _tensor_to_pil(inputs[box_rep])
    gt_img = base.copy().convert("RGB")
    gt_draw = ImageDraw.Draw(gt_img)
    for box in target_boxes:
        _draw_xy_box(gt_draw, box, (0, 255, 0), gt_img.size)
    gt_img.save(output_dir / f"{stem}_{box_rep}_gt_box.png")

    pred_img = base.copy().convert("RGB")
    pred_draw = ImageDraw.Draw(pred_img)
    for box, score in zip(pred_boxes, pred_scores):
        score_value = float(score)
        if score_value < score_threshold:
            continue
        _draw_xy_box(pred_draw, box, (255, 196, 0), pred_img.size)
        w, h = pred_img.size
        cx, cy, bw, bh = [float(v) for v in box]
        x0 = (cx - bw / 2.0) * w
        y0 = (cy - bh / 2.0) * h
        label = f"score {score_value:.3f}"
        text_x = max(2.0, x0 + 2.0)
        text_y = max(2.0, y0 - 14.0)
        bbox = pred_draw.textbbox((text_x, text_y), label)
        pred_draw.rectangle(bbox, fill=(0, 0, 0))
        pred_draw.text((text_x, text_y), label, fill=(255, 196, 0))
    pred_img.save(output_dir / f"{stem}_{box_rep}_pred_box.png")
