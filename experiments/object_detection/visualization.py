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


def _draw_xy_box(img: Image.Image, box: torch.Tensor, color: tuple[int, int, int], width: int = 2) -> Image.Image:
    out = img.copy().convert("RGB")
    draw = ImageDraw.Draw(out)
    w, h = out.size
    cx, cy, bw, bh = [float(v) for v in box]
    draw.rectangle(
        [
            (cx - bw / 2.0) * w,
            (cy - bh / 2.0) * h,
            (cx + bw / 2.0) * w,
            (cy + bh / 2.0) * h,
        ],
        outline=color,
        width=width,
    )
    return out


def _draw_pred_box_with_score(
    img: Image.Image,
    box: torch.Tensor,
    score: float,
    color: tuple[int, int, int],
    width: int = 2,
) -> Image.Image:
    out = _draw_xy_box(img, box, color, width=width)
    draw = ImageDraw.Draw(out)
    w, h = out.size
    cx, cy, bw, bh = [float(v) for v in box]
    x0 = (cx - bw / 2.0) * w
    y0 = (cy - bh / 2.0) * h
    label = f"score {score:.3f}"
    text_x = max(2.0, x0 + 2.0)
    text_y = max(2.0, y0 - 14.0)
    bbox = draw.textbbox((text_x, text_y), label)
    draw.rectangle(bbox, fill=(0, 0, 0))
    draw.text((text_x, text_y), label, fill=color)
    return out


def save_sample_visualization(
    *,
    output_dir: Path,
    stem: str,
    inputs: Dict[str, torch.Tensor],
    pred_boxes: torch.Tensor,
    pred_score: float,
    target_box: torch.Tensor,
    pred_heatmaps: Dict[str, torch.Tensor],
    target_heatmaps: Dict[str, torch.Tensor],
    xy_backdrop_rep: str,
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
    _draw_xy_box(base, target_box, (0, 255, 0)).save(output_dir / f"{stem}_{box_rep}_gt_box.png")
    _draw_pred_box_with_score(base, pred_boxes, pred_score, (255, 196, 0)).save(
        output_dir / f"{stem}_{box_rep}_pred_box.png"
    )
