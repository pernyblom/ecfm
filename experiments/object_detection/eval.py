from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.object_detection.losses import compute_losses
from experiments.object_detection.metrics import (
    build_detections,
    detection_scores,
    map_metrics,
    summarize_metrics,
)
from experiments.object_detection.models.factory import build_model
from experiments.object_detection.train import _build_dataset, _make_loader
from experiments.object_detection.utils.config import load_config
from experiments.object_detection.visualization import save_sample_visualization


def _weighted_mean_dict(rows: List[Dict[str, float]], weights: List[int]) -> Dict[str, float]:
    if not rows:
        return {}
    keys = set()
    for row in rows:
        keys.update(row.keys())
    out: Dict[str, float] = {}
    for key in sorted(keys):
        weighted_sum = 0.0
        key_weight = 0.0
        for row, weight in zip(rows, weights):
            if key not in row:
                continue
            weighted_sum += row[key] * weight
            key_weight += weight
        if key_weight > 0:
            out[key] = weighted_sum / key_weight
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", type=str, default="val")
    args = parser.parse_args()

    cfg = load_config(args.config)
    dataset = _build_dataset(cfg, args.split)
    loader = _make_loader(
        dataset,
        batch_size=int(cfg["train"].get("batch_size", 16)),
        shuffle=False,
        train_cfg=cfg["train"],
    )
    device = torch.device(cfg["train"].get("device", "cpu") if torch.cuda.is_available() else "cpu")
    model = build_model(cfg, device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()

    rows: List[Dict[str, float]] = []
    row_weights: List[int] = []
    detections_all: List[dict] = []
    gt_by_frame_all: Dict[str, torch.Tensor] = {}
    eval_cfg = cfg.get("eval", {})
    vis_dir = Path(eval_cfg.get("vis_output_dir", "outputs/object_detection_eval_vis"))
    max_vis = int(eval_cfg.get("max_visualizations", 0))
    backdrop_rep = str(eval_cfg.get("vis_backdrop_rep", "cstr3"))
    metrics_output = eval_cfg.get("metrics_output")

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            inputs = {k: v.to(device) for k, v in batch.inputs.items()}
            target_heatmaps = {k: v.to(device) for k, v in batch.heatmaps.items()}
            preds = model(inputs)
            _, _, aux = compute_losses(
                preds,
                batch.gt_boxes_xywh,
                target_heatmaps,
                heatmap_weight=float(cfg["train"].get("heatmap_weight", 1.0)),
                box_weight=float(cfg["train"].get("box_weight", 1.0)),
                objectness_weight=float(cfg["train"].get("objectness_weight", 1.0)),
                box_l1_weight=float(cfg["train"].get("box_l1_weight", 1.0)),
                box_ciou_weight=float(cfg["train"].get("box_ciou_weight", 1.0)),
                match_score_weight=float(cfg["train"].get("match_score_weight", 1.0)),
                match_l1_weight=float(cfg["train"].get("match_l1_weight", 1.0)),
                match_ciou_weight=float(cfg["train"].get("match_ciou_weight", 1.0)),
            )
            pred_scores = detection_scores(preds)
            rows.append(
                summarize_metrics(
                    preds,
                    batch.gt_boxes_xywh,
                    target_heatmaps,
                    aux["target_objectness"],
                    aux["frame_matches"],
                    batch.frame_keys,
                    tuple(cfg["data"]["frame_size"]),
                )
            )
            row_weights.append(len(batch.frame_keys))
            dets, gt_map = build_detections(
                preds["boxes"].detach().cpu(),
                pred_scores.detach().cpu(),
                batch.gt_boxes_xywh,
                batch.frame_keys,
            )
            detections_all.extend(dets)
            gt_by_frame_all.update({k: v.detach().cpu() for k, v in gt_map.items()})

            for i in range(len(batch.frame_keys)):
                vis_idx = batch_idx * loader.batch_size + i
                if vis_idx >= max_vis:
                    break
                stem = batch.frame_keys[i].replace("/", "_")
                save_sample_visualization(
                    output_dir=vis_dir,
                    stem=stem,
                    inputs={rep: batch.inputs[rep][i] for rep in batch.inputs},
                    pred_boxes=preds["boxes"][i].cpu(),
                    pred_scores=preds["objectness_logits"][i].sigmoid().cpu(),
                    target_boxes=batch.gt_boxes_xywh[i].cpu(),
                    pred_heatmaps={rep: preds["heatmaps"][rep][i].cpu() for rep in preds.get("heatmaps", {})},
                    target_heatmaps={rep: batch.heatmaps[rep][i].cpu() for rep in batch.heatmaps if rep in preds.get("heatmaps", {})},
                    xy_backdrop_rep=backdrop_rep,
                    score_threshold=float(eval_cfg.get("vis_score_threshold", 0.5)),
                )

    summary = _weighted_mean_dict(rows, row_weights)
    if detections_all:
        summary.update(
            map_metrics(
                detections=detections_all,
                gt_by_frame=gt_by_frame_all,
                frame_size=tuple(cfg["data"]["frame_size"]),
            )
        )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if metrics_output:
        out_path = Path(metrics_output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
