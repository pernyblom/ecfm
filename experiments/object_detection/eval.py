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

from experiments.object_detection.metrics import summarize_metrics
from experiments.object_detection.models.factory import build_model
from experiments.object_detection.train import _build_dataset, _collate
from experiments.object_detection.utils.config import load_config
from experiments.object_detection.visualization import save_sample_visualization


def _mean_dict(rows: List[Dict[str, float]]) -> Dict[str, float]:
    if not rows:
        return {}
    keys = rows[0].keys()
    return {key: sum(row[key] for row in rows) / len(rows) for key in keys}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", type=str, default="val")
    args = parser.parse_args()

    cfg = load_config(args.config)
    dataset = _build_dataset(cfg, args.split)
    loader = DataLoader(
        dataset,
        batch_size=int(cfg["train"].get("batch_size", 16)),
        shuffle=False,
        num_workers=int(cfg["train"].get("num_workers", 0)),
        collate_fn=_collate,
    )
    device = torch.device(cfg["train"].get("device", "cpu") if torch.cuda.is_available() else "cpu")
    model = build_model(cfg, device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()

    rows: List[Dict[str, float]] = []
    eval_cfg = cfg.get("eval", {})
    vis_dir = Path(eval_cfg.get("vis_output_dir", "outputs/object_detection_eval_vis"))
    max_vis = int(eval_cfg.get("max_visualizations", 0))
    backdrop_rep = str(eval_cfg.get("vis_backdrop_rep", "cstr3"))
    metrics_output = eval_cfg.get("metrics_output")

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            inputs = {k: v.to(device) for k, v in batch.inputs.items()}
            target_boxes = batch.box_xywh.to(device)
            target_heatmaps = {k: v.to(device) for k, v in batch.heatmaps.items()}
            preds = model(inputs)
            rows.append(summarize_metrics(preds, target_boxes, target_heatmaps, tuple(cfg["data"]["frame_size"])))

            for i in range(batch.box_xywh.shape[0]):
                vis_idx = batch_idx * loader.batch_size + i
                if vis_idx >= max_vis:
                    break
                stem = batch.frame_keys[i].replace("/", "_")
                save_sample_visualization(
                    output_dir=vis_dir,
                    stem=stem,
                    inputs={rep: batch.inputs[rep][i] for rep in batch.inputs},
                    pred_boxes=preds["boxes"][i].cpu(),
                    target_box=batch.box_xywh[i].cpu(),
                    pred_heatmaps={rep: preds["heatmaps"][rep][i].cpu() for rep in preds["heatmaps"]},
                    target_heatmaps={rep: batch.heatmaps[rep][i].cpu() for rep in batch.heatmaps},
                    xy_backdrop_rep=backdrop_rep,
                )

    summary = _mean_dict(rows)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if metrics_output:
        out_path = Path(metrics_output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
