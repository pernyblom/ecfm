from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.object_detection import losses


def test_detr_lite_matching_cost_does_not_track_grad(monkeypatch) -> None:
    seen_requires_grad = []
    original_match_queries = losses._match_queries

    def wrapped_match_queries(cost: torch.Tensor):
        seen_requires_grad.append(cost.requires_grad)
        return original_match_queries(cost)

    monkeypatch.setattr(losses, "_match_queries", wrapped_match_queries)
    pred_boxes = torch.tensor([[[0.5, 0.5, 0.2, 0.2], [0.2, 0.2, 0.1, 0.1]]], requires_grad=True)
    pred_logits = torch.zeros((1, 2), requires_grad=True)
    target_boxes = [torch.tensor([[0.5, 0.5, 0.2, 0.2]])]

    loss, _, _ = losses.compute_detr_lite_losses(
        {"boxes": pred_boxes, "objectness_logits": pred_logits, "heatmaps": {}},
        target_boxes,
        {},
    )
    loss.backward()

    assert seen_requires_grad == [False]
    assert pred_boxes.grad is not None
    assert pred_logits.grad is not None
