from collections import defaultdict

import torch
import pytest

from experiments.pose_dynamics.train import (
    _collect_sequence_metrics,
    _finalize_metric_sums,
    _format_sequence_summary,
    _init_metric_sums,
    _summarize_horizons,
    _update_metric_sums,
)


def test_pose_dynamics_metric_aggregation_and_sequence_summary():
    sums = _init_metric_sums(future_steps=3)
    metrics = {
        "loss": 1.0,
        "center_l1": 0.5,
        "pose_reg": 0.1,
        "intr_reg": 0.2,
        "acc_reg": 0.3,
    }
    pred = {
        "pred_centers": torch.tensor(
            [
                [[0.10, 0.20], [1.20, 0.40], [0.30, 0.60]],
                [[0.20, 0.10], [0.30, 0.20], [0.40, 0.30]],
            ],
            dtype=torch.float32,
        )
    }
    target = torch.tensor(
        [
            [[0.00, 0.10], [0.20, 0.30], [0.40, 0.50]],
            [[0.10, 0.10], [0.20, 0.20], [0.30, 0.20]],
        ],
        dtype=torch.float32,
    )

    _update_metric_sums(sums, metrics, pred, target)
    finalized = _finalize_metric_sums(sums, count=1)

    assert finalized["horizon_l1"] == pytest.approx([0.07500000298023224, 0.30000001192092896, 0.10000000894069672])
    assert finalized["oob_step_rate"] == pytest.approx(1.0 / 6.0)
    assert finalized["oob_traj_rate"] == pytest.approx(0.5)
    assert _summarize_horizons(finalized["horizon_l1"]) == "h1 0.07500 h2 0.30000 h3 0.10000"

    seq_sums = defaultdict(lambda: [0.0, 0])
    _collect_sequence_metrics(
        seq_sums,
        ["seq_a/frame_1", "seq_b/frame_2"],
        pred["pred_centers"],
        target,
    )
    summary = _format_sequence_summary(seq_sums, limit=2)
    assert summary.startswith("worst_seq ")
    assert "seq_a:" in summary
    assert "seq_b:" in summary
