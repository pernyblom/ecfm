from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.object_detection.models.centernet import CenterNetDetector


def test_centernet_output_grid_uses_configured_image_size_not_frame_size() -> None:
    model = CenterNetDetector(
        representations=["cstr3"],
        image_sizes={"cstr3": (224, 224)},
        frame_size=(1280, 720),
        backbone_cfg={"type": "small_cnn", "in_channels": 3, "channels": [8, 16], "out_dim": 16},
        hidden_dim=16,
        output_stride=4,
        topk=10,
        predict_velocity=False,
    )

    with torch.enable_grad():
        out = model({"cstr3": torch.rand(2, 3, 224, 224)})

    assert out["centernet_heatmap_logits"].shape == (2, 1, 56, 56)
    assert out["centernet_size_raw"].shape == (2, 2, 56, 56)
    assert out["boxes"].shape == (2, 10, 4)
    assert not out["boxes"].requires_grad
    assert not out["scores"].requires_grad
    assert not out["fused_features"].requires_grad


def test_centernet_single_representation_disables_fusion_block() -> None:
    model = CenterNetDetector(
        representations=["cstr3"],
        image_sizes={"cstr3": (32, 32)},
        frame_size=(1280, 720),
        backbone_cfg={"type": "small_cnn", "in_channels": 3, "channels": [8, 16], "out_dim": 16},
        hidden_dim=32,
        output_stride=4,
        topk=10,
        predict_velocity=False,
    )

    assert isinstance(model.fusion, torch.nn.Identity)


def test_centernet_multiple_representations_keeps_fusion_block() -> None:
    model = CenterNetDetector(
        representations=["cstr3", "xt_my"],
        image_sizes={"cstr3": (32, 32), "xt_my": (32, 16)},
        frame_size=(1280, 720),
        backbone_cfg={"type": "small_cnn", "in_channels": 3, "channels": [8, 16], "out_dim": 16},
        hidden_dim=32,
        output_stride=4,
        topk=10,
        predict_velocity=False,
    )

    assert not isinstance(model.fusion, torch.nn.Identity)
