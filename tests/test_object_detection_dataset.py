from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.object_detection.data.dataset import FredDetectionDataset


def _make_dataset(tmp_path: Path, *, representations, heatmap_representations):
    return FredDetectionDataset(
        images_root=tmp_path / "images",
        labels_root=tmp_path / "labels",
        representations=list(representations),
        heatmap_representations=heatmap_representations,
        image_sizes={rep: (8, 8) for rep in representations},
        frame_size=(8, 8),
        require_boxes=False,
    )


def test_explicit_empty_heatmap_representations_disable_heatmaps(tmp_path: Path) -> None:
    dataset = _make_dataset(
        tmp_path,
        representations=["cstr3"],
        heatmap_representations=[],
    )

    assert dataset.heatmap_representations == []


def test_omitted_heatmap_representations_keep_legacy_default(tmp_path: Path) -> None:
    dataset = _make_dataset(
        tmp_path,
        representations=["xt_my", "yt_mx", "cstr3"],
        heatmap_representations=None,
    )

    assert dataset.heatmap_representations == ["xt_my", "yt_mx"]


def test_enabled_heatmap_representations_must_be_inputs(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Missing heatmap reps"):
        _make_dataset(
            tmp_path,
            representations=["cstr3"],
            heatmap_representations=["xt_my"],
        )
