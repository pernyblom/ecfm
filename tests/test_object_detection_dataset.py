from pathlib import Path
import sys

import numpy as np
import pytest
from PIL import Image

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


def _write_label(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("0 0.5 0.5 0.25 0.25\n", encoding="utf-8")


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(path)


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


def test_missing_representation_files_are_skipped_with_warning(tmp_path: Path, capsys) -> None:
    _write_label(tmp_path / "labels" / "Event_YOLO" / "frame_000001.txt")
    _write_label(tmp_path / "labels" / "Event_YOLO" / "frame_000002.txt")
    _write_image(tmp_path / "images" / "frame_000001_cstr3.png")
    _write_image(tmp_path / "images" / "frame_000001_xt_my.png")
    _write_image(tmp_path / "images" / "frame_000002_cstr3.png")

    dataset = FredDetectionDataset(
        images_root=tmp_path / "images",
        labels_root=tmp_path / "labels",
        representations=["cstr3", "xt_my"],
        heatmap_representations=[],
        image_sizes={"cstr3": (8, 8), "xt_my": (8, 8)},
        frame_size=(8, 8),
        folders=[""],
        labels_subdir="Event_YOLO",
        verify_render_manifest=False,
        require_boxes=False,
        filter_missing_representations=True,
    )

    captured = capsys.readouterr()
    assert len(dataset) == 1
    assert dataset.samples[0]["stem"] == "frame_000001"
    assert "Warning: skipped 1 object detection samples" in captured.out
    assert "xt_my=1" in captured.out


def test_missing_representation_files_can_be_strict(tmp_path: Path) -> None:
    _write_label(tmp_path / "labels" / "Event_YOLO" / "frame_000001.txt")
    _write_image(tmp_path / "images" / "frame_000001_cstr3.png")

    with pytest.raises(FileNotFoundError, match="Missing representation file"):
        FredDetectionDataset(
            images_root=tmp_path / "images",
            labels_root=tmp_path / "labels",
            representations=["cstr3", "xt_my"],
            heatmap_representations=[],
            image_sizes={"cstr3": (8, 8), "xt_my": (8, 8)},
            frame_size=(8, 8),
            folders=[""],
            labels_subdir="Event_YOLO",
            verify_render_manifest=False,
            require_boxes=False,
            filter_missing_representations=False,
        )
