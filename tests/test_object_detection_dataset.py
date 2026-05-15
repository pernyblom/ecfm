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


def test_rgb_only_auto_labels_use_rgb_yolo_and_dataset_rgb_frames(tmp_path: Path) -> None:
    _write_label(tmp_path / "labels" / "RGB_YOLO" / "Video_0_16_03_03.363444.txt")
    _write_image(tmp_path / "labels" / "RGB" / "Video_0_16_03_03.363444.jpg")

    dataset = FredDetectionDataset(
        images_root=tmp_path / "images",
        labels_root=tmp_path / "labels",
        representations=["rgb"],
        heatmap_representations=[],
        image_sizes={"rgb": (8, 8)},
        frame_size=(8, 8),
        folders=[""],
        labels_subdir="auto",
        verify_render_manifest=False,
        require_boxes=False,
    )

    assert dataset.labels_subdir == "RGB_YOLO"
    assert len(dataset) == 1
    assert dataset.samples[0]["input_paths"]["rgb"].endswith("Video_0_16_03_03.363444.jpg")
    assert dataset[0].inputs["rgb"].shape == (3, 8, 8)


def test_auto_labels_use_event_yolo_when_any_event_rep_is_present(tmp_path: Path) -> None:
    _write_label(tmp_path / "labels" / "Event_YOLO" / "frame_000001.txt")
    _write_image(tmp_path / "images" / "frame_000001_cstr3.png")
    _write_image(tmp_path / "images" / "frame_000001_rgb.png")

    dataset = FredDetectionDataset(
        images_root=tmp_path / "images",
        labels_root=tmp_path / "labels",
        representations=["cstr3", "rgb"],
        heatmap_representations=[],
        image_sizes={"cstr3": (8, 8), "rgb": (8, 8)},
        frame_size=(8, 8),
        folders=[""],
        labels_subdir="auto",
        verify_render_manifest=False,
        require_boxes=False,
    )

    assert dataset.labels_subdir == "Event_YOLO"
    assert len(dataset) == 1


def test_mixed_event_rgb_uses_nearest_dataset_rgb_frame(tmp_path: Path) -> None:
    _write_label(tmp_path / "labels" / "Event_YOLO" / "Video_0_frame_100000000.txt")
    _write_image(tmp_path / "images" / "Video_0_frame_100000000_cstr3.png")
    _write_image(tmp_path / "labels" / "RGB" / "Video_0_16_03_03.000000.jpg")
    _write_image(tmp_path / "labels" / "RGB" / "Video_0_16_04_43.000000.jpg")

    dataset = FredDetectionDataset(
        images_root=tmp_path / "images",
        labels_root=tmp_path / "labels",
        representations=["cstr3", "rgb"],
        heatmap_representations=[],
        image_sizes={"cstr3": (8, 8), "rgb": (8, 8)},
        frame_size=(8, 8),
        folders=[""],
        labels_subdir="auto",
        label_time_unit=1.0e-6,
        verify_render_manifest=False,
        require_boxes=False,
    )

    assert dataset.labels_subdir == "Event_YOLO"
    assert len(dataset) == 1
    assert dataset.samples[0]["input_paths"]["rgb"].endswith("Video_0_16_04_43.000000.jpg")


def test_mixed_event_rgb_manifest_validation_allows_dataset_rgb(tmp_path: Path) -> None:
    _write_label(tmp_path / "labels" / "Event_YOLO" / "Video_0_frame_100000000.txt")
    _write_image(tmp_path / "images" / "Video_0_frame_100000000_cstr3.png")
    _write_image(tmp_path / "labels" / "RGB" / "Video_0_16_04_43.000000.jpg")
    (tmp_path / "images" / "render_manifest.json").write_text(
        """
{
  "render_params": {"window_mode": "trailing"},
  "files": [
    {
      "label_stem": "Video_0_frame_100000000",
      "window_duration_render_units": 33333,
      "representations": {
        "cstr3": {"image_size": [8, 8]}
      }
    }
  ]
}
""".strip(),
        encoding="utf-8",
    )

    dataset = FredDetectionDataset(
        images_root=tmp_path / "images",
        labels_root=tmp_path / "labels",
        representations=["cstr3", "rgb"],
        heatmap_representations=[],
        image_sizes={"cstr3": (8, 8), "rgb": (8, 8)},
        frame_size=(8, 8),
        folders=[""],
        labels_subdir="auto",
        label_time_unit=1.0e-6,
        image_window_ms=33.333,
        image_window_mode="trailing",
        verify_render_manifest=True,
        require_boxes=False,
    )

    assert len(dataset) == 1
