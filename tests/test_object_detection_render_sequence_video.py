from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.object_detection.render_sequence_video import _load_background_image
from experiments.object_detection.render_folder_video import _find_image_stems


def _write_rgb(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.zeros((4, 4, 3), dtype=np.uint8)
    arr[:, :] = color
    Image.fromarray(arr).save(path)


def _pixel(img: Image.Image) -> tuple[int, int, int]:
    return tuple(int(v) for v in img.getpixel((0, 0)))


def test_padded_rgb_rep_uses_padded_rgb_folder(tmp_path: Path) -> None:
    images_dir = tmp_path / "rendered"
    dataset_dir = tmp_path / "dataset"
    _write_rgb(dataset_dir / "RGB" / "frame_000.png", (10, 20, 30))
    _write_rgb(dataset_dir / "PADDED_RGB" / "frame_000.png", (40, 50, 60))

    img = _load_background_image(
        rep="padded_rgb",
        stem="frame_000",
        label_time_s=0.0,
        images_dir=images_dir,
        dataset_folder_dir=dataset_dir,
        label_time_unit=1.0,
        rgb_indices={},
        rgb_source="auto",
    )

    assert _pixel(img) == (40, 50, 60)


def test_rgb_source_padded_rgb_overrides_rendered_rgb(tmp_path: Path) -> None:
    images_dir = tmp_path / "rendered"
    dataset_dir = tmp_path / "dataset"
    _write_rgb(images_dir / "frame_000_rgb.png", (10, 20, 30))
    _write_rgb(dataset_dir / "PADDED_RGB" / "frame_000.png", (40, 50, 60))

    img = _load_background_image(
        rep="rgb",
        stem="frame_000",
        label_time_s=0.0,
        images_dir=images_dir,
        dataset_folder_dir=dataset_dir,
        label_time_unit=1.0,
        rgb_indices={},
        rgb_source="padded_rgb",
    )

    assert _pixel(img) == (40, 50, 60)


def test_find_image_stems_requires_all_rendered_reps(tmp_path: Path) -> None:
    images_dir = tmp_path / "rendered"
    _write_rgb(images_dir / "frame_000000_000000010000_cstr3.png", (10, 20, 30))
    _write_rgb(images_dir / "frame_000000_000000010000_xt_my.png", (10, 20, 30))
    _write_rgb(images_dir / "frame_000001_000000020000_cstr3.png", (10, 20, 30))

    stems = _find_image_stems(images_dir=images_dir, required_reps=["cstr3", "xt_my"])

    assert stems == ["frame_000000_000000010000"]
