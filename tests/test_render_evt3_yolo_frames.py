from pathlib import Path
import sys

import pytest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scripts.render_evt3_yolo_frames import _parse_image_sizes, _render_histogram_grid


def test_parse_image_sizes() -> None:
    assert _parse_image_sizes("xt_my=398x224;yt_mx=224x224;cstr3=398x224") == {
        "xt_my": (398, 224),
        "yt_mx": (224, 224),
        "cstr3": (398, 224),
    }


def test_parse_image_sizes_rejects_invalid_entry() -> None:
    with pytest.raises(ValueError, match="Expected rep=WIDTHxHEIGHT"):
        _parse_image_sizes("xt_my:398x224")


def test_histogram_grid_uses_explicit_output_size_as_native_size() -> None:
    events = np.array(
        [
            [10.0, 20.0, 0.1, 1.0],
            [100.0, 50.0, 0.2, 0.0],
        ],
        dtype=np.float32,
    )

    img = _render_histogram_grid(
        events,
        width=1280,
        height=720,
        t0=0.0,
        dt=1.0,
        plane="xt_my",
        time_bins=224,
        patch_size=224,
        grid_x=1,
        grid_y=1,
        output_size=(398, 224),
    )

    assert img.shape == (224, 398, 3)
