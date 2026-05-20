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

from scripts.render_evt3_yolo_frames import (
    _parse_image_sizes,
    _render_histogram_grid,
    _resolve_representation_alias,
)
from scripts.render_evt3_representations import _frame_slices, _resolve_frame_windows, _time_frame_slices


def test_parse_image_sizes() -> None:
    assert _parse_image_sizes("xt_my=398x224;yt_mx=224x224;cstr3=398x224") == {
        "xt_my": (398, 224),
        "yt_mx": (224, 224),
        "cstr3": (398, 224),
    }


def test_parse_image_sizes_rejects_invalid_entry() -> None:
    with pytest.raises(ValueError, match="Expected rep=WIDTHxHEIGHT"):
        _parse_image_sizes("xt_my:398x224")


def test_grid_split_representation_alias_parses_base_and_grid() -> None:
    assert _resolve_representation_alias("xt_my_10x10") == ("xt_my", 10, 10)
    assert _resolve_representation_alias("yt_mx_4x2") == ("yt_mx", 4, 2)
    assert _resolve_representation_alias("xt_my") == ("xt_my", None, None)


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


def test_frame_slices_use_event_count_windows() -> None:
    assert _frame_slices(
        10,
        events_per_frame=4,
        stride_events=4,
        max_frames=None,
    ) == [(0, 4), (4, 8), (8, 10)]


def test_frame_slices_support_overlapping_stride() -> None:
    assert _frame_slices(
        8,
        events_per_frame=4,
        stride_events=2,
        max_frames=3,
    ) == [(0, 4), (2, 6), (4, 8)]


def test_time_frame_slices_use_fixed_duration_windows() -> None:
    timestamps = np.array([0.0, 10.0, 20.0, 35.0, 42.0], dtype=np.float32)

    assert _time_frame_slices(
        timestamps,
        window=20.0,
        stride_window=20.0,
        max_frames=None,
    ) == [(0, 2, 0.0, 20.0), (2, 4, 20.0, 40.0), (4, 5, 40.0, 60.0)]


def test_resolve_frame_windows_requires_one_mode() -> None:
    args = type(
        "Args",
        (),
        {
            "events_per_frame": 4,
            "stride_events": None,
            "window": 20.0,
            "stride_window": None,
            "max_frames": None,
        },
    )()

    with pytest.raises(ValueError, match="exactly one"):
        _resolve_frame_windows(np.array([0.0, 1.0], dtype=np.float32), args)
