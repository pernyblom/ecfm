from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scripts.render_evt3_yolo_frames import _parse_image_sizes


def test_parse_image_sizes() -> None:
    assert _parse_image_sizes("xt_my=398x224;yt_mx=224x224;cstr3=398x224") == {
        "xt_my": (398, 224),
        "yt_mx": (224, 224),
        "cstr3": (398, 224),
    }


def test_parse_image_sizes_rejects_invalid_entry() -> None:
    with pytest.raises(ValueError, match="Expected rep=WIDTHxHEIGHT"):
        _parse_image_sizes("xt_my:398x224")
