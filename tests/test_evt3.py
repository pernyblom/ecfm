from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ecfm.utils.evt3 import (
    Evt3AuxiliaryEvents,
    _Evt3DecoderState,
    _decode_evt3_words,
    _empty_counters,
    _finish_evt3_stream,
    decode_evt3_bytes,
)


def _words(*words: int) -> bytes:
    return np.asarray(words, dtype="<u2").tobytes()


def test_decodes_monitoring_continuation_payload() -> None:
    auxiliary = Evt3AuxiliaryEvents()
    _, counters = decode_evt3_bytes(
        _words(0x8001, 0x6023, 0xE014, 0xF123, 0xF456, 0x7007),
        auxiliary=auxiliary,
    )

    assert auxiliary.monitoring[0].timestamp == 0x1023
    assert auxiliary.monitoring[0].subtype == 0x014
    assert auxiliary.monitoring[0].payload == 0x7456123
    assert counters["monitoring_master_in_cd_count"] == 1
    assert counters["malformed_continued"] == 0


def test_decodes_marker_and_external_trigger() -> None:
    auxiliary = Evt3AuxiliaryEvents()
    _, counters = decode_evt3_bytes(
        _words(0x8002, 0x6004, 0xE0FF, 0xA301), auxiliary=auxiliary
    )

    assert auxiliary.monitoring[0].subtype == 0x0FF
    assert auxiliary.monitoring[0].payload is None
    assert auxiliary.ext_triggers[0].timestamp == 0x2004
    assert auxiliary.ext_triggers[0].id == 3
    assert auxiliary.ext_triggers[0].value == 1
    assert counters["monitoring_markers"] == 1


def test_monitoring_continuation_survives_chunk_boundaries() -> None:
    state = _Evt3DecoderState()
    counters = _empty_counters()
    auxiliary = Evt3AuxiliaryEvents()

    for chunk in ([0xE016, 0xF001], [0xF002], [0x7003]):
        _, counters, state = _decode_evt3_words(
            np.asarray(chunk, dtype="<u2"),
            state=state,
            counters=counters,
            auxiliary=auxiliary,
        )
    _finish_evt3_stream(state, counters, auxiliary)

    assert auxiliary.monitoring[0].payload == 0x3002001
    assert counters["monitoring_master_rate_control_cd_count"] == 1
    assert counters["malformed_continued"] == 0


def test_orphan_continuation_is_reported() -> None:
    _, counters = decode_evt3_bytes(_words(0xF123, 0x7004))

    assert counters["continued_12"] == 1
    assert counters["continued_4"] == 1
    assert counters["malformed_continued"] == 2
