from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np


def _parse_header_lines(header_lines: List[str]) -> Dict[str, str]:
    meta: Dict[str, str] = {}
    for line in header_lines:
        line = line.strip()
        if not line.startswith("%"):
            continue
        content = line[1:].strip()
        if not content:
            continue
        if content.startswith("format "):
            # Example: "format EVT3;height=720;width=1280"
            parts = content.split(" ", 1)
            if len(parts) == 2:
                meta["format"] = parts[1]
                fmt_parts = parts[1].split(";")
                meta["format_name"] = fmt_parts[0]
                for kv in fmt_parts[1:]:
                    if "=" in kv:
                        k, v = kv.split("=", 1)
                        meta[k.strip()] = v.strip()
        else:
            # Example: "geometry 1280x720"
            if " " in content:
                k, v = content.split(" ", 1)
                meta[k.strip()] = v.strip()
    return meta


def read_raw_header(path: Path) -> Tuple[List[str], int, Dict[str, str]]:
    header_lines: List[str] = []
    with path.open("rb") as f:
        while True:
            pos = f.tell()
            line = f.readline()
            if not line:
                break
            if not line.startswith(b"%"):
                f.seek(pos)
                break
            try:
                decoded = line.decode("ascii", errors="ignore").rstrip("\r\n")
            except Exception:
                decoded = line.decode("latin-1", errors="ignore").rstrip("\r\n")
            header_lines.append(decoded)
            if decoded.startswith("% end"):
                break
        data_offset = f.tell()
    meta = _parse_header_lines(header_lines)
    return header_lines, data_offset, meta


@dataclass
class _Evt3DecoderState:
    y: int = 0
    vbase_x: int = 0
    vpol: int = 0
    time_low: int = 0
    time_high: int = 0
    time_high_raw: int = 0
    time_high_base: int = 0
    last_time_high_raw: int | None = None
    last_time_low: int | None = None
    time_high_updated_since_low: bool = False
    have_y: bool = False
    have_vbase: bool = False
    pending_other_subtype: int | None = None
    pending_other_timestamp: int = 0
    pending_continued_12: Tuple[int, ...] = ()


@dataclass(frozen=True)
class Evt3ExtTrigger:
    """Decoded EVT3 external-trigger event."""

    timestamp: int
    id: int
    value: int


@dataclass(frozen=True)
class Evt3MonitoringEvent:
    """Decoded EVT3 OTHERS event and its optional continuation payload."""

    timestamp: int
    subtype: int
    payload: int | None = None


@dataclass
class Evt3AuxiliaryEvents:
    """Optional sink for non-CD events encountered while decoding."""

    ext_triggers: List[Evt3ExtTrigger]
    monitoring: List[Evt3MonitoringEvent]

    def __init__(self) -> None:
        self.ext_triggers = []
        self.monitoring = []


def _empty_counters() -> Dict[str, int]:
    return {
        "continued_4": 0,
        "continued_12": 0,
        "ext_trigger": 0,
        "others": 0,
        "unknown": 0,
        "time_high_wrap": 0,
        "time_low_wrap_fix": 0,
        "missing_y": 0,
        "missing_vbase": 0,
        "monitoring_events": 0,
        "monitoring_master_in_cd_count": 0,
        "monitoring_master_rate_control_cd_count": 0,
        "monitoring_markers": 0,
        "malformed_continued": 0,
    }


def _timestamp(state: _Evt3DecoderState) -> int:
    return (state.time_high << 12) | state.time_low


def _emit_monitoring_event(
    state: _Evt3DecoderState,
    counters: Dict[str, int],
    auxiliary: Evt3AuxiliaryEvents | None,
    payload: int | None,
) -> None:
    subtype = state.pending_other_subtype
    if subtype is None:
        return
    counters["monitoring_events"] += 1
    if subtype == 0x014:
        counters["monitoring_master_in_cd_count"] += 1
    elif subtype == 0x016:
        counters["monitoring_master_rate_control_cd_count"] += 1
    if payload is None:
        counters["monitoring_markers"] += 1
    if auxiliary is not None:
        auxiliary.monitoring.append(
            Evt3MonitoringEvent(state.pending_other_timestamp, subtype, payload)
        )
    state.pending_other_subtype = None
    state.pending_continued_12 = ()


def _finish_evt3_stream(
    state: _Evt3DecoderState,
    counters: Dict[str, int],
    auxiliary: Evt3AuxiliaryEvents | None,
) -> None:
    """Flush an OTHERS event held for continuation look-ahead at EOF."""
    if state.pending_other_subtype is None:
        return
    if state.pending_continued_12:
        counters["malformed_continued"] += 1
        state.pending_other_subtype = None
        state.pending_continued_12 = ()
    else:
        _emit_monitoring_event(state, counters, auxiliary, None)


def _decode_evt3_words(
    words: np.ndarray,
    *,
    state: _Evt3DecoderState | None = None,
    counters: Dict[str, int] | None = None,
    auxiliary: Evt3AuxiliaryEvents | None = None,
) -> Tuple[np.ndarray, Dict[str, int], _Evt3DecoderState]:
    state = state or _Evt3DecoderState()
    counters = counters or _empty_counters()

    events: List[Tuple[int, int, int, int]] = []

    for w in words:
        typ = (w >> 12) & 0xF
        payload = w & 0x0FFF

        # OTHERS may be followed by a 12+12+4-bit continuation payload. Keep
        # this state across chunks so streamed and in-memory decoding agree.
        if state.pending_other_subtype is not None:
            ncontinued = len(state.pending_continued_12)
            if ncontinued < 2 and typ == 0xF:
                counters["continued_12"] += 1
                state.pending_continued_12 += (int(payload),)
                continue
            if ncontinued == 2 and typ == 0x7:
                counters["continued_4"] += 1
                value = (
                    state.pending_continued_12[0]
                    | (state.pending_continued_12[1] << 12)
                    | (int(payload & 0xF) << 24)
                )
                _emit_monitoring_event(state, counters, auxiliary, value)
                continue

            if ncontinued:
                counters["malformed_continued"] += 1
                state.pending_other_subtype = None
                state.pending_continued_12 = ()
            else:
                _emit_monitoring_event(state, counters, auxiliary, None)

        if typ == 0x0:  # EVT_ADDR_Y
            state.y = int(payload & 0x7FF)
            state.have_y = True
        elif typ == 0x2:  # EVT_ADDR_X
            if not state.have_y:
                counters["missing_y"] += 1
                continue
            pol = int((w >> 11) & 0x1)
            x = int(payload & 0x7FF)
            ts = _timestamp(state)
            events.append((x, state.y, ts, pol))
        elif typ == 0x3:  # VECT_BASE_X
            state.vpol = int((w >> 11) & 0x1)
            state.vbase_x = int(payload & 0x7FF)
            state.have_vbase = True
        elif typ == 0x4:  # VECT_12
            if not state.have_y or not state.have_vbase:
                counters["missing_y"] += int(not state.have_y)
                counters["missing_vbase"] += int(not state.have_vbase)
                continue
            valid = int(payload & 0x0FFF)
            ts = _timestamp(state)
            for i in range(12):
                if (valid >> i) & 0x1:
                    events.append((state.vbase_x + i, state.y, ts, state.vpol))
            state.vbase_x += 12
        elif typ == 0x5:  # VECT_8
            if not state.have_y or not state.have_vbase:
                counters["missing_y"] += int(not state.have_y)
                counters["missing_vbase"] += int(not state.have_vbase)
                continue
            valid = int(payload & 0x00FF)
            ts = _timestamp(state)
            for i in range(8):
                if (valid >> i) & 0x1:
                    events.append((state.vbase_x + i, state.y, ts, state.vpol))
            state.vbase_x += 8
        elif typ == 0x6:  # EVT_TIME_LOW
            payload_int = int(payload)
            if state.last_time_low is not None and payload_int < state.last_time_low:
                if not state.time_high_updated_since_low:
                    state.time_high += 1
                    state.time_high_raw = (state.time_high_raw + 1) & 0xFFF
                    counters["time_low_wrap_fix"] += 1
            state.time_low = payload_int
            state.last_time_low = payload_int
            state.time_high_updated_since_low = False
        elif typ == 0x8:  # EVT_TIME_HIGH
            payload_int = int(payload)
            if state.last_time_high_raw is not None and payload_int < state.last_time_high_raw:
                state.time_high_base += 0x1000
                counters["time_high_wrap"] += 1
            state.time_high_raw = payload_int
            state.time_high = state.time_high_base + payload_int
            state.last_time_high_raw = payload_int
            state.time_high_updated_since_low = True
        elif typ == 0x7:  # CONTINUED_4
            counters["continued_4"] += 1
            counters["malformed_continued"] += 1
        elif typ == 0xA:  # EXT_TRIGGER
            counters["ext_trigger"] += 1
            if auxiliary is not None:
                auxiliary.ext_triggers.append(
                    Evt3ExtTrigger(
                        timestamp=_timestamp(state),
                        id=int((payload >> 8) & 0xF),
                        value=int(payload & 0x1),
                    )
                )
        elif typ == 0xE:  # OTHERS
            counters["others"] += 1
            state.pending_other_subtype = int(payload)
            state.pending_other_timestamp = _timestamp(state)
            state.pending_continued_12 = ()
        elif typ == 0xF:  # CONTINUED_12
            counters["continued_12"] += 1
            counters["malformed_continued"] += 1
        else:
            counters["unknown"] += 1

    if not events:
        return np.empty((0, 4), dtype=np.float32), counters, state
    return np.asarray(events, dtype=np.float32), counters, state


def decode_evt3_bytes(
    data: bytes,
    endian: str = "little",
    *,
    auxiliary: Evt3AuxiliaryEvents | None = None,
) -> Tuple[np.ndarray, Dict[str, int]]:
    if endian not in {"little", "big"}:
        raise ValueError("endian must be 'little' or 'big'")
    dtype = "<u2" if endian == "little" else ">u2"
    words = np.frombuffer(data, dtype=dtype)
    events, counters, state = _decode_evt3_words(words, auxiliary=auxiliary)
    _finish_evt3_stream(state, counters, auxiliary)
    return events, counters


def decode_evt3_raw(
    path: Path,
    endian: str = "little",
    require_evt3: bool = True,
    *,
    auxiliary: Evt3AuxiliaryEvents | None = None,
) -> Tuple[np.ndarray, Dict[str, int], Dict[str, str], List[str]]:
    header_lines, data_offset, meta = read_raw_header(path)
    fmt = meta.get("format_name") or meta.get("format", "")
    if require_evt3 and fmt and "EVT3" not in fmt:
        raise ValueError(f"Unsupported format in header: {fmt}")
    with path.open("rb") as f:
        f.seek(data_offset)
        data = f.read()
    events, counters = decode_evt3_bytes(data, endian=endian, auxiliary=auxiliary)
    return events, counters, meta, header_lines


def decode_evt3_raw_to_arrayfile(
    path: Path,
    out_path: Path,
    *,
    endian: str = "little",
    require_evt3: bool = True,
    chunk_bytes: int = 64 * 1024 * 1024,
    event_unit: float = 1.0,
    ts_shift_us: float | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
    auxiliary: Evt3AuxiliaryEvents | None = None,
) -> Tuple[int, Dict[str, int], Dict[str, str], List[str]]:
    header_lines, data_offset, meta = read_raw_header(path)
    fmt = meta.get("format_name") or meta.get("format", "")
    if require_evt3 and fmt and "EVT3" not in fmt:
        raise ValueError(f"Unsupported format in header: {fmt}")

    dtype = "<u2" if endian == "little" else ">u2"
    unit = float(event_unit)
    shift = None if ts_shift_us is None else float(ts_shift_us) * unit
    chunk_bytes = max(2, int(chunk_bytes))
    chunk_bytes -= chunk_bytes % 2

    state = _Evt3DecoderState()
    counters = _empty_counters()
    carry = b""
    num_events = 0
    last_time = -np.inf
    total_bytes = max(0, path.stat().st_size - data_offset)
    processed_bytes = 0
    if progress_callback is not None:
        progress_callback(0, total_bytes)

    with path.open("rb") as src, out_path.open("wb") as dst:
        src.seek(data_offset)
        while True:
            chunk = src.read(chunk_bytes)
            if not chunk:
                break
            processed_bytes += len(chunk)
            if progress_callback is not None:
                progress_callback(min(processed_bytes, total_bytes), total_bytes)
            if carry:
                chunk = carry + chunk
                carry = b""
            if len(chunk) % 2:
                carry = chunk[-1:]
                chunk = chunk[:-1]
            if not chunk:
                continue

            words = np.frombuffer(chunk, dtype=dtype)
            events_chunk, counters, state = _decode_evt3_words(
                words, state=state, counters=counters, auxiliary=auxiliary
            )
            if events_chunk.size == 0:
                continue

            t = events_chunk[:, 2].astype(np.float64) * unit
            if shift is not None:
                t = t - shift
                keep = t >= 0.0
                if not np.any(keep):
                    continue
                events_chunk = events_chunk[keep]
                t = t[keep]

            if t.size and t[0] < last_time:
                raise ValueError(
                    "Decoded event timestamps are not monotonic during streamed decode. "
                    "Streaming render cannot continue safely for this file."
                )
            if t.size:
                last_time = float(t[-1])

            out_chunk = events_chunk.astype(np.float32, copy=True)
            out_chunk[:, 2] = t.astype(np.float32)
            out_chunk.tofile(dst)
            num_events += int(out_chunk.shape[0])

    _finish_evt3_stream(state, counters, auxiliary)
    if progress_callback is not None:
        progress_callback(total_bytes, total_bytes)
    return num_events, counters, meta, header_lines
