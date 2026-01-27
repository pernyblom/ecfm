from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

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


def decode_evt3_bytes(data: bytes, endian: str = "little") -> Tuple[np.ndarray, Dict[str, int]]:
    if endian not in {"little", "big"}:
        raise ValueError("endian must be 'little' or 'big'")
    dtype = "<u2" if endian == "little" else ">u2"
    words = np.frombuffer(data, dtype=dtype)

    events: List[Tuple[int, int, int, int]] = []
    y = 0
    vbase_x = 0
    vpol = 0
    time_low = 0
    time_high = 0
    time_high_raw = 0
    time_high_base = 0
    last_time_high_raw = None
    last_time_low = None
    time_high_updated_since_low = False
    have_y = False
    have_vbase = False
    counters = {
        "continued_4": 0,
        "continued_12": 0,
        "ext_trigger": 0,
        "others": 0,
        "unknown": 0,
        "time_high_wrap": 0,
        "time_low_wrap_fix": 0,
        "missing_y": 0,
        "missing_vbase": 0,
    }

    for w in words:
        typ = (w >> 12) & 0xF
        payload = w & 0x0FFF

        if typ == 0x0:  # EVT_ADDR_Y
            y = int(payload & 0x7FF)
            have_y = True
        elif typ == 0x2:  # EVT_ADDR_X
            if not have_y:
                counters["missing_y"] += 1
                continue
            pol = int((w >> 11) & 0x1)
            x = int(payload & 0x7FF)
            ts = (time_high << 12) | time_low
            events.append((x, y, ts, pol))
        elif typ == 0x3:  # VECT_BASE_X
            vpol = int((w >> 11) & 0x1)
            vbase_x = int(payload & 0x7FF)
            have_vbase = True
        elif typ == 0x4:  # VECT_12
            if not have_y or not have_vbase:
                counters["missing_y"] += int(not have_y)
                counters["missing_vbase"] += int(not have_vbase)
                continue
            valid = int(payload & 0x0FFF)
            ts = (time_high << 12) | time_low
            for i in range(12):
                if (valid >> i) & 0x1:
                    events.append((vbase_x + i, y, ts, vpol))
            vbase_x += 12
        elif typ == 0x5:  # VECT_8
            if not have_y or not have_vbase:
                counters["missing_y"] += int(not have_y)
                counters["missing_vbase"] += int(not have_vbase)
                continue
            valid = int(payload & 0x00FF)
            ts = (time_high << 12) | time_low
            for i in range(8):
                if (valid >> i) & 0x1:
                    events.append((vbase_x + i, y, ts, vpol))
            vbase_x += 8
        elif typ == 0x6:  # EVT_TIME_LOW
            payload_int = int(payload)
            if last_time_low is not None and payload_int < last_time_low:
                if not time_high_updated_since_low:
                    time_high += 1
                    time_high_raw = (time_high_raw + 1) & 0xFFF
                    counters["time_low_wrap_fix"] += 1
            time_low = payload_int
            last_time_low = payload_int
            time_high_updated_since_low = False
        elif typ == 0x8:  # EVT_TIME_HIGH
            payload_int = int(payload)
            if last_time_high_raw is not None and payload_int < last_time_high_raw:
                time_high_base += 0x1000
                counters["time_high_wrap"] += 1
            time_high_raw = payload_int
            time_high = time_high_base + payload_int
            last_time_high_raw = payload_int
            time_high_updated_since_low = True
        elif typ == 0x7:  # CONTINUED_4
            counters["continued_4"] += 1
        elif typ == 0xA:  # EXT_TRIGGER
            counters["ext_trigger"] += 1
        elif typ == 0xE:  # OTHERS
            counters["others"] += 1
        elif typ == 0xF:  # CONTINUED_12
            counters["continued_12"] += 1
        else:
            counters["unknown"] += 1

    if not events:
        return np.empty((0, 4), dtype=np.float32), counters
    return np.asarray(events, dtype=np.float32), counters


def decode_evt3_raw(
    path: Path, endian: str = "little", require_evt3: bool = True
) -> Tuple[np.ndarray, Dict[str, int], Dict[str, str], List[str]]:
    header_lines, data_offset, meta = read_raw_header(path)
    fmt = meta.get("format_name") or meta.get("format", "")
    if require_evt3 and fmt and "EVT3" not in fmt:
        raise ValueError(f"Unsupported format in header: {fmt}")
    with path.open("rb") as f:
        f.seek(data_offset)
        data = f.read()
    events, counters = decode_evt3_bytes(data, endian=endian)
    return events, counters, meta, header_lines
