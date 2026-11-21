"""Build conservative single-object tracks where FRED has box coordinates only."""

from __future__ import annotations

import hashlib
import math
import os
from itertools import groupby
from pathlib import Path

from scripts.build_cleaned_tracks_from_coordinates import _read_coordinates, build_track_lines


def coordinate_tracks(data: dict, folder: str) -> Path | None:
    directory = data.get("coordinate_track_cache")
    if not directory:
        return None
    source = Path(data["labels_root"]) / folder / "coordinates.txt"
    if not source.is_file():
        return None
    stat = source.stat()
    period = float(data["label_period_s"])
    frame_w, frame_h = data["frame_size"]
    needed = sum(round(data[key] / (1000 * period)) for key in ("history_ms", "forecast_ms"))
    method = data.get("coordinate_track_method", "conservative")
    source_key = f"{source.resolve()}:{stat.st_size}:{stat.st_mtime_ns}:{period}"
    if method == "cleaned_script":
        cache_key = f"cleaned_script_v1:{source_key}"
    elif method == "conservative":
        cache_key = f"v1:{source_key}:{frame_w}:{frame_h}:{needed}"
    else:
        raise ValueError(f"Unknown coordinate_track_method: {method}")
    signature = hashlib.sha256(cache_key.encode()).hexdigest()[:20]
    output = Path(directory) / f"{folder}_{signature}.txt"
    if output.exists():
        return output if output.stat().st_size else None

    if method == "cleaned_script":
        content = "\n".join(build_track_lines(source, int(folder), dt=period))
    else:
        content = _conservative_tracks(source, int(folder), period, frame_w, frame_h, needed)

    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(f".{os.getpid()}.tmp")
    temp.write_text(content, encoding="utf-8")
    temp.replace(output)
    return output if content else None


def _conservative_tracks(source: Path, sequence: int, period: float, frame_w: int, frame_h: int,
                         needed: int) -> str:
    result = []
    segment = []
    track_id = sequence * 100000

    def flush():
        nonlocal track_id
        if len(segment) >= needed + 1:
            result.extend(f"{t},{track_id},{x},{y},{w},{h}\n" for t, x, y, w, h in segment)
            track_id += 1
        segment.clear()

    for t, group in groupby(_read_coordinates(source), key=lambda row: row.t):
        rows = list(group)
        if len(rows) != 1:
            flush()
            continue
        row = rows[0]
        x, y, w, h = row.x0, row.y0, row.x1 - row.x0, row.y1 - row.y0
        if not all(map(math.isfinite, (t, x, y, w, h))) or t < 0 or not (
            0 <= x < x + w <= frame_w + 1 and 0 <= y < y + h <= frame_h + 1
        ):
            flush()
            continue
        if segment:
            prev_t, prev_x, prev_y, prev_w, prev_h = segment[-1]
            step = math.hypot(x + w / 2 - prev_x - prev_w / 2, y + h / 2 - prev_y - prev_h / 2)
            area_ratio = w * h / (prev_w * prev_h)
            adjacent = 0 < t - prev_t <= 1.5 * period
            max_step = max(45, 1.25 * max(w, h, prev_w, prev_h))
            if not adjacent or step > max_step or not 0.30 <= area_ratio <= 3.3:
                flush()
        segment.append((t, x, y, w, h))
    flush()

    return "".join(result)
