from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable
import re

import yaml

_GRID_REP_RE = re.compile(r"^(?P<base>.+)_(?P<grid_x>\d+)x(?P<grid_y>\d+)$", re.IGNORECASE)
_GRID_SPLIT_BASE_REPS = {"xy", "xt", "yt", "cstr2", "cstr3", "xt_my", "yt_mx", "events"}


def load_config(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def _normalize_size(value: Iterable[Any]) -> tuple[int, int]:
    items = list(value)
    if len(items) != 2:
        raise ValueError(f"Expected image size as [width, height], got: {value}")
    return int(items[0]), int(items[1])


def _base_representation(rep: str) -> str:
    match = _GRID_REP_RE.match(str(rep))
    if match is None:
        return str(rep)
    base = match.group("base")
    return base if base in _GRID_SPLIT_BASE_REPS else str(rep)


def resolve_representation_image_sizes(data_cfg: Dict[str, Any]) -> Dict[str, tuple[int, int]]:
    reps = list(data_cfg.get("representations", []))
    if not reps:
        return {}

    explicit_sizes = data_cfg.get("image_sizes")
    if explicit_sizes:
        resolved = {str(rep): _normalize_size(size) for rep, size in dict(explicit_sizes).items()}
        for rep in reps:
            if rep not in resolved:
                base_rep = _base_representation(rep)
                if base_rep in resolved:
                    resolved[rep] = resolved[base_rep]
        missing = [rep for rep in reps if rep not in resolved]
        if missing:
            raise ValueError(f"Missing image_sizes entries for representations: {missing}")
        return {rep: resolved[rep] for rep in reps}

    legacy_size = data_cfg.get("image_size")
    if legacy_size is not None and not bool(data_cfg.get("retain_spatial_dimensions", False)):
        size = _normalize_size(legacy_size)
        return {rep: size for rep in reps}

    frame_size_raw = data_cfg.get("frame_size")
    if frame_size_raw is None:
        raise ValueError("frame_size is required when retain_spatial_dimensions is enabled.")
    frame_w, frame_h = _normalize_size(frame_size_raw)
    temporal_bins = int(data_cfg.get("temporal_bins", 0))
    if temporal_bins <= 0:
        raise ValueError("temporal_bins must be > 0 when retain_spatial_dimensions is enabled.")

    resolved: Dict[str, tuple[int, int]] = {}
    for rep in reps:
        if rep.startswith("xt"):
            resolved[rep] = (frame_w, temporal_bins)
        elif rep.startswith("yt"):
            resolved[rep] = (temporal_bins, frame_h)
        else:
            resolved[rep] = (frame_w, frame_h)
    return resolved


def read_split_file(path: Path) -> list[str]:
    out: list[str] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(line.strip("/"))
    return out
