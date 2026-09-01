from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable
import re

import yaml

_GRID_REP_RE = re.compile(r"^(?P<base>.+)_(?P<grid_x>\d+)x(?P<grid_y>\d+)$", re.IGNORECASE)
_GRID_SPLIT_BASE_REPS = {
    "xy", "xt", "yt", "cstr2", "cstr3", "cstr3_fixed", "xt_my", "yt_mx", "events"
}


def resolve_representation_sequences(data_cfg: Dict[str, Any]) -> Dict[str, Dict[str, int]]:
    """Validate temporal image-sequence settings from the data config."""
    raw = data_cfg.get("representation_sequences") or {}
    if not isinstance(raw, dict):
        raise ValueError("data.representation_sequences must be a mapping.")
    configured = {str(rep) for rep in data_cfg.get("representations", [])}
    result: Dict[str, Dict[str, int]] = {}
    for rep, value in raw.items():
        rep = str(rep)
        if rep not in configured:
            raise ValueError(
                f"data.representation_sequences contains unconfigured representation {rep!r}."
            )
        if value is False or value is None:
            continue
        if isinstance(value, dict):
            length = int(value.get("length", 1))
            stride = int(value.get("stride", 1))
        else:
            length = int(value)
            stride = 1
        if length < 2:
            raise ValueError(f"Sequence length for {rep!r} must be at least 2.")
        if stride < 1:
            raise ValueError(f"Sequence stride for {rep!r} must be at least 1.")
        result[rep] = {"length": length, "stride": stride}
    return result


def load_config(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def _normalize_size(value: Iterable[Any]) -> tuple[int, int]:
    items = list(value)
    if len(items) != 2:
        raise ValueError(f"Expected image size as [width, height], got: {value}")
    return int(items[0]), int(items[1])


def _size_pair_from_config(value: Any, *, name: str) -> tuple[int, int]:
    if isinstance(value, dict):
        if "width" in value or "height" in value:
            return int(value.get("width", value.get("x", 0))), int(value.get("height", value.get("y", 0)))
        if "x" in value or "y" in value:
            return int(value.get("x", 0)), int(value.get("y", 0))
    if isinstance(value, (list, tuple)):
        return _normalize_size(value)
    scalar = int(value)
    return scalar, scalar


def _base_representation(rep: str) -> str:
    match = _GRID_REP_RE.match(str(rep))
    if match is None:
        return str(rep)
    base = match.group("base")
    return base if base in _GRID_SPLIT_BASE_REPS else str(rep)


def resolve_spatial_cutout_config(cutout_cfg: Dict[str, Any] | None, rep: str) -> Dict[str, Any]:
    """Merge the global spatial-cutout defaults with an optional per-rep override."""
    resolved = dict(cutout_cfg or {})
    overrides = resolved.pop("by_representation", {}) or {}
    if not isinstance(overrides, dict):
        raise ValueError("data.spatial_cutout.by_representation must be a mapping.")

    rep_name = str(rep)
    base_rep = _base_representation(rep_name)
    override = overrides.get(rep_name)
    if override is None and base_rep != rep_name:
        override = overrides.get(base_rep)
    if override is None:
        return resolved
    if not isinstance(override, dict):
        raise ValueError(
            f"data.spatial_cutout.by_representation.{rep_name} must be a mapping."
        )
    resolved.update(override)
    return resolved


def _apply_fixed_spatial_cutout_sizes(
    resolved: Dict[str, tuple[int, int]],
    data_cfg: Dict[str, Any],
) -> Dict[str, tuple[int, int]]:
    cutout_cfg = dict(data_cfg.get("spatial_cutout") or {})
    out = dict(resolved)
    for rep, current in resolved.items():
        rep_cutout_cfg = resolve_spatial_cutout_config(cutout_cfg, rep)
        mode = str(rep_cutout_cfg.get("mode", "none")).lower()
        if mode not in {"fixed", "fixed_pixels", "fixed_px"}:
            continue
        size_value = rep_cutout_cfg.get(
            "size_px", rep_cutout_cfg.get("fixed_size_px", rep_cutout_cfg.get("size", None))
        )
        if size_value is None:
            raise ValueError(
                f"data.spatial_cutout fixed mode for representation '{rep}' requires "
                "size_px or fixed_size_px."
            )
        cut_w, cut_h = _size_pair_from_config(
            size_value, name=f"data.spatial_cutout size_px for representation '{rep}'"
        )
        if cut_w <= 0 or cut_h <= 0:
            raise ValueError(
                f"data.spatial_cutout fixed size for representation '{rep}' must be positive."
            )
        if rep.startswith("xt"):
            out[rep] = (cut_w, current[1])
        elif rep.startswith("yt"):
            out[rep] = (current[0], cut_h)
        else:
            out[rep] = (cut_w, cut_h)
    return out


def resolve_representation_source_image_sizes(data_cfg: Dict[str, Any]) -> Dict[str, tuple[int, int]]:
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


def resolve_representation_image_sizes(data_cfg: Dict[str, Any]) -> Dict[str, tuple[int, int]]:
    return _apply_fixed_spatial_cutout_sizes(resolve_representation_source_image_sizes(data_cfg), data_cfg)


def read_split_file(path: Path) -> list[str]:
    out: list[str] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(line.strip("/"))
    return out
