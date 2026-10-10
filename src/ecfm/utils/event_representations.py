"""Shared in-memory event image representations used by FRED and render scripts."""
from __future__ import annotations
import re
import numpy as np
from ecfm.data.tokenizer import Region, build_patch
from ecfm.utils.evt3_vis import events_to_image

_GRID_REP_RE = re.compile(r"^(?P<base>.+)_(?P<grid_x>\d+)x(?P<grid_y>\d+)$", re.IGNORECASE)
_GRID_SPLIT_BASE_REPS = {"xy", "xt", "yt", "cstr2", "cstr3", "cstr3_fixed", "xt_my", "yt_mx", "events"}


def render_event_representation(events, representation, *, width, height, start_us,
                                end_us, output_size=None, time_bins=224, patch_size=224,
                                cstr_max_count=None):
    """Return RGB uint8 pixels without writing an image or decoded event file."""
    if end_us <= start_us:
        raise ValueError('Representation window must have positive duration')
    plane, gx, gy = _resolve_representation_alias(representation)
    valid = {'events', 'xy', 'xt', 'yt', 'xy_p45', 'xy_m45', 'yt_p45', 'yt_m45',
             'cstr2', 'cstr3', 'cstr3_fixed', 'xt_my', 'yt_mx'}
    if plane not in valid:
        raise ValueError(f'Unsupported event representation: {representation}')
    if plane == 'cstr3_fixed' and (cstr_max_count is None or cstr_max_count <= 0):
        raise ValueError('cstr3_fixed requires a positive cstr_max_count')
    local = np.asarray(events, dtype=np.float64).copy()
    # Subtract integer timestamps before float conversion whenever possible.
    local[:, 2] = np.asarray(events)[:, 2] - start_us
    if output_size is None:
        output_size = _default_output_size_for_rep(plane, sensor_width=width,
                                                   sensor_height=height, temporal_bins=time_bins)
    if plane == 'events' and gx is None:
        return _resize_to(events_to_image(local, width, height), output_size)
    return _render_histogram_grid(local, width=width, height=height, t0=0, dt=end_us-start_us,
                                  plane='xy' if plane == 'events' else plane,
                                  time_bins=time_bins, patch_size=patch_size,
                                  grid_x=gx or 1, grid_y=gy or 1, output_size=output_size,
                                  cstr3_max_count=cstr_max_count)

def _resolve_representation_alias(rep: str) -> tuple[str, int | None, int | None]:
    match = _GRID_REP_RE.match(rep)
    if match is None:
        return rep, None, None
    base = match.group("base")
    if base not in _GRID_SPLIT_BASE_REPS:
        return rep, None, None
    grid_x = int(match.group("grid_x"))
    grid_y = int(match.group("grid_y"))
    if grid_x <= 0 or grid_y <= 0:
        raise ValueError(f"Grid dimensions in representation '{rep}' must be positive.")
    return base, grid_x, grid_y


def _patch_to_rgb(patch: np.ndarray, *, time_horizontal: bool = False) -> np.ndarray:
    if patch.ndim != 3 or patch.shape[0] != 2:
        raise ValueError("patch must be shaped [2, H, W]")
    # If time is horizontal, swap (T, Y) -> (Y, T) so time runs left->right.
    if time_horizontal:
        patch = patch.transpose(0, 2, 1)
    p0 = np.clip(patch[0], 0.0, 1.0)
    p1 = np.clip(patch[1], 0.0, 1.0)
    h, w = p0.shape
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :, 0] = (p1 * 255.0).astype(np.uint8)
    img[:, :, 2] = (p0 * 255.0).astype(np.uint8)
    return img


def _resize_rgb(img: np.ndarray, patch_size: int) -> np.ndarray:
    if img.shape[0] == patch_size and img.shape[1] == patch_size:
        return img
    return _resize_to(img, (patch_size, patch_size))


def _resize_gray(img: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    if img.shape[0] == size[1] and img.shape[1] == size[0]:
        return img
    try:
        import torch
        import torch.nn.functional as F

        t = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float()
        t = F.interpolate(t, size=(size[1], size[0]), mode="bilinear", align_corners=False)
        out = t.squeeze(0).squeeze(0).clamp(0, 1).numpy()
        return out
    except Exception:
        return img


def _resize_to(img: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    if img.shape[1] == size[0] and img.shape[0] == size[1]:
        return img
    try:
        import torch
        import torch.nn.functional as F

        t = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).float()
        t = F.interpolate(t, size=(size[1], size[0]), mode="bilinear", align_corners=False)
        out = t.squeeze(0).permute(1, 2, 0).clamp(0, 255).byte().numpy()
        return out
    except Exception:
        return img


def _default_output_size_for_rep(
    rep: str,
    *,
    sensor_width: int,
    sensor_height: int,
    temporal_bins: int,
) -> tuple[int, int]:
    if rep.startswith("xt"):
        return sensor_width, temporal_bins
    if rep.startswith("yt"):
        return temporal_bins, sensor_height
    return sensor_width, sensor_height


def _cstr_patch(
    events: np.ndarray,
    region: Region,
    *,
    patch_size: int,
    output_size: tuple[int, int] | None = None,
    include_count: bool,
    max_count: float | None = None,
) -> np.ndarray:
    # events are expected to be [x, y, t, p] with t in the same units as region.t/dt
    mask = (
        (events[:, 0] >= region.x)
        & (events[:, 0] < region.x + region.dx)
        & (events[:, 1] >= region.y)
        & (events[:, 1] < region.y + region.dy)
        & (events[:, 2] >= region.t)
        & (events[:, 2] < region.t + region.dt)
    )
    sub = events[mask]
    h = region.dy
    w = region.dx
    if sub.shape[0] == 0:
        img = np.zeros((h, w, 3), dtype=np.uint8)
        if output_size is not None:
            return _resize_to(img, output_size)
        return _resize_rgb(img, patch_size)

    t_norm = (sub[:, 2] - region.t) / max(region.dt, 1e-6)
    t_norm = np.clip(t_norm, 0.0, 1.0)
    x = (sub[:, 0] - region.x).astype(np.int64)
    y = (sub[:, 1] - region.y).astype(np.int64)
    p = sub[:, 3].astype(np.int64)

    sum_pos = np.zeros((h, w), dtype=np.float32)
    sum_neg = np.zeros((h, w), dtype=np.float32)
    cnt_pos = np.zeros((h, w), dtype=np.float32)
    cnt_neg = np.zeros((h, w), dtype=np.float32)

    pos_mask = p == 1
    neg_mask = ~pos_mask
    if np.any(pos_mask):
        np.add.at(sum_pos, (y[pos_mask], x[pos_mask]), t_norm[pos_mask])
        np.add.at(cnt_pos, (y[pos_mask], x[pos_mask]), 1.0)
    if np.any(neg_mask):
        np.add.at(sum_neg, (y[neg_mask], x[neg_mask]), t_norm[neg_mask])
        np.add.at(cnt_neg, (y[neg_mask], x[neg_mask]), 1.0)

    mean_pos = np.zeros((h, w), dtype=np.float32)
    mean_neg = np.zeros((h, w), dtype=np.float32)
    if np.any(cnt_pos > 0):
        mean_pos = np.divide(sum_pos, cnt_pos, out=mean_pos, where=cnt_pos > 0)
    if np.any(cnt_neg > 0):
        mean_neg = np.divide(sum_neg, cnt_neg, out=mean_neg, where=cnt_neg > 0)

    img = np.zeros((h, w, 3), dtype=np.float32)
    img[:, :, 0] = mean_pos
    img[:, :, 2] = mean_neg
    if include_count:
        cnt = cnt_pos + cnt_neg
        maxv = float(max_count) if max_count is not None else float(cnt.max()) if cnt.size else 0.0
        if maxv > 0:
            img[:, :, 1] = np.clip(cnt / maxv, 0.0, 1.0)

    img = np.clip(img * 255.0, 0.0, 255.0).astype(np.uint8)
    if output_size is not None:
        return _resize_to(img, output_size)
    return _resize_rgb(img, patch_size)


def _mean_axis_map(
    events: np.ndarray,
    region: Region,
    *,
    time_bins: int,
    axis: str,
) -> np.ndarray:
    mask = (
        (events[:, 0] >= region.x)
        & (events[:, 0] < region.x + region.dx)
        & (events[:, 1] >= region.y)
        & (events[:, 1] < region.y + region.dy)
        & (events[:, 2] >= region.t)
        & (events[:, 2] < region.t + region.dt)
    )
    sub = events[mask]
    if sub.shape[0] == 0:
        if axis == "y":
            return np.zeros((time_bins, region.dx), dtype=np.float32)
        if axis == "x":
            return np.zeros((time_bins, region.dy), dtype=np.float32)
        raise ValueError(f"Unknown axis: {axis}")

    t_norm = (sub[:, 2] - region.t) / max(region.dt, 1e-6)
    t_idx = np.clip((t_norm * time_bins).astype(np.int64), 0, time_bins - 1)

    if axis == "y":
        x = (sub[:, 0] - region.x).astype(np.int64)
        y_norm = (sub[:, 1] - region.y) / max(region.dy - 1, 1)
        y_norm = np.clip(y_norm, 0.0, 1.0)
        sums = np.zeros((time_bins, region.dx), dtype=np.float32)
        counts = np.zeros((time_bins, region.dx), dtype=np.float32)
        np.add.at(sums, (t_idx, x), y_norm)
        np.add.at(counts, (t_idx, x), 1.0)
    elif axis == "x":
        y = (sub[:, 1] - region.y).astype(np.int64)
        x_norm = (sub[:, 0] - region.x) / max(region.dx - 1, 1)
        x_norm = np.clip(x_norm, 0.0, 1.0)
        sums = np.zeros((time_bins, region.dy), dtype=np.float32)
        counts = np.zeros((time_bins, region.dy), dtype=np.float32)
        np.add.at(sums, (t_idx, y), x_norm)
        np.add.at(counts, (t_idx, y), 1.0)
    else:
        raise ValueError(f"Unknown axis: {axis}")

    mean = np.zeros_like(sums, dtype=np.float32)
    mask_nonzero = counts > 0
    mean[mask_nonzero] = sums[mask_nonzero] / counts[mask_nonzero]
    return mean


def _render_histogram_grid(
    events: np.ndarray,
    *,
    width: int,
    height: int,
    t0: float,
    dt: float,
    plane: str,
    time_bins: int,
    patch_size: int,
    grid_x: int,
    grid_y: int,
    retain_spatial_dimensions: bool = False,
    output_size: tuple[int, int] | None = None,
    cstr3_max_count: float | None = None,
) -> np.ndarray:
    x_edges = np.linspace(0, width, grid_x + 1, dtype=np.int64)
    y_edges = np.linspace(0, height, grid_y + 1, dtype=np.int64)

    def _cell_size(gx: int, gy: int) -> tuple[int, int]:
        dx = max(1, int(x_edges[gx + 1]) - int(x_edges[gx]))
        dy = max(1, int(y_edges[gy + 1]) - int(y_edges[gy]))
        if output_size is not None:
            out_w, out_h = output_size
            x_out_edges = np.linspace(0, out_w, grid_x + 1, dtype=np.int64)
            y_out_edges = np.linspace(0, out_h, grid_y + 1, dtype=np.int64)
            return (
                max(1, int(x_out_edges[gx + 1]) - int(x_out_edges[gx])),
                max(1, int(y_out_edges[gy + 1]) - int(y_out_edges[gy])),
            )
        if not retain_spatial_dimensions:
            return patch_size, patch_size
        if plane in {"xt", "xt_my"}:
            return dx, time_bins
        if plane in {"yt", "yt_mx"}:
            return time_bins, dy
        return dx, dy

    col_widths = [_cell_size(gx, 0)[0] for gx in range(grid_x)]
    row_heights = [_cell_size(0, gy)[1] for gy in range(grid_y)]
    out_h = sum(row_heights)
    out_w = sum(col_widths)
    canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    x_offsets = np.cumsum([0] + col_widths[:-1]).tolist()
    y_offsets = np.cumsum([0] + row_heights[:-1]).tolist()

    for gy in range(grid_y):
        for gx in range(grid_x):
            x0 = int(x_edges[gx])
            x1 = int(x_edges[gx + 1])
            y0 = int(y_edges[gy])
            y1 = int(y_edges[gy + 1])
            dx = max(1, x1 - x0)
            dy = max(1, y1 - y0)
            cell_w, cell_h = _cell_size(gx, gy)
            region = Region(
                x=x0,
                y=y0,
                t=t0,
                dx=dx,
                dy=dy,
                dt=dt,
                plane=plane,
            )
            if plane in {"cstr2", "cstr3", "cstr3_fixed"}:
                patch_img = _cstr_patch(
                    events,
                    region,
                    patch_size=patch_size,
                    output_size=(cell_w, cell_h),
                    include_count=plane in {"cstr3", "cstr3_fixed"},
                    max_count=cstr3_max_count if plane == "cstr3_fixed" else None,
                )
            elif plane in {"xt_my", "yt_mx"}:
                base_plane = "xt" if plane == "xt_my" else "yt"
                time_horizontal = plane.startswith("yt")
                patch_output_size = (
                    (cell_w, cell_h) if time_horizontal else (cell_h, cell_w)
                )
                region_base = Region(
                    x=region.x,
                    y=region.y,
                    t=region.t,
                    dx=region.dx,
                    dy=region.dy,
                    dt=region.dt,
                    plane=base_plane,
                )
                patch_t, _ = build_patch(
                    events,
                    region_base,
                    patch_size=patch_size,
                    time_bins=time_bins,
                    output_size=patch_output_size,
                )
                patch = patch_t.detach().cpu().numpy()
                patch_img = _patch_to_rgb(patch, time_horizontal=time_horizontal)

                axis = "y" if plane == "xt_my" else "x"
                mean_map = _mean_axis_map(events, region, time_bins=time_bins, axis=axis)
                mean_map_size = (cell_h, cell_w) if time_horizontal else (cell_w, cell_h)
                mean_map = _resize_gray(mean_map, mean_map_size)
                if time_horizontal:
                    mean_map = mean_map.T
                patch_img = patch_img.copy()
                patch_img[:, :, 1] = np.clip(mean_map * 255.0, 0, 255).astype(np.uint8)
            else:
                time_horizontal = plane.startswith("yt")
                patch_output_size = (
                    (cell_w, cell_h) if time_horizontal else (cell_h, cell_w)
                )
                patch_t, _ = build_patch(
                    events,
                    region,
                    patch_size=patch_size,
                    time_bins=time_bins,
                    output_size=patch_output_size,
                )
                patch = patch_t.detach().cpu().numpy()
                patch_img = _patch_to_rgb(patch, time_horizontal=time_horizontal)
            x_start = x_offsets[gx]
            y_start = y_offsets[gy]
            canvas[y_start : y_start + cell_h, x_start : x_start + cell_w] = patch_img
    return canvas

