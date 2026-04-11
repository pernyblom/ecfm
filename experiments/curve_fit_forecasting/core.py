from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np
import torch

from scripts.fit_event_trace import (
    build_signal_image,
    clip_curve_to_spatial,
    extract_trace_and_green,
    poly_derivative,
    poly_eval,
    preprocess_mask,
)


@dataclass
class PlaneFitResult:
    coeffs: np.ndarray
    spatial_size: int
    time_len: int
    time_axis: str
    history_times_px: np.ndarray
    raw_mask: np.ndarray
    guided_mask: np.ndarray
    detected_times: np.ndarray
    detected_coords: np.ndarray
    history_rmse_px: float
    source: str
    observed_fit: np.ndarray


@dataclass
class CurveForecastResult:
    pred_boxes: torch.Tensor
    pred_centers_xy: torch.Tensor
    fit_centers_xy: torch.Tensor
    xt: PlaneFitResult
    yt: PlaneFitResult
    debug: Dict[str, Any]


def _fit_polynomial_weighted(
    times: np.ndarray, coords: np.ndarray, degree: int, weights: np.ndarray | None = None
) -> np.ndarray:
    unique_times = np.unique(times)
    eff_degree = int(min(max(0, degree), max(0, unique_times.size - 1)))
    if eff_degree == 0:
        return np.asarray([float(np.average(coords, weights=weights))], dtype=np.float64)
    return np.polyfit(times, coords, eff_degree, w=weights)


def _predict_from_end_state(
    observed_times: np.ndarray,
    coeffs: np.ndarray,
    future_query_times: np.ndarray,
) -> np.ndarray:
    if future_query_times.size == 0:
        return np.zeros((0,), dtype=np.float32)
    t_end = float(np.max(observed_times))
    s_poly = np.poly1d(coeffs)
    d1 = poly_derivative(coeffs, 1)
    d2 = poly_derivative(coeffs, 2) if len(coeffs) >= 3 else np.poly1d([0.0])
    s0 = float(s_poly(t_end))
    v0 = float(d1(t_end))
    a0 = float(d2(t_end))
    dt = future_query_times.astype(np.float64) - t_end
    pred = s0 + v0 * dt + 0.5 * a0 * dt * dt
    return pred.astype(np.float32)


def _map_times_to_image_axis(
    times_s: np.ndarray,
    anchor_time_s: float,
    image_window_ms: float,
    time_len: int,
    window_mode: str,
) -> np.ndarray:
    window_s = float(image_window_ms) / 1000.0
    if window_mode == "trailing":
        t0 = anchor_time_s - window_s
    elif window_mode == "center":
        t0 = anchor_time_s - window_s / 2.0
    elif window_mode == "leading":
        t0 = anchor_time_s
    else:
        raise ValueError(f"Unknown window_mode: {window_mode}")
    scale = max(float(time_len - 1), 1.0) / max(window_s, 1.0e-6)
    return (times_s - t0) * scale


def _build_history_corridor_mask(
    mask_shape: tuple[int, int],
    time_axis: str,
    history_times_px: np.ndarray,
    history_centers_px: np.ndarray,
    history_half_sizes_px: np.ndarray,
    temporal_slack_px: float,
    spatial_slack_px: float,
    size_scale: float,
) -> np.ndarray:
    h, w = mask_shape
    time_len = w if time_axis == "x" else h
    spatial_len = h if time_axis == "x" else w
    out = np.zeros(mask_shape, dtype=bool)

    if history_times_px.size == 0:
        out[:] = True
        return out

    order = np.argsort(history_times_px)
    times = history_times_px[order]
    centers = history_centers_px[order]
    radii = np.maximum(history_half_sizes_px[order] * size_scale + spatial_slack_px, 1.0)

    if times.size == 1:
        t_values = np.arange(
            int(np.floor(times[0] - temporal_slack_px)),
            int(np.ceil(times[0] + temporal_slack_px)) + 1,
        )
        c_values = np.full_like(t_values, centers[0], dtype=np.float64)
        r_values = np.full_like(t_values, radii[0], dtype=np.float64)
    else:
        t_all = []
        c_all = []
        r_all = []
        for idx in range(times.size - 1):
            lo = int(np.floor(min(times[idx], times[idx + 1]) - temporal_slack_px))
            hi = int(np.ceil(max(times[idx], times[idx + 1]) + temporal_slack_px))
            if hi < lo:
                continue
            t_seg = np.arange(lo, hi + 1)
            dt = max(abs(times[idx + 1] - times[idx]), 1.0e-6)
            alpha = np.clip((t_seg - times[idx]) / dt, 0.0, 1.0)
            c_seg = centers[idx] + alpha * (centers[idx + 1] - centers[idx])
            r_seg = radii[idx] + alpha * (radii[idx + 1] - radii[idx])
            t_all.append(t_seg)
            c_all.append(c_seg)
            r_all.append(r_seg)
        t_values = np.concatenate(t_all) if t_all else np.asarray([], dtype=np.int64)
        c_values = np.concatenate(c_all) if c_all else np.asarray([], dtype=np.float64)
        r_values = np.concatenate(r_all) if r_all else np.asarray([], dtype=np.float64)

    for t, center, radius in zip(t_values, c_values, r_values):
        ti = int(np.clip(t, 0, time_len - 1))
        s0 = max(0, int(np.floor(center - radius)))
        s1 = min(spatial_len - 1, int(np.ceil(center + radius)))
        if time_axis == "x":
            out[s0 : s1 + 1, ti] = True
        else:
            out[ti, s0 : s1 + 1] = True
    return out


def _interp_history_support(
    query_times_px: np.ndarray,
    history_times_px: np.ndarray,
    history_centers_px: np.ndarray,
    history_half_sizes_px: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if query_times_px.size == 0:
        return (
            np.zeros((0,), dtype=np.float64),
            np.zeros((0,), dtype=np.float64),
        )
    order = np.argsort(history_times_px)
    times = history_times_px[order].astype(np.float64)
    centers = history_centers_px[order].astype(np.float64)
    radii = history_half_sizes_px[order].astype(np.float64)
    if times.size == 1:
        expected = np.full(query_times_px.shape, centers[0], dtype=np.float64)
        support = np.full(query_times_px.shape, radii[0], dtype=np.float64)
        return expected, support
    clipped_times = np.clip(query_times_px.astype(np.float64), times[0], times[-1])
    expected = np.interp(clipped_times, times, centers)
    support = np.interp(clipped_times, times, radii)
    return expected, support


def _filter_points_by_history(
    times: np.ndarray,
    coords: np.ndarray,
    history_times_px: np.ndarray,
    history_centers_px: np.ndarray,
    history_half_sizes_px: np.ndarray,
    params: Dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    if times.size == 0:
        return times, coords
    expected, support = _interp_history_support(
        times,
        history_times_px=history_times_px,
        history_centers_px=history_centers_px,
        history_half_sizes_px=history_half_sizes_px,
    )
    max_dev = (
        support * float(params.get("history_size_scale", 1.25))
        + float(params.get("history_spatial_slack_px", 8.0))
        + float(params.get("max_point_deviation_extra_px", 12.0))
    )
    keep = np.abs(coords.astype(np.float64) - expected) <= np.maximum(max_dev, 1.0)
    return times[keep], coords[keep]


def _fit_plane_guided(
    image_path: Path,
    time_axis: str,
    history_times_px: np.ndarray,
    history_centers_px: np.ndarray,
    history_half_sizes_px: np.ndarray,
    params: Dict[str, Any],
) -> PlaneFitResult:
    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    if str(params.get("point_source", "image+history")).lower() == "history_only":
        history_weights = np.full(
            history_times_px.shape,
            float(params.get("history_point_weight", 4.0)),
            dtype=np.float64,
        )
        coeffs = _fit_polynomial_weighted(
            history_times_px.astype(np.float64),
            history_centers_px.astype(np.float64),
            degree=int(params.get("fit_degree", 2)),
            weights=history_weights,
        )
        history_fit = poly_eval(coeffs, history_times_px.astype(np.float64))
        history_rmse_px = float(
            np.sqrt(np.mean((history_fit - history_centers_px.astype(np.float64)) ** 2))
        )
        spatial_size = bgr.shape[0] if time_axis == "x" else bgr.shape[1]
        time_len = bgr.shape[1] if time_axis == "x" else bgr.shape[0]
        observed_fit = clip_curve_to_spatial(
            poly_eval(coeffs, history_times_px.astype(np.float64)), spatial_size
        )
        empty_mask = np.zeros(bgr.shape[:2], dtype=np.uint8)
        return PlaneFitResult(
            coeffs=np.asarray(coeffs, dtype=np.float64),
            spatial_size=int(spatial_size),
            time_len=int(time_len),
            time_axis=time_axis,
            history_times_px=np.asarray(history_times_px, dtype=np.float32),
            raw_mask=empty_mask,
            guided_mask=empty_mask,
            detected_times=np.zeros((0,), dtype=np.float32),
            detected_coords=np.zeros((0,), dtype=np.float32),
            history_rmse_px=history_rmse_px,
            source="history_only",
            observed_fit=np.asarray(observed_fit, dtype=np.float32),
        )

    signal = build_signal_image(bgr, source=params.get("signal_source", "rbmax"))
    blurred, raw_mask = preprocess_mask(
        signal=signal,
        blur_ksize=int(params.get("blur_ksize", 5)),
        blur_sigma=float(params.get("blur_sigma", 1.2)),
        threshold_mode=params.get("threshold_mode", "otsu"),
        threshold_value=float(params.get("threshold_value", 40.0)),
        invert=bool(params.get("invert", False)),
        morph_open=int(params.get("morph_open", 3)),
        morph_close=int(params.get("morph_close", 3)),
        min_component_area=int(params.get("min_component_area", 20)),
    )
    guided_region = _build_history_corridor_mask(
        raw_mask.shape,
        time_axis=time_axis,
        history_times_px=history_times_px,
        history_centers_px=history_centers_px,
        history_half_sizes_px=history_half_sizes_px,
        temporal_slack_px=float(params.get("history_time_slack_px", 2.0)),
        spatial_slack_px=float(params.get("history_spatial_slack_px", 8.0)),
        size_scale=float(params.get("history_size_scale", 1.25)),
    )
    guided_mask = np.where(guided_region, raw_mask > 0, False).astype(np.uint8) * 255
    green = bgr[:, :, 1]

    times, coords, _ = extract_trace_and_green(
        mask=guided_mask,
        signal=blurred,
        green_channel=green,
        time_axis=time_axis,
        coord_method=params.get("coord_method", "weighted_centroid"),
        neighborhood_radius=int(params.get("green_neighborhood_radius", 1)),
    )
    times, coords = _filter_points_by_history(
        times,
        coords,
        history_times_px=history_times_px,
        history_centers_px=history_centers_px,
        history_half_sizes_px=history_half_sizes_px,
        params=params,
    )
    source = "guided"
    if times.size < int(params.get("min_event_points", 12)) and bool(params.get("retry_without_guidance", True)):
        times, coords, _ = extract_trace_and_green(
            mask=raw_mask,
            signal=blurred,
            green_channel=green,
            time_axis=time_axis,
            coord_method=params.get("coord_method", "weighted_centroid"),
            neighborhood_radius=int(params.get("green_neighborhood_radius", 1)),
        )
        times, coords = _filter_points_by_history(
            times,
            coords,
            history_times_px=history_times_px,
            history_centers_px=history_centers_px,
            history_half_sizes_px=history_half_sizes_px,
            params=params,
        )
        source = "unguided_filtered"

    history_weights = np.full(
        history_times_px.shape,
        float(params.get("history_point_weight", 4.0)),
        dtype=np.float64,
    )
    history_coeffs = _fit_polynomial_weighted(
        history_times_px.astype(np.float64),
        history_centers_px.astype(np.float64),
        degree=int(params.get("fit_degree", 2)),
        weights=history_weights,
    )

    if times.size >= int(params.get("min_event_points", 12)):
        fit_times = np.concatenate([times.astype(np.float64), history_times_px.astype(np.float64)])
        fit_coords = np.concatenate([coords.astype(np.float64), history_centers_px.astype(np.float64)])
        weights = np.concatenate(
            [
                np.full(times.shape, float(params.get("event_point_weight", 1.0)), dtype=np.float64),
                history_weights,
            ]
        )
        coeffs = _fit_polynomial_weighted(
            fit_times,
            fit_coords,
            degree=int(params.get("fit_degree", 2)),
            weights=weights,
        )
    else:
        coeffs = history_coeffs
        source = "history_only_sparse"

    history_fit = poly_eval(coeffs, history_times_px.astype(np.float64))
    history_rmse_px = float(np.sqrt(np.mean((history_fit - history_centers_px.astype(np.float64)) ** 2)))
    if bool(params.get("fallback_to_history", True)) and history_rmse_px > float(
        params.get("max_history_rmse_px", 30.0)
    ):
        coeffs = history_coeffs
        source = "history_only_rejected"
        history_fit = poly_eval(coeffs, history_times_px.astype(np.float64))
        history_rmse_px = float(np.sqrt(np.mean((history_fit - history_centers_px.astype(np.float64)) ** 2)))

    spatial_size = raw_mask.shape[0] if time_axis == "x" else raw_mask.shape[1]
    time_len = raw_mask.shape[1] if time_axis == "x" else raw_mask.shape[0]
    observed_fit = clip_curve_to_spatial(poly_eval(coeffs, history_times_px.astype(np.float64)), spatial_size)
    return PlaneFitResult(
        coeffs=np.asarray(coeffs, dtype=np.float64),
        spatial_size=int(spatial_size),
        time_len=int(time_len),
        time_axis=time_axis,
        history_times_px=np.asarray(history_times_px, dtype=np.float32),
        raw_mask=raw_mask,
        guided_mask=guided_mask,
        detected_times=np.asarray(times, dtype=np.float32),
        detected_coords=np.asarray(coords, dtype=np.float32),
        history_rmse_px=history_rmse_px,
        source=source,
        observed_fit=np.asarray(observed_fit, dtype=np.float32),
    )


def _predict_box_sizes(past_boxes: torch.Tensor, strategy: str) -> torch.Tensor:
    sizes = past_boxes[:, 2:].float()
    if strategy == "last":
        base = sizes[-1]
    elif strategy == "median":
        base = sizes.median(dim=0).values
    elif strategy == "mean":
        base = sizes.mean(dim=0)
    else:
        raise ValueError(f"Unknown size strategy: {strategy}")
    return base


def forecast_sample(
    xt_path: Path,
    yt_path: Path,
    past_boxes: torch.Tensor,
    past_times_s: torch.Tensor,
    future_times_s: torch.Tensor,
    anchor_time_s: float,
    image_window_ms: float,
    image_window_mode: str,
    params: Dict[str, Any],
) -> CurveForecastResult:
    xt_img = cv2.imread(str(xt_path), cv2.IMREAD_COLOR)
    yt_img = cv2.imread(str(yt_path), cv2.IMREAD_COLOR)
    if xt_img is None or yt_img is None:
        raise FileNotFoundError(f"Missing input image(s): {xt_path} {yt_path}")

    past_boxes_np = past_boxes.detach().cpu().numpy().astype(np.float64)
    past_times_np = past_times_s.detach().cpu().numpy().astype(np.float64)
    future_times_np = future_times_s.detach().cpu().numpy().astype(np.float64)

    xt_time_len = xt_img.shape[0]
    yt_time_len = yt_img.shape[1]
    xt_spatial = xt_img.shape[1]
    yt_spatial = yt_img.shape[0]

    xt_history_times_px = _map_times_to_image_axis(
        past_times_np, anchor_time_s, image_window_ms, xt_time_len, image_window_mode
    )
    yt_history_times_px = _map_times_to_image_axis(
        past_times_np, anchor_time_s, image_window_ms, yt_time_len, image_window_mode
    )
    xt_future_times_px = _map_times_to_image_axis(
        future_times_np, anchor_time_s, image_window_ms, xt_time_len, image_window_mode
    )
    yt_future_times_px = _map_times_to_image_axis(
        future_times_np, anchor_time_s, image_window_ms, yt_time_len, image_window_mode
    )

    xt_result = _fit_plane_guided(
        xt_path,
        time_axis="y",
        history_times_px=xt_history_times_px,
        history_centers_px=past_boxes_np[:, 0] * max(1.0, xt_spatial - 1),
        history_half_sizes_px=past_boxes_np[:, 2] * xt_spatial / 2.0,
        params=params,
    )
    yt_result = _fit_plane_guided(
        yt_path,
        time_axis="x",
        history_times_px=yt_history_times_px,
        history_centers_px=past_boxes_np[:, 1] * max(1.0, yt_spatial - 1),
        history_half_sizes_px=past_boxes_np[:, 3] * yt_spatial / 2.0,
        params=params,
    )

    pred_x_px = clip_curve_to_spatial(
        _predict_from_end_state(xt_history_times_px, xt_result.coeffs, xt_future_times_px),
        xt_result.spatial_size,
    )
    pred_y_px = clip_curve_to_spatial(
        _predict_from_end_state(yt_history_times_px, yt_result.coeffs, yt_future_times_px),
        yt_result.spatial_size,
    )
    pred_x = torch.from_numpy(pred_x_px / max(1.0, xt_result.spatial_size - 1)).float()
    pred_y = torch.from_numpy(pred_y_px / max(1.0, yt_result.spatial_size - 1)).float()
    pred_centers_xy = torch.stack([pred_x, pred_y], dim=-1).clamp(0.0, 1.0)
    fit_x = torch.from_numpy(
        np.asarray(xt_result.observed_fit, dtype=np.float32) / max(1.0, xt_result.spatial_size - 1)
    ).float()
    fit_y = torch.from_numpy(
        np.asarray(yt_result.observed_fit, dtype=np.float32) / max(1.0, yt_result.spatial_size - 1)
    ).float()
    fit_centers_xy = torch.stack([fit_x, fit_y], dim=-1).clamp(0.0, 1.0)

    base_size = _predict_box_sizes(
        past_boxes, strategy=str(params.get("size_strategy", "mean"))
    ).clamp(0.0, 1.0)
    pred_sizes = base_size.unsqueeze(0).repeat(pred_centers_xy.shape[0], 1)
    pred_boxes = torch.cat([pred_centers_xy, pred_sizes], dim=-1).clamp(0.0, 1.0)

    debug = {
        "xt_source": xt_result.source,
        "yt_source": yt_result.source,
        "xt_history_rmse_px": xt_result.history_rmse_px,
        "yt_history_rmse_px": yt_result.history_rmse_px,
        "xt_event_points": int(xt_result.detected_times.size),
        "yt_event_points": int(yt_result.detected_times.size),
    }
    return CurveForecastResult(
        pred_boxes=pred_boxes,
        pred_centers_xy=pred_centers_xy,
        fit_centers_xy=fit_centers_xy,
        xt=xt_result,
        yt=yt_result,
        debug=debug,
    )
