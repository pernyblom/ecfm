#!/usr/bin/env python3
"""
fit_event_xy_trace.py

Fit paired x(t) and y(t) traces from XT and YT event images, then render
observed and predicted curves in XT, YT, and XY.

This script reuses the single-plane fitting utilities from fit_event_trace.py,
but it does not estimate cross-planes from green-channel metadata. Detection is
normally run with `rbmax` so the green channel does not influence the extracted
trace.

Example:
    python scripts/fit_event_xy_trace.py \
        --xt-input outputs/fred_reps/0/Video_0_frame_100032333_xt_my.png \
        --yt-input outputs/fred_reps/0/Video_0_frame_100032333_yt_mx.png \
        --out-prefix outputs/trace_xy/run1 \
        --future-steps 0
"""

from __future__ import annotations

import argparse
import cv2
import numpy as np

from fit_event_trace import (
    clip_curve_to_spatial,
    clip_points,
    draw_points,
    draw_polyline,
    ensure_dir_for_prefix,
    extract_fit_predict_trace,
    infer_plane_and_time_axis,
    make_canvas,
    poly_eval,
)


def draw_xy_polyline(img, xs, ys, color=(255, 255, 0), thickness=2):
    h, w = img.shape[:2]
    pts = clip_points(list(zip(xs, ys)), w, h)
    if len(pts) >= 2:
        cv2.polylines(img, [np.array(pts, dtype=np.int32)], False, color, thickness, cv2.LINE_AA)


def draw_xy_points(img, xs, ys, color=(0, 255, 255), radius=1):
    h, w = img.shape[:2]
    for x, y in zip(xs, ys):
        if not np.isfinite(x) or not np.isfinite(y):
            continue
        xi = int(round(np.clip(x, 0, w - 1)))
        yi = int(round(np.clip(y, 0, h - 1)))
        cv2.circle(img, (xi, yi), radius, color, -1, cv2.LINE_AA)


def parse_args():
    p = argparse.ArgumentParser(description="Fit paired XT/YT traces and predict future XY curve.")
    p.add_argument("--xt-input", required=True, help="Path to XT image, e.g. *_xt_my.png")
    p.add_argument("--yt-input", required=True, help="Path to YT image, e.g. *_yt_mx.png")
    p.add_argument("--out-prefix", default="output/xy_trace", help="Output file prefix")
    p.add_argument("--sensor-width", type=int, default=None, help="Optional XY canvas width override")
    p.add_argument("--sensor-height", type=int, default=None, help="Optional XY canvas height override")
    p.add_argument("--signal-source", default="rbmax", choices=["gray", "max", "rbmax", "green", "red", "blue"])
    p.add_argument("--blur-ksize", type=int, default=5)
    p.add_argument("--blur-sigma", type=float, default=1.2)
    p.add_argument("--threshold-mode", default="otsu", choices=["otsu", "fixed", "adaptive"])
    p.add_argument("--threshold-value", type=float, default=40.0)
    p.add_argument("--invert", action="store_true")
    p.add_argument("--morph-open", type=int, default=3)
    p.add_argument("--morph-close", type=int, default=3)
    p.add_argument("--min-component-area", type=int, default=20)
    p.add_argument("--coord-method", default="weighted_centroid",
                   choices=["weighted_centroid", "centroid", "median", "argmax"])
    p.add_argument("--green-neighborhood-radius", type=int, default=1)
    p.add_argument("--fit-degree", type=int, default=2)
    p.add_argument("--future-steps", type=int, default=0,
                   help="Predicted next-image time length. 0 means reuse current time length.")
    p.add_argument("--draw-points", action="store_true")
    return p.parse_args()


def _validate_plane(path: str, expected_plane: str) -> tuple[str, str]:
    plane, time_axis = infer_plane_and_time_axis(path)
    if plane != expected_plane:
        raise ValueError(f"Expected {expected_plane} image, but {path} was inferred as {plane}.")
    return plane, time_axis


def _fit_single_plane(image_path: str, time_axis: str, args):
    bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    result = extract_fit_predict_trace(
        bgr,
        time_axis=time_axis,
        fit_degree=args.fit_degree,
        future_steps=args.future_steps,
        signal_source=args.signal_source,
        blur_ksize=args.blur_ksize,
        blur_sigma=args.blur_sigma,
        threshold_mode=args.threshold_mode,
        threshold_value=args.threshold_value,
        invert=args.invert,
        morph_open=args.morph_open,
        morph_close=args.morph_close,
        min_component_area=args.min_component_area,
        coord_method=args.coord_method,
        green_neighborhood_radius=args.green_neighborhood_radius,
    )
    result["bgr"] = bgr
    result["time_axis"] = time_axis
    return result


def _render_plane_overlay(bgr, trace, *, color, draw_points_flag):
    overlay = bgr.copy()
    if draw_points_flag:
        draw_points(overlay, trace["times"], trace["coords"], time_axis=trace["time_axis"], color=(0, 255, 255), radius=1)
    draw_polyline(overlay, trace["times"], trace["fit_coords"], time_axis=trace["time_axis"], color=color, thickness=2)
    return overlay


def _render_plane_future(trace, *, color):
    canvas = make_canvas(trace["future_time_len"], trace["spatial_size"], time_axis=trace["time_axis"], bg=0)
    draw_polyline(canvas, trace["future_times"], trace["future_coords"], time_axis=trace["time_axis"], color=color, thickness=2)
    return canvas


def main():
    args = parse_args()
    ensure_dir_for_prefix(args.out_prefix)

    _, xt_time_axis = _validate_plane(args.xt_input, "XT")
    _, yt_time_axis = _validate_plane(args.yt_input, "YT")

    xt_trace = _fit_single_plane(args.xt_input, xt_time_axis, args)
    yt_trace = _fit_single_plane(args.yt_input, yt_time_axis, args)

    sensor_width = int(args.sensor_width) if args.sensor_width is not None else int(xt_trace["spatial_size"])
    sensor_height = int(args.sensor_height) if args.sensor_height is not None else int(yt_trace["spatial_size"])

    observed_time_len = min(int(xt_trace["input_time_len"]), int(yt_trace["input_time_len"]))
    future_time_len = min(int(xt_trace["future_time_len"]), int(yt_trace["future_time_len"]))
    observed_times = np.arange(observed_time_len, dtype=np.float32)

    x_obs = clip_curve_to_spatial(poly_eval(xt_trace["coeffs"], observed_times), sensor_width)
    y_obs = clip_curve_to_spatial(poly_eval(yt_trace["coeffs"], observed_times), sensor_height)
    x_future = clip_curve_to_spatial(xt_trace["future_coords"][:future_time_len], sensor_width)
    y_future = clip_curve_to_spatial(yt_trace["future_coords"][:future_time_len], sensor_height)

    xt_overlay = _render_plane_overlay(xt_trace["bgr"], xt_trace, color=(0, 0, 255), draw_points_flag=args.draw_points)
    yt_overlay = _render_plane_overlay(yt_trace["bgr"], yt_trace, color=(255, 0, 0), draw_points_flag=args.draw_points)
    xt_future = _render_plane_future(xt_trace, color=(0, 0, 255))
    yt_future = _render_plane_future(yt_trace, color=(255, 0, 0))

    xy_observed = np.zeros((sensor_height, sensor_width, 3), dtype=np.uint8)
    xy_future = np.zeros((sensor_height, sensor_width, 3), dtype=np.uint8)
    if args.draw_points:
        draw_xy_points(xy_observed, x_obs, y_obs, color=(0, 255, 255), radius=1)
    draw_xy_polyline(xy_observed, x_obs, y_obs, color=(255, 255, 0), thickness=2)
    draw_xy_polyline(xy_future, x_future, y_future, color=(0, 255, 255), thickness=2)

    cv2.putText(xt_overlay, "XT fit", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(yt_overlay, "YT fit", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(xt_future, "XT future", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(yt_future, "YT future", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(xy_observed, "XY observed", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(xy_future, "XY future", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

    outputs = {
        "xt_overlay": xt_overlay,
        "yt_overlay": yt_overlay,
        "xt_future": xt_future,
        "yt_future": yt_future,
        "xy_observed": xy_observed,
        "xy_future": xy_future,
        "xt_mask": xt_trace["mask"],
        "yt_mask": yt_trace["mask"],
    }
    for key, img in outputs.items():
        cv2.imwrite(f"{args.out_prefix}_{key}.png", img)

    print("Done.")
    print(f"XT points: {len(xt_trace['times'])}")
    print(f"YT points: {len(yt_trace['times'])}")
    print(f"Observed XY length: {observed_time_len}")
    print(f"Future XY length: {future_time_len}")
    for key in outputs:
        print(f"Saved: {args.out_prefix}_{key}.png")


if __name__ == "__main__":
    main()
