#!/usr/bin/env python3
"""
fit_event_trace.py

Fit a curve to an XT or YT event-histogram trace, predict a future curve that
preserves the fitted curve's end acceleration, and optionally estimate the
other view (XT from YT, or YT from XT) using the green channel as the missing
normalized coordinate.

Dependencies:
    pip install opencv-python numpy

Example:
    python fit_event_trace.py \
        --input my_yt_mx.png \
        --out-prefix out/run1

Typical assumptions:
    - The visible trace is reasonably single-valued w.r.t. time.
    - The green channel stores the mean normalized missing coordinate:
        * XT image -> green encodes normalized Y
        * YT image -> green encodes normalized X
    - Plane and time-axis are inferred from the filename whenever possible:
        * `xt*` -> plane XT, time along y
        * `yt*` -> plane YT, time along x

Outputs:
    <prefix>_overlay.png
    <prefix>_future.png
    <prefix>_cross_estimated.png
    <prefix>_cross_future.png
    <prefix>_mask.png
"""

import os
from pathlib import Path
import cv2
import math
import argparse
import numpy as np


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def odd_ksize(k: int) -> int:
    k = max(1, int(k))
    return k if k % 2 == 1 else k + 1


def ensure_dir_for_prefix(prefix: str):
    d = os.path.dirname(prefix)
    if d:
        os.makedirs(d, exist_ok=True)


def infer_plane_and_time_axis(input_path: str):
    stem = Path(input_path).stem.lower()
    if stem.endswith("_xt_my") or stem.endswith("_xt") or "_xt_" in stem:
        return "XT", "y"
    if stem.endswith("_yt_mx") or stem.endswith("_yt") or "_yt_" in stem:
        return "YT", "x"
    raise ValueError(
        "Could not infer plane/time-axis from filename. "
        "Expected a name containing xt/yt such as *_xt_my.png or *_yt_mx.png."
    )


def canonical_time_axis_for_plane(plane: str) -> str:
    plane = plane.upper()
    if plane == "XT":
        return "y"
    if plane == "YT":
        return "x"
    raise ValueError(f"Unknown plane: {plane}")


def poly_eval(coeffs, t):
    return np.poly1d(coeffs)(t)


def poly_derivative(coeffs, order=1):
    p = np.poly1d(coeffs)
    for _ in range(order):
        p = np.polyder(p)
    return p


def clip_points(points, width, height):
    out = []
    for x, y in points:
        xi = int(round(np.clip(x, 0, width - 1)))
        yi = int(round(np.clip(y, 0, height - 1)))
        out.append((xi, yi))
    return out


def draw_polyline(img, times, coords, time_axis='x', color=(0, 0, 255), thickness=2):
    h, w = img.shape[:2]
    pts = []
    for t, s in zip(times, coords):
        if not np.isfinite(t) or not np.isfinite(s):
            continue
        if time_axis == 'x':
            x = t
            y = s
        else:
            x = s
            y = t
        pts.append((x, y))
    pts = clip_points(pts, w, h)
    if len(pts) >= 2:
        cv2.polylines(img, [np.array(pts, dtype=np.int32)], False, color, thickness, cv2.LINE_AA)


def draw_points(img, times, coords, time_axis='x', color=(0, 255, 255), radius=1):
    h, w = img.shape[:2]
    for t, s in zip(times, coords):
        if not np.isfinite(t) or not np.isfinite(s):
            continue
        if time_axis == 'x':
            x = t
            y = s
        else:
            x = s
            y = t
        xi = int(round(np.clip(x, 0, w - 1)))
        yi = int(round(np.clip(y, 0, h - 1)))
        cv2.circle(img, (xi, yi), radius, color, -1, cv2.LINE_AA)


def normalize_channel_to_pixel(green_vals, spatial_size):
    """
    Convert green [0..255] -> normalized [0..1] -> pixel coordinate [0..spatial_size-1].
    """
    g = np.asarray(green_vals, dtype=np.float32)
    norm = g / 255.0
    return norm * (spatial_size - 1)


def filter_small_components(binary_mask, min_area):
    if min_area <= 1:
        return binary_mask.copy()

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    out = np.zeros_like(binary_mask)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            out[labels == i] = 255
    return out


def make_canvas(time_len, spatial_size, time_axis='x', bg=0):
    """
    Returns a 3-channel canvas for drawing a synthetic plane.
    If time_axis == 'x': width=time_len, height=spatial_size
    If time_axis == 'y': width=spatial_size, height=time_len
    """
    if time_axis == 'x':
        canvas = np.full((spatial_size, time_len, 3), bg, dtype=np.uint8)
    else:
        canvas = np.full((time_len, spatial_size, 3), bg, dtype=np.uint8)
    return canvas


# -----------------------------------------------------------------------------
# Signal extraction
# -----------------------------------------------------------------------------

def build_signal_image(bgr, source='max'):
    """
    Build a scalar image used for trace detection.

    Options:
        gray  -> cv2 grayscale
        max   -> max(B,G,R)
        rbmax -> max(R,B)  (ignores green for detection, useful if green is metadata-heavy)
        green -> green only
        red   -> red only
        blue  -> blue only
    """
    source = source.lower()
    b, g, r = cv2.split(bgr)

    if source == 'gray':
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if source == 'max':
        return np.maximum(np.maximum(b, g), r)
    if source == 'rbmax':
        return np.maximum(r, b)
    if source == 'green':
        return g
    if source == 'red':
        return r
    if source == 'blue':
        return b

    raise ValueError(f"Unknown signal source: {source}")


def preprocess_mask(signal,
                    blur_ksize=5,
                    blur_sigma=1.0,
                    threshold_mode='otsu',
                    threshold_value=40,
                    invert=False,
                    morph_open=3,
                    morph_close=3,
                    min_component_area=20):
    """
    Blur + threshold + morphology + connected-component filtering.
    """
    blur_ksize = odd_ksize(blur_ksize)
    blurred = cv2.GaussianBlur(signal, (blur_ksize, blur_ksize), blur_sigma)

    thresh_flag = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    threshold_mode = threshold_mode.lower()

    if threshold_mode == 'otsu':
        _, mask = cv2.threshold(blurred, 0, 255, thresh_flag + cv2.THRESH_OTSU)
    elif threshold_mode == 'fixed':
        _, mask = cv2.threshold(blurred, threshold_value, 255, thresh_flag)
    elif threshold_mode == 'adaptive':
        block_size = max(3, odd_ksize(11))
        mask = cv2.adaptiveThreshold(
            blurred, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY,
            block_size, 2
        )
    else:
        raise ValueError(f"Unknown threshold mode: {threshold_mode}")

    if morph_open > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (odd_ksize(morph_open), odd_ksize(morph_open)))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)

    if morph_close > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (odd_ksize(morph_close), odd_ksize(morph_close)))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

    mask = filter_small_components(mask, min_component_area)
    return blurred, mask


def extract_trace_and_green(mask,
                            signal,
                            green_channel,
                            time_axis='x',
                            coord_method='weighted_centroid',
                            neighborhood_radius=1):
    """
    Extract a 1D trace coordinate s(t) from the binary mask, plus the green-channel
    statistic along that trace.

    Returns:
        times, coords, green_vals
    """
    h, w = mask.shape[:2]
    times = []
    coords = []
    green_vals = []

    coord_method = coord_method.lower()

    if time_axis == 'x':
        T = w
        spatial_len = h
        for t in range(T):
            idx = np.where(mask[:, t] > 0)[0]
            if len(idx) == 0:
                continue

            weights = signal[:, t].astype(np.float32)
            weights = weights * (mask[:, t] > 0).astype(np.float32)

            if coord_method == 'weighted_centroid':
                s = np.sum(np.arange(spatial_len, dtype=np.float32) * weights) / max(np.sum(weights), 1e-6)
            elif coord_method == 'centroid':
                s = float(np.mean(idx))
            elif coord_method == 'median':
                s = float(np.median(idx))
            elif coord_method == 'argmax':
                s = float(np.argmax(weights))
            else:
                raise ValueError(f"Unknown coord method: {coord_method}")

            s_int = int(round(np.clip(s, 0, h - 1)))
            y0 = max(0, s_int - neighborhood_radius)
            y1 = min(h, s_int + neighborhood_radius + 1)

            # Sample green near the extracted center, weighted by local signal
            local_g = green_channel[y0:y1, t].astype(np.float32)
            local_w = signal[y0:y1, t].astype(np.float32)
            if np.sum(local_w) > 1e-6:
                gval = float(np.sum(local_g * local_w) / np.sum(local_w))
            else:
                gval = float(np.mean(local_g))

            times.append(float(t))
            coords.append(float(s))
            green_vals.append(gval)

    else:
        T = h
        spatial_len = w
        for t in range(T):
            idx = np.where(mask[t, :] > 0)[0]
            if len(idx) == 0:
                continue

            weights = signal[t, :].astype(np.float32)
            weights = weights * (mask[t, :] > 0).astype(np.float32)

            if coord_method == 'weighted_centroid':
                s = np.sum(np.arange(spatial_len, dtype=np.float32) * weights) / max(np.sum(weights), 1e-6)
            elif coord_method == 'centroid':
                s = float(np.mean(idx))
            elif coord_method == 'median':
                s = float(np.median(idx))
            elif coord_method == 'argmax':
                s = float(np.argmax(weights))
            else:
                raise ValueError(f"Unknown coord method: {coord_method}")

            s_int = int(round(np.clip(s, 0, w - 1)))
            x0 = max(0, s_int - neighborhood_radius)
            x1 = min(w, s_int + neighborhood_radius + 1)

            local_g = green_channel[t, x0:x1].astype(np.float32)
            local_w = signal[t, x0:x1].astype(np.float32)
            if np.sum(local_w) > 1e-6:
                gval = float(np.sum(local_g * local_w) / np.sum(local_w))
            else:
                gval = float(np.mean(local_g))

            times.append(float(t))
            coords.append(float(s))
            green_vals.append(gval)

    return np.asarray(times, dtype=np.float32), np.asarray(coords, dtype=np.float32), np.asarray(green_vals, dtype=np.float32)


# -----------------------------------------------------------------------------
# Fitting / prediction
# -----------------------------------------------------------------------------

def fit_polynomial(times, coords, degree=2):
    if len(times) < degree + 1:
        raise ValueError(
            f"Not enough points ({len(times)}) for polynomial degree {degree}. "
            f"Need at least {degree + 1}."
        )
    coeffs = np.polyfit(times, coords, degree)
    return coeffs


def predict_future_constant_acceleration(times, coeffs, future_steps=60):
    """
    Predict future using:
        s_future(dt) = s0 + v0*dt + 0.5*a0*dt^2
    where s0,v0,a0 are taken at the end of the fitted curve.

    This "maintains the fitted curve's acceleration" at the end.
    """
    t_end = float(np.max(times))
    s_poly = np.poly1d(coeffs)
    d1 = poly_derivative(coeffs, 1)
    d2 = poly_derivative(coeffs, 2) if len(coeffs) >= 3 else np.poly1d([0.0])

    s0 = float(s_poly(t_end))
    v0 = float(d1(t_end))
    a0 = float(d2(t_end))

    dt = np.arange(1, future_steps + 1, dtype=np.float32)
    s_future = s0 + v0 * dt + 0.5 * a0 * dt * dt
    # Render the predicted state into its own future image, so the time axis
    # starts at zero for the next window instead of continuing inside the
    # current image.
    t_future_relative = np.arange(future_steps, dtype=np.float32)
    return t_future_relative, s_future, {'t_end': t_end, 's0': s0, 'v0': v0, 'a0': a0}


def clip_curve_to_spatial(coords, spatial_size):
    return np.clip(coords, 0, spatial_size - 1)


# -----------------------------------------------------------------------------
# Main logic
# -----------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Fit event-camera XT/YT trace and predict future curve.")
    p.add_argument('--input', required=True, help='Input image path')
    p.add_argument('--plane', default=None, choices=['XT', 'YT'],
                   help='Type of input plane. Default: inferred from filename')
    p.add_argument('--time-axis', default=None, choices=['x', 'y'],
                   help='Which axis is time in the image. Default: inferred from filename')

    # output
    p.add_argument('--out-prefix', default='output/result', help='Output file prefix')

    # dimensions for cross-plane generation
    p.add_argument('--sensor-width', type=int, default=None,
                   help='Sensor width in pixels for normalized X from green. Default: input width')
    p.add_argument('--sensor-height', type=int, default=None,
                   help='Sensor height in pixels for normalized Y from green. Default: input height')

    # preprocessing
    p.add_argument('--signal-source', default='rbmax',
                   choices=['gray', 'max', 'rbmax', 'green', 'red', 'blue'],
                   help='Image channel combination used for trace detection. Default: rbmax')
    p.add_argument('--blur-ksize', type=int, default=5, help='Gaussian blur kernel size. Default: 5')
    p.add_argument('--blur-sigma', type=float, default=1.2, help='Gaussian blur sigma. Default: 1.2')
    p.add_argument('--threshold-mode', default='otsu', choices=['otsu', 'fixed', 'adaptive'],
                   help='Threshold mode. Default: otsu')
    p.add_argument('--threshold-value', type=float, default=40.0,
                   help='Threshold value if threshold-mode=fixed. Default: 40')
    p.add_argument('--invert', action='store_true',
                   help='Invert threshold if trace is dark on bright background')
    p.add_argument('--morph-open', type=int, default=3, help='Morphological open kernel size. Default: 3')
    p.add_argument('--morph-close', type=int, default=3, help='Morphological close kernel size. Default: 3')
    p.add_argument('--min-component-area', type=int, default=20,
                   help='Remove connected components smaller than this. Default: 20')

    # coordinate extraction
    p.add_argument('--coord-method', default='weighted_centroid',
                   choices=['weighted_centroid', 'centroid', 'median', 'argmax'],
                   help='How to extract one trace point per time sample. Default: weighted_centroid')
    p.add_argument('--green-neighborhood-radius', type=int, default=1,
                   help='Radius around extracted trace used for green sampling. Default: 1')

    # fitting / future
    p.add_argument('--fit-degree', type=int, default=2,
                   help='Polynomial degree for fit. Use 2 for constant acceleration fit. Default: 2')
    p.add_argument('--future-steps', type=int, default=60,
                   help='Number of time samples in the predicted next image. '
                        'Use 0 to reuse the full input image time length. Default: 60')

    # visualization
    p.add_argument('--draw-points', action='store_true', help='Draw extracted points on overlay')

    return p.parse_args()


def main():
    args = parse_args()
    ensure_dir_for_prefix(args.out_prefix)
    inferred_plane, inferred_time_axis = infer_plane_and_time_axis(args.input)
    plane = args.plane if args.plane is not None else inferred_plane
    time_axis = args.time_axis if args.time_axis is not None else inferred_time_axis

    bgr = cv2.imread(args.input, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {args.input}")

    h, w = bgr.shape[:2]
    sensor_w = args.sensor_width if args.sensor_width is not None else w
    sensor_h = args.sensor_height if args.sensor_height is not None else h
    input_time_len = w if time_axis == 'x' else h
    future_time_len = input_time_len if args.future_steps <= 0 else int(args.future_steps)

    signal = build_signal_image(bgr, source=args.signal_source)
    green = bgr[:, :, 1]

    blurred, mask = preprocess_mask(
        signal=signal,
        blur_ksize=args.blur_ksize,
        blur_sigma=args.blur_sigma,
        threshold_mode=args.threshold_mode,
        threshold_value=args.threshold_value,
        invert=args.invert,
        morph_open=args.morph_open,
        morph_close=args.morph_close,
        min_component_area=args.min_component_area
    )

    times, coords, green_vals = extract_trace_and_green(
        mask=mask,
        signal=blurred,
        green_channel=green,
        time_axis=time_axis,
        coord_method=args.coord_method,
        neighborhood_radius=args.green_neighborhood_radius
    )

    if len(times) < max(args.fit_degree + 1, 5):
        raise RuntimeError(
            f"Too few extracted trace points ({len(times)}). "
            f"Try changing thresholding, signal source, morphology, or coord method."
        )

    # Sort by time just in case
    order = np.argsort(times)
    times = times[order]
    coords = coords[order]
    green_vals = green_vals[order]

    # Primary fit (visible plane)
    coeffs = fit_polynomial(times, coords, degree=args.fit_degree)
    fit_coords = poly_eval(coeffs, times)

    # Future prediction from end state
    future_times, future_coords, future_info = predict_future_constant_acceleration(
        times, coeffs, future_steps=future_time_len
    )

    # Clip primary curves
    primary_spatial_size = h if time_axis == 'x' else w
    fit_coords = clip_curve_to_spatial(fit_coords, primary_spatial_size)
    future_coords = clip_curve_to_spatial(future_coords, primary_spatial_size)

    # Overlay on original
    overlay = bgr.copy()
    if args.draw_points:
        draw_points(overlay, times, coords, time_axis=time_axis, color=(0, 255, 255), radius=1)
    draw_polyline(overlay, times, fit_coords, time_axis=time_axis, color=(0, 0, 255), thickness=2)

    # Future-only canvas for the input plane
    future_canvas = make_canvas(future_time_len, primary_spatial_size, time_axis=time_axis, bg=0)
    draw_polyline(future_canvas, future_times, future_coords, time_axis=time_axis, color=(0, 0, 255), thickness=2)

    # -------------------------------------------------------------------------
    # Cross-plane estimate from green channel
    # -------------------------------------------------------------------------
    #
    # If input is XT:
    #   visible coord = X(t)
    #   green encodes mean Y(t), normalized
    #   estimated other view = YT, using green->Y
    #
    # If input is YT:
    #   visible coord = Y(t)
    #   green encodes mean X(t), normalized
    #   estimated other view = XT, using green->X
    #
    # This is only approximate and assumes the green value sampled at the trace
    # really corresponds to the same object/trajectory point.
    #
    if plane == 'XT':
        cross_plane = 'YT'
        cross_spatial_size = sensor_h
    else:
        cross_plane = 'XT'
        cross_spatial_size = sensor_w
    cross_time_axis = canonical_time_axis_for_plane(cross_plane)
    cross_input_time_len = input_time_len if cross_time_axis == time_axis else (h if cross_time_axis == 'y' else w)
    cross_future_time_len = future_time_len

    cross_coords_obs = normalize_channel_to_pixel(green_vals, cross_spatial_size)

    # Fit the missing coordinate vs time
    cross_coeffs = fit_polynomial(times, cross_coords_obs, degree=args.fit_degree)
    cross_fit_coords = poly_eval(cross_coeffs, times)
    cross_future_times, cross_future_coords, cross_future_info = predict_future_constant_acceleration(
        times, cross_coeffs, future_steps=future_time_len
    )

    cross_fit_coords = clip_curve_to_spatial(cross_fit_coords, cross_spatial_size)
    cross_future_coords = clip_curve_to_spatial(cross_future_coords, cross_spatial_size)

    cross_canvas = make_canvas(cross_input_time_len, cross_spatial_size, time_axis=cross_time_axis, bg=0)
    cross_future_canvas = make_canvas(cross_future_time_len, cross_spatial_size, time_axis=cross_time_axis, bg=0)

    # Draw observed estimated cross-plane fit
    draw_polyline(cross_canvas, times, cross_fit_coords, time_axis=cross_time_axis, color=(0, 255, 0), thickness=2)

    draw_polyline(cross_future_canvas, cross_future_times, cross_future_coords,
                  time_axis=cross_time_axis, color=(0, 255, 0), thickness=2)

    # Add labels
    cv2.putText(overlay, f"{plane} fit", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(future_canvas, f"{plane} future", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(cross_canvas, f"Estimated {cross_plane} from green", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(cross_future_canvas, f"Estimated {cross_plane} future", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

    # Write output files
    overlay_path = f"{args.out_prefix}_overlay.png"
    future_path = f"{args.out_prefix}_future.png"
    cross_path = f"{args.out_prefix}_cross_estimated.png"
    cross_future_path = f"{args.out_prefix}_cross_future.png"
    mask_path = f"{args.out_prefix}_mask.png"

    cv2.imwrite(overlay_path, overlay)
    cv2.imwrite(future_path, future_canvas)
    cv2.imwrite(cross_path, cross_canvas)
    cv2.imwrite(cross_future_path, cross_future_canvas)
    cv2.imwrite(mask_path, mask)

    # Print a compact summary
    print("Done.")
    print(f"Input plane: {plane} (inferred: {inferred_plane})")
    print(f"Time axis: {time_axis} (inferred: {inferred_time_axis})")
    print(f"Extracted points: {len(times)}")
    print(f"Primary fit degree: {args.fit_degree}")
    print(f"Primary future state @ end: s0={future_info['s0']:.3f}, v0={future_info['v0']:.3f}, a0={future_info['a0']:.3f}")
    print(f"Cross-plane future state @ end: s0={cross_future_info['s0']:.3f}, v0={cross_future_info['v0']:.3f}, a0={cross_future_info['a0']:.3f}")
    print(f"Saved: {overlay_path}")
    print(f"Saved: {future_path}")
    print(f"Saved: {cross_path}")
    print(f"Saved: {cross_future_path}")
    print(f"Saved: {mask_path}")


if __name__ == "__main__":
    main()
