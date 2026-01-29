import argparse
import re
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from ecfm.utils.evt3 import decode_evt3_raw, read_raw_header
from ecfm.data.tokenizer import Region, build_patch
from ecfm.utils.evt3_vis import draw_rectangles, events_to_image, write_image


_FRAME_RE = re.compile(r"_frame_(\d+)", re.IGNORECASE)
_RGB_TIME_RE = re.compile(r"_(\d{2})_(\d{2})_(\d{2})\.(\d+)$")


def _parse_geometry(meta: dict) -> Tuple[Optional[int], Optional[int]]:
    width = meta.get("width")
    height = meta.get("height")
    if width is not None and height is not None:
        try:
            return int(width), int(height)
        except ValueError:
            pass
    geometry = meta.get("geometry")
    if geometry and "x" in geometry:
        parts = geometry.lower().split("x", 1)
        try:
            return int(parts[0]), int(parts[1])
        except ValueError:
            return None, None
    return None, None


def _parse_label_time(path: Path) -> Optional[int]:
    match = _FRAME_RE.search(path.stem)
    if not match:
        return None
    return int(match.group(1))


def _parse_rgb_time(path: Path) -> Optional[float]:
    match = _RGB_TIME_RE.search(path.stem)
    if not match:
        return None
    hh, mm, ss, frac = match.groups()
    try:
        h = int(hh)
        m = int(mm)
        s = int(ss)
        micros = int(frac.ljust(6, "0")[:6])
    except ValueError:
        return None
    return h * 3600.0 + m * 60.0 + s + micros / 1_000_000.0


def _build_rgb_index(rgb_dir: Path, *, label_unit: float) -> list[tuple[float, Path]]:
    if not rgb_dir.exists():
        return []
    files = sorted(rgb_dir.glob("*.jpg")) + sorted(rgb_dir.glob("*.png"))
    if not files:
        return []
    times: list[tuple[float, Path]] = []
    parsed = [_parse_rgb_time(p) for p in files]
    if any(t is not None for t in parsed):
        base = next(t for t in parsed if t is not None)
        for path, t in zip(files, parsed):
            if t is None:
                continue
            rel_us = (t - base) * 1_000_000.0
            times.append((rel_us * float(label_unit), path))
    else:
        for idx, path in enumerate(files):
            times.append((float(idx), path))
    times.sort(key=lambda item: item[0])
    return times


def _find_rgb_frame(rgb_index: list[tuple[float, Path]], label_time: float) -> Optional[Path]:
    if not rgb_index:
        return None
    times = [t for t, _ in rgb_index]
    idx = int(np.searchsorted(times, label_time, side="left"))
    if idx <= 0:
        return rgb_index[0][1]
    if idx >= len(rgb_index):
        return rgb_index[-1][1]
    before_t, before_p = rgb_index[idx - 1]
    after_t, after_p = rgb_index[idx]
    if abs(label_time - before_t) <= abs(after_t - label_time):
        return before_p
    return after_p


def _read_rgb_image(path: Path) -> np.ndarray:
    try:
        from PIL import Image

        img = Image.open(path).convert("RGB")
        return np.array(img, dtype=np.uint8)
    except Exception as exc:
        raise RuntimeError(f"Failed to read image: {path}") from exc


def _to_grayscale(img: np.ndarray) -> np.ndarray:
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("Expected RGB image for grayscale conversion.")
    gray = (
        0.299 * img[:, :, 0].astype(np.float32)
        + 0.587 * img[:, :, 1].astype(np.float32)
        + 0.114 * img[:, :, 2].astype(np.float32)
    )
    gray = np.clip(gray, 0.0, 255.0).astype(np.uint8)
    out = np.zeros_like(img)
    out[:, :, 0] = gray
    out[:, :, 1] = gray
    out[:, :, 2] = gray
    return out


def _crop_to_boxes(
    img: np.ndarray,
    boxes: List[Tuple[int, int, int, int]],
) -> Tuple[np.ndarray, List[Tuple[int, int, int, int]]]:
    if not boxes:
        return img, boxes
    xs = [b[0] for b in boxes] + [b[2] for b in boxes]
    ys = [b[1] for b in boxes] + [b[3] for b in boxes]
    x0 = max(0, min(xs))
    y0 = max(0, min(ys))
    x1 = min(img.shape[1], max(xs))
    y1 = min(img.shape[0], max(ys))
    if x1 <= x0 or y1 <= y0:
        return img, boxes
    cropped = img[y0:y1, x0:x1]
    shifted = [(bx0 - x0, by0 - y0, bx1 - x0, by1 - y0) for bx0, by0, bx1, by1 in boxes]
    return cropped, shifted


def _read_ts_shift_us(raw_path: Path) -> Optional[int]:
    tmp_index = raw_path.with_suffix(raw_path.suffix + ".tmp_index")
    if not tmp_index.exists():
        return None
    _, _, meta = read_raw_header(tmp_index)
    shift = meta.get("ts_shift_us")
    if shift is None:
        return None
    try:
        return int(shift)
    except ValueError:
        return None


def _read_yolo_boxes(
    path: Path,
) -> List[Tuple[float, float, float, float]]:
    boxes: List[Tuple[float, float, float, float]] = []
    if path.stat().st_size == 0:
        return boxes
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        _, cx, cy, bw, bh = parts[:5]
        try:
            cx = float(cx)
            cy = float(cy)
            bw = float(bw)
            bh = float(bh)
        except ValueError:
            continue
        boxes.append((cx, cy, bw, bh))
    return boxes


def _project_boxes(
    boxes: List[Tuple[float, float, float, float]],
    *,
    dst_w: int,
    dst_h: int,
) -> List[Tuple[int, int, int, int]]:
    if not boxes:
        return []
    projected: List[Tuple[int, int, int, int]] = []
    for cx, cy, bw, bh in boxes:
        x0 = (cx - bw / 2.0) * dst_w
        y0 = (cy - bh / 2.0) * dst_h
        x1 = (cx + bw / 2.0) * dst_w
        y1 = (cy + bh / 2.0) * dst_h
        projected.append((int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))))
    return projected


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


def _scale_magnitude(mag: np.ndarray, mode: str, eps: float) -> np.ndarray:
    if mode == "linear":
        out = mag
    elif mode == "log":
        out = np.log1p(mag)
    elif mode == "db":
        out = 20.0 * np.log10(mag + eps)
        out = out - out.max() if out.size else out
        out = np.maximum(out, out.min()) if out.size else out
        out = out - out.min() if out.size else out
    else:
        raise ValueError(f"Unknown scale mode: {mode}")
    maxv = float(out.max()) if out.size else 0.0
    if maxv > 0:
        out = out / maxv
    return out


def _spectrum2d(img: np.ndarray, *, scale_mode: str, eps: float) -> np.ndarray:
    if img.size == 0:
        return img
    img_f = img.astype(np.float32) / 255.0
    out = np.zeros_like(img_f)
    for ch in range(3):
        comp = np.fft.fftshift(np.fft.fft2(img_f[:, :, ch]))
        mag = np.abs(comp)
        out[:, :, ch] = _scale_magnitude(mag, scale_mode, eps)
    out = np.clip(out * 255.0, 0.0, 255.0).astype(np.uint8)
    return out


def _spectrum2d_bin(img: np.ndarray, *, scale_mode: str, eps: float) -> np.ndarray:
    if img.size == 0:
        return img
    img_f = img.astype(np.float32) / 255.0
    signed = img_f[:, :, 0] - img_f[:, :, 2]
    comp = np.fft.fftshift(np.fft.fft2(signed))
    mag = np.abs(comp)
    mag = _scale_magnitude(mag, scale_mode, eps)
    gray = np.clip(mag * 255.0, 0.0, 255.0).astype(np.uint8)
    out = np.zeros_like(img)
    out[:, :, 0] = gray
    out[:, :, 1] = gray
    out[:, :, 2] = gray
    return out


def _stft_spectrogram(
    signal: np.ndarray, n_fft: int = 64, hop: int = 16, *, scale_mode: str, eps: float
) -> np.ndarray:
    if signal.size == 0:
        return np.zeros((0, 0), dtype=np.float32)
    sig = signal.astype(np.float32)
    if sig.size < n_fft:
        pad = n_fft - sig.size
        sig = np.pad(sig, (0, pad), mode="constant")
    window = np.hanning(n_fft).astype(np.float32)
    frames = 1 + max(0, (sig.size - n_fft) // hop)
    spec = np.zeros((n_fft // 2 + 1, frames), dtype=np.float32)
    for i in range(frames):
        start = i * hop
        frame = sig[start : start + n_fft]
        if frame.size < n_fft:
            frame = np.pad(frame, (0, n_fft - frame.size), mode="constant")
        frame = frame * window
        fft = np.fft.rfft(frame)
        mag = np.abs(fft)
        spec[:, i] = mag
    spec = _scale_magnitude(spec, scale_mode, eps)
    return spec


def _spectrogram_from_image(
    img: np.ndarray,
    *,
    time_horizontal: bool,
    n_fft: int = 64,
    hop: int = 16,
    scale_mode: str,
    eps: float,
) -> np.ndarray:
    if img.size == 0:
        return img
    img_f = img.astype(np.float32) / 255.0
    # time axis: x if horizontal else y
    if time_horizontal:
        sig_r = img_f[:, :, 0].sum(axis=0)
        sig_b = img_f[:, :, 2].sum(axis=0)
    else:
        sig_r = img_f[:, :, 0].sum(axis=1)
        sig_b = img_f[:, :, 2].sum(axis=1)
    spec_r = _stft_spectrogram(sig_r, n_fft=n_fft, hop=hop, scale_mode=scale_mode, eps=eps)
    spec_b = _stft_spectrogram(sig_b, n_fft=n_fft, hop=hop, scale_mode=scale_mode, eps=eps)
    h = spec_r.shape[0]
    w = spec_r.shape[1]
    out = np.zeros((h, w, 3), dtype=np.uint8)
    out[:, :, 0] = (spec_r * 255.0).astype(np.uint8)
    out[:, :, 2] = (spec_b * 255.0).astype(np.uint8)
    return out


def _spectrogram_from_image_bin(
    img: np.ndarray,
    *,
    time_horizontal: bool,
    n_fft: int = 64,
    hop: int = 16,
    scale_mode: str,
    eps: float,
) -> np.ndarray:
    if img.size == 0:
        return img
    img_f = img.astype(np.float32) / 255.0
    if time_horizontal:
        sig = (img_f[:, :, 0] - img_f[:, :, 2]).sum(axis=0)
    else:
        sig = (img_f[:, :, 0] - img_f[:, :, 2]).sum(axis=1)
    spec = _stft_spectrogram(sig, n_fft=n_fft, hop=hop, scale_mode=scale_mode, eps=eps)
    out = np.zeros((spec.shape[0], spec.shape[1], 3), dtype=np.uint8)
    gray = (spec * 255.0).astype(np.uint8)
    out[:, :, 0] = gray
    out[:, :, 1] = gray
    out[:, :, 2] = gray
    return out


def _apply_transform(
    img: np.ndarray,
    *,
    rep: str,
    transform: str,
    time_horizontal: bool,
    scale_mode: str,
    eps: float,
) -> np.ndarray:
    transform = transform.lower()
    if transform == "none":
        return img
    if transform == "spectrogram":
        if rep in {"events", "xy", "cstr2", "cstr3", "rgb", "grayscale", "gray"}:
            print(f"Warning: spectrogram ignored for spatial-only rep '{rep}'.")
            return img
        return _spectrogram_from_image(
            img, time_horizontal=time_horizontal, scale_mode=scale_mode, eps=eps
        )
    if transform == "spectrum2d":
        return _spectrum2d(img, scale_mode=scale_mode, eps=eps)
    if transform == "spectrogram_bin":
        if rep in {"events", "xy", "cstr2", "cstr3", "rgb", "grayscale", "gray"}:
            print(f"Warning: spectrogram ignored for spatial-only rep '{rep}'.")
            return img
        return _spectrogram_from_image_bin(
            img, time_horizontal=time_horizontal, scale_mode=scale_mode, eps=eps
        )
    if transform == "spectrum2d_bin":
        return _spectrum2d_bin(img, scale_mode=scale_mode, eps=eps)
    raise ValueError(f"Unknown transform: {transform}")


def _cstr_patch(
    events: np.ndarray,
    region: Region,
    *,
    patch_size: int,
    include_count: bool,
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
        maxv = float(cnt.max()) if cnt.size else 0.0
        if maxv > 0:
            img[:, :, 1] = cnt / maxv

    img = np.clip(img * 255.0, 0.0, 255.0).astype(np.uint8)
    return _resize_rgb(img, patch_size)


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
) -> np.ndarray:
    out_h = patch_size * grid_y
    out_w = patch_size * grid_x
    canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    x_edges = np.linspace(0, width, grid_x + 1, dtype=np.int64)
    y_edges = np.linspace(0, height, grid_y + 1, dtype=np.int64)

    for gy in range(grid_y):
        for gx in range(grid_x):
            x0 = int(x_edges[gx])
            x1 = int(x_edges[gx + 1])
            y0 = int(y_edges[gy])
            y1 = int(y_edges[gy + 1])
            dx = max(1, x1 - x0)
            dy = max(1, y1 - y0)
            region = Region(
                x=x0,
                y=y0,
                t=t0,
                dx=dx,
                dy=dy,
                dt=dt,
                plane=plane,
            )
            if plane in {"cstr2", "cstr3"}:
                patch_img = _cstr_patch(
                    events,
                    region,
                    patch_size=patch_size,
                    include_count=plane == "cstr3",
                )
            else:
                patch_t, _ = build_patch(
                    events,
                    region,
                    patch_size=patch_size,
                    time_bins=time_bins,
                )
                patch = patch_t.detach().cpu().numpy()
                time_horizontal = plane.startswith("yt")
                patch_img = _patch_to_rgb(patch, time_horizontal=time_horizontal)
            y1 = (gy + 1) * patch_size
            x1 = (gx + 1) * patch_size
            canvas[gy * patch_size : y1, gx * patch_size : x1] = patch_img
    return canvas


def render_yolo_frames(args: argparse.Namespace) -> None:
    events, counters, meta, _ = decode_evt3_raw(args.raw, endian=args.endian)
    width, height = _parse_geometry(meta)
    if args.width is not None:
        width = args.width
    if args.height is not None:
        height = args.height
    if width is None or height is None:
        raise ValueError("Could not determine geometry; pass --width/--height.")

    t = events[:, 2].astype(np.float64) * float(args.event_unit)
    ts_shift_us = args.ts_shift_us
    if ts_shift_us is None:
        ts_shift_us = _read_ts_shift_us(args.raw)
    if ts_shift_us is not None:
        shift = float(ts_shift_us) * float(args.event_unit)
        t = t - shift
        keep = t >= 0
        if np.any(keep):
            events = events[keep]
            t = t[keep]
        else:
            events = events[:0]
            t = t[:0]
        print(f"Applied ts_shift_us={ts_shift_us} (dropped events before shift)")
    if np.any(np.diff(t) < 0):
        order = np.argsort(t, kind="stable")
        events = events[order]
        t = t[order]

    label_files = list(args.yolo_dir.glob("*.txt"))
    label_files.sort(key=lambda p: (_parse_label_time(p) is None, _parse_label_time(p) or 0, p.name))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rgb_dir = args.rgb_dir
    if rgb_dir is None:
        candidate = args.yolo_dir.parent / "RGB"
        if candidate.exists():
            rgb_dir = candidate
        else:
            candidate = args.yolo_dir.parent / "PADDED_RGB"
            if candidate.exists():
                rgb_dir = candidate
    rgb_index = _build_rgb_index(rgb_dir, label_unit=args.label_unit) if rgb_dir else []

    for label_path in label_files:
        label_time_raw = _parse_label_time(label_path)
        if label_time_raw is None:
            continue
        label_time = float(label_time_raw) * float(args.label_unit)
        window = float(args.window) * float(args.label_unit)
        if args.center:
            t0 = label_time - window / 2.0
            t1 = label_time + window / 2.0
        else:
            t0 = label_time
            t1 = label_time + window

        idx0 = int(np.searchsorted(t, t0, side="left"))
        idx1 = int(np.searchsorted(t, t1, side="left"))
        ev = events[idx0:idx1]
        ev_time = ev
        if ev.size:
            ev_time = ev.copy()
            ev_time[:, 2] = t[idx0:idx1]

        boxes = _read_yolo_boxes(label_path)
        if args.only_with_rects and not boxes:
            continue
        reps_raw = args.representation.replace(",", ";")
        representations = [r.strip() for r in reps_raw.split(";") if r.strip()]
        crop_raw = args.crop_representations.replace(",", ";")
        crop_reps = {r.strip() for r in crop_raw.split(";") if r.strip()}
        valid = {
            "events",
            "xy",
            "xt",
            "yt",
            "xy_p45",
            "xy_m45",
            "yt_p45",
            "yt_m45",
            "cstr2",
            "cstr3",
            "rgb",
            "grayscale",
            "gray",
        }
        for rep in representations:
            if rep not in valid:
                raise ValueError(f"Unknown representation: {rep}")
            if rep in {"rgb", "grayscale", "gray"}:
                rgb_path = _find_rgb_frame(rgb_index, label_time)
                if rgb_path is None:
                    print(f"Warning: no RGB frame found for {label_path.name}")
                    continue
                img = _read_rgb_image(rgb_path)
                if rep in {"grayscale", "gray"}:
                    img = _to_grayscale(img)
            elif rep == "events":
                if args.grid_x == 1 and args.grid_y == 1:
                    img = events_to_image(
                        ev,
                        width,
                        height,
                        pixel_size=args.pixel_size,
                    )
                else:
                    img = _render_histogram_grid(
                        ev_time,
                        width=width,
                        height=height,
                        t0=t0,
                        dt=t1 - t0,
                        plane="xy",
                        time_bins=1,
                        patch_size=args.spatial_bins,
                        grid_x=args.grid_x,
                        grid_y=args.grid_y,
                    )
            else:
                img = _render_histogram_grid(
                    ev_time,
                    width=width,
                    height=height,
                    t0=t0,
                    dt=t1 - t0,
                    plane=rep,
                    time_bins=args.temporal_bins,
                    patch_size=args.spatial_bins,
                    grid_x=args.grid_x,
                    grid_y=args.grid_y,
                )
            time_horizontal = rep.startswith("yt")
            img = _apply_transform(
                img,
                rep=rep,
                transform=args.transform,
                time_horizontal=time_horizontal,
                scale_mode=args.transform_scale,
                eps=args.transform_eps,
            )
            if args.output_size is not None:
                img = _resize_to(img, (args.output_size[0], args.output_size[1]))
            scaled_boxes = _project_boxes(
                boxes,
                dst_w=img.shape[1],
                dst_h=img.shape[0],
            )
            if rep in crop_reps:
                img, scaled_boxes = _crop_to_boxes(img, scaled_boxes)
            if args.draw_rectangles and scaled_boxes:
                draw_rectangles(
                    img,
                    scaled_boxes,
                    color=tuple(args.rect_color),
                    thickness=args.rect_thickness,
                )
            out_path = args.output_dir / f"{label_path.stem}_{rep}.png"
            final_path = write_image(out_path, img)
            print(f"Wrote {final_path}")

    if any(counters.values()):
        print("Ignored word types:")
        for k, v in counters.items():
            if v:
                print(f"  {k}: {v}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Decode EVT3 .raw, render images only for non-empty YOLO label files, "
            "and draw target rectangles."
        )
    )
    parser.add_argument("raw", type=Path, help="Path to .raw file")
    parser.add_argument("yolo_dir", type=Path, help="Directory with YOLO txt files")
    parser.add_argument("output_dir", type=Path, help="Output directory for images")
    parser.add_argument(
        "--rgb-dir",
        type=Path,
        default=None,
        help="Optional RGB image directory (defaults to sibling RGB or PADDED_RGB)",
    )
    parser.add_argument(
        "--window",
        type=float,
        default=33333.0,
        help="Time window length in label units (default: 33333.0)",
    )
    parser.add_argument(
        "--label-unit",
        type=float,
        default=1.0,
        help="Label timestamp unit scale to seconds (default: 1.0 = microseconds)",
    )
    parser.add_argument(
        "--event-unit",
        type=float,
        default=1.0,
        help="Event timestamp unit scale to seconds (default: 1.0 = microseconds)",
    )
    parser.add_argument(
        "--ts-shift-us",
        type=float,
        default=None,
        help=(
            "Timestamp shift (microseconds) to add to event times. "
            "Defaults to ts_shift_us from .raw.tmp_index if available."
        ),
    )
    parser.add_argument(
        "--center",
        action="store_true",
        help="Center the time window on the label timestamp (default: start at label)",
    )
    parser.add_argument(
        "--endian",
        choices=["little", "big"],
        default="little",
        help="Byte order of EVT3 words (default: little)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Frame width override (defaults to header geometry)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Frame height override (defaults to header geometry)",
    )
    parser.add_argument(
        "--pixel-size",
        type=int,
        default=1,
        help="Event pixel size in output image (default: 1)",
    )
    parser.add_argument(
        "--representation",
        type=str,
        default="events",
        help=(
            "Representation(s) to render, separated by ';' (default: events). "
            "Options: events, xy, xt, yt, xy_p45, xy_m45, yt_p45, yt_m45, cstr2, cstr3, "
            "rgb, grayscale."
        ),
    )
    parser.add_argument(
        "--crop-representations",
        type=str,
        default="",
        help="Representations to crop to YOLO boxes, separated by ';' (default: none).",
    )
    parser.add_argument(
        "--temporal-bins",
        type=int,
        default=64,
        help="Temporal bins for histogram representations (default: 64)",
    )
    parser.add_argument(
        "--transform",
        type=str,
        default="none",
        choices=["none", "spectrogram", "spectrum2d", "spectrogram_bin", "spectrum2d_bin"],
        help=(
            "Optional transform on the rendered patch: none, spectrogram, spectrum2d "
            "(default: none)."
        ),
    )
    parser.add_argument(
        "--transform-scale",
        type=str,
        default="linear",
        choices=["linear", "log", "db"],
        help="Magnitude scaling for spectral transforms (default: linear).",
    )
    parser.add_argument(
        "--transform-eps",
        type=float,
        default=1e-6,
        help="Epsilon for log/db scaling (default: 1e-6).",
    )
    parser.add_argument(
        "--spatial-bins",
        type=int,
        default=32,
        help="Output patch size for histogram representations (default: 32)",
    )
    parser.add_argument(
        "--output-size",
        type=int,
        nargs=2,
        default=None,
        help="Final output image size as W H (default: None)",
    )
    parser.add_argument(
        "--grid-x",
        type=int,
        default=1,
        help="Number of grid cells in x (default: 1)",
    )
    parser.add_argument(
        "--grid-y",
        type=int,
        default=1,
        help="Number of grid cells in y (default: 1)",
    )
    parser.add_argument(
        "--rect-color",
        type=int,
        nargs=3,
        default=(0, 255, 0),
        help="Rectangle color as R G B (default: 0 255 0)",
    )
    parser.add_argument(
        "--draw-rectangles",
        action="store_true",
        default=False,
        help="Draw rectangles from YOLO labels (default: False)",
    )
    parser.add_argument(
        "--no-rectangles",
        action="store_false",
        dest="draw_rectangles",
        help="Disable drawing rectangles from YOLO labels",
    )
    parser.add_argument(
        "--rect-thickness",
        type=int,
        default=1,
        help="Rectangle thickness in pixels (default: 1)",
    )
    parser.add_argument(
        "--only-with-rects",
        action="store_true",
        default=True,
        help="Only write images that have at least one rectangle (default: True)",
    )
    parser.add_argument(
        "--include-empty",
        action="store_false",
        dest="only_with_rects",
        help="Also write images with zero rectangles",
    )
    args = parser.parse_args()
    render_yolo_frames(args)


if __name__ == "__main__":
    main()
