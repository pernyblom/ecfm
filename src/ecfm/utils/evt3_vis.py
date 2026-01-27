from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np


def events_to_image(
    events: np.ndarray,
    width: int,
    height: int,
    *,
    polarity_colors: Tuple[Tuple[int, int, int], Tuple[int, int, int]] = (
        (0, 0, 255),
        (255, 0, 0),
    ),
    background: Tuple[int, int, int] = (0, 0, 0),
    pixel_size: int = 1,
) -> np.ndarray:
    if pixel_size < 1:
        raise ValueError("pixel_size must be >= 1")
    img_h = height * pixel_size
    img_w = width * pixel_size
    img = np.zeros((img_h, img_w, 3), dtype=np.uint8)
    img[:, :] = np.asarray(background, dtype=np.uint8)

    if events.size == 0:
        return img

    xs = events[:, 0].astype(np.int64)
    ys = events[:, 1].astype(np.int64)
    ps = events[:, 3].astype(np.int64)
    ps = np.clip(ps, 0, 1)

    in_bounds = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
    if not np.any(in_bounds):
        return img

    xs = xs[in_bounds]
    ys = ys[in_bounds]
    ps = ps[in_bounds]

    colors = np.asarray(polarity_colors, dtype=np.uint8)

    if pixel_size == 1:
        img[ys, xs] = colors[ps]
        return img

    for x, y, p in zip(xs, ys, ps):
        x0 = x * pixel_size
        y0 = y * pixel_size
        img[y0 : y0 + pixel_size, x0 : x0 + pixel_size] = colors[p]

    return img


def events_to_time_slices(
    events: np.ndarray,
    dt: float,
    *,
    time_unit: float = 1.0,
    max_frames: Optional[int] = None,
    start_time: Optional[float] = None,
    sort_by_time: bool = True,
) -> Tuple[List[np.ndarray], List[Tuple[float, float]]]:
    if dt <= 0:
        raise ValueError("dt must be > 0")
    if events.size == 0:
        return [], []

    t = events[:, 2].astype(np.float64) * float(time_unit)
    if sort_by_time and np.any(np.diff(t) < 0):
        order = np.argsort(t, kind="stable")
        events = events[order]
        t = t[order]

    if start_time is None:
        start_time = float(t[0])

    frames: List[np.ndarray] = []
    ranges: List[Tuple[float, float]] = []

    idx0 = int(np.searchsorted(t, start_time, side="left"))
    frame = 0
    while idx0 < t.size:
        if max_frames is not None and frame >= max_frames:
            break
        end_time = start_time + dt
        idx1 = int(np.searchsorted(t, end_time, side="left"))
        frames.append(events[idx0:idx1])
        ranges.append((start_time, end_time))
        start_time = end_time
        idx0 = idx1
        frame += 1

    return frames, ranges


def write_image(path: Path, img: np.ndarray) -> Path:
    path = Path(path)
    try:
        import torch
        from torchvision.io import write_png

        tensor = torch.from_numpy(img).permute(2, 0, 1)
        write_png(tensor, str(path))
        return path
    except Exception:
        ppm_path = path.with_suffix(".ppm")
        _write_ppm(ppm_path, img)
        return ppm_path


def _write_ppm(path: Path, img: np.ndarray) -> None:
    path = Path(path)
    h, w, _ = img.shape
    header = f"P6 {w} {h} 255\n".encode("ascii")
    with path.open("wb") as f:
        f.write(header)
        f.write(img.tobytes(order="C"))


def render_event_frames(
    events: np.ndarray,
    width: int,
    height: int,
    dt: float,
    *,
    time_unit: float = 1.0,
    max_frames: Optional[int] = None,
    pixel_size: int = 1,
    polarity_colors: Tuple[Tuple[int, int, int], Tuple[int, int, int]] = (
        (0, 0, 255),
        (255, 0, 0),
    ),
    background: Tuple[int, int, int] = (0, 0, 0),
    sort_by_time: bool = True,
) -> Tuple[List[np.ndarray], List[Tuple[float, float]]]:
    slices, ranges = events_to_time_slices(
        events,
        dt,
        time_unit=time_unit,
        max_frames=max_frames,
        sort_by_time=sort_by_time,
    )
    images: List[np.ndarray] = []
    for ev in slices:
        img = events_to_image(
            ev,
            width,
            height,
            polarity_colors=polarity_colors,
            background=background,
            pixel_size=pixel_size,
        )
        images.append(img)
    return images, ranges


def draw_rectangles(
    img: np.ndarray,
    boxes: Iterable[Tuple[int, int, int, int]],
    *,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 1,
) -> np.ndarray:
    if thickness < 1:
        raise ValueError("thickness must be >= 1")
    h, w, _ = img.shape
    col = np.asarray(color, dtype=np.uint8)
    for x0, y0, x1, y1 in boxes:
        x0 = max(0, min(int(x0), w - 1))
        y0 = max(0, min(int(y0), h - 1))
        x1 = max(0, min(int(x1), w - 1))
        y1 = max(0, min(int(y1), h - 1))
        if x1 < x0 or y1 < y0:
            continue
        for t in range(thickness):
            xt0 = max(0, x0 - t)
            yt0 = max(0, y0 - t)
            xt1 = min(w - 1, x1 + t)
            yt1 = min(h - 1, y1 + t)
            img[yt0, xt0 : xt1 + 1] = col
            img[yt1, xt0 : xt1 + 1] = col
            img[yt0 : yt1 + 1, xt0] = col
            img[yt0 : yt1 + 1, xt1] = col
    return img
