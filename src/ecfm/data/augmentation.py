from __future__ import annotations

import math
from typing import Tuple

import numpy as np


def rotate_events_90(
    events: np.ndarray, image_size: Tuple[int, int], k: int
) -> np.ndarray:
    """Rotate events by 90-degree steps around image center."""
    if events.ndim != 2 or events.shape[1] != 4:
        raise ValueError("events must be shaped [N, 4] with (x, y, t, p)")
    k = k % 4
    if k == 0:
        return events.copy()

    h, w = image_size
    x = events[:, 0].copy()
    y = events[:, 1].copy()
    t = events[:, 2].copy()
    p = events[:, 3].copy()

    if k == 1:
        x_new = h - 1 - y
        y_new = x
    elif k == 2:
        x_new = w - 1 - x
        y_new = h - 1 - y
    else:
        x_new = y
        y_new = w - 1 - x

    return np.stack([x_new, y_new, t, p], axis=1)


def rotate_events(
    events: np.ndarray, image_size: Tuple[int, int], angle_deg: float
) -> np.ndarray:
    """Rotate events by an arbitrary angle using nearest-neighbor rounding."""
    if events.ndim != 2 or events.shape[1] != 4:
        raise ValueError("events must be shaped [N, 4] with (x, y, t, p)")

    h, w = image_size
    cx = (w - 1) / 2.0
    cy = (h - 1) / 2.0
    angle = math.radians(angle_deg)
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)

    x = events[:, 0] - cx
    y = events[:, 1] - cy
    x_new = cos_a * x - sin_a * y + cx
    y_new = sin_a * x + cos_a * y + cy

    x_new = np.rint(x_new).clip(0, w - 1)
    y_new = np.rint(y_new).clip(0, h - 1)

    return np.stack([x_new, y_new, events[:, 2], events[:, 3]], axis=1)

