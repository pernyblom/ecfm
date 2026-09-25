"""Actions act on sensing windows in normalized (x, y, t), not on events."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from ecfm.data.tokenizer import Region


@dataclass
class Action:
    name: str
    rotation: np.ndarray
    scale: np.ndarray
    translation: np.ndarray

    def vector(self) -> np.ndarray:
        # Continuous parameters permit angles/axes unseen during training.
        return np.concatenate([(self.rotation - np.eye(3)).ravel(),
                               np.log(self.scale), self.translation]).astype(np.float32)


def _vector(value, name: str) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    if value.shape != (3,) or not np.isfinite(value).all():
        raise ValueError(f"{name} must contain three finite values in x,y,t order")
    return value


def parse_actions(specs: list[dict]) -> list[Action]:
    actions = []
    for spec in specs:
        kind = spec['type']
        rotation, scale, translation = np.eye(3), np.ones(3), np.zeros(3)
        if kind == 'rotate':
            axis = spec['axis']
            if isinstance(axis, str):
                if axis.lower() not in ('x', 'y', 't'):
                    raise ValueError('rotation axis must be x, y, t or a vector')
                axis = np.eye(3)[('x', 'y', 't').index(axis.lower())]
            axis = _vector(axis, 'axis')
            if np.linalg.norm(axis) < 1e-12:
                raise ValueError('rotation axis cannot be zero')
            axis = axis / np.linalg.norm(axis)
            angle = np.deg2rad(float(spec['degrees']))
            if not np.isfinite(angle):
                raise ValueError('degrees must be finite')
            x, y, t = axis
            skew = np.array([[0, -t, y], [t, 0, -x], [-y, x, 0]])
            rotation = np.eye(3) + np.sin(angle) * skew + (1 - np.cos(angle)) * (skew @ skew)
        elif kind == 'scale':
            factor = spec['factor']
            scale = _vector([factor] * 3 if np.isscalar(factor) else factor, 'factor')
            if (scale <= 0).any():
                raise ValueError('scale factors must be positive')
        elif kind == 'translate':
            translation = _vector(spec['offset'], 'offset')
        elif kind != 'identity':
            raise ValueError(f'Unknown action type: {kind}')
        actions.append(Action(str(spec['name']), rotation, scale, translation))
    if not actions or len({a.name for a in actions}) != len(actions):
        raise ValueError('actions must be nonempty with unique names')
    return actions


def transform_region(region: Region, action: Action, width: int, height: int) -> Region:
    """Rotate centers around (0.5,0.5,0.5); resize about each local center.

    Windows remain axis aligned. Clamp sizes to the volume, then shift each
    window inside it. Spatial extents/origins are rounded to integer pixels.
    This explicit boundary rule is deterministic but not generally invertible.
    """
    units = np.array([width, height, 1.0])
    size = np.array([region.dx, region.dy, region.dt]) / units
    center = np.array([region.x, region.y, region.t]) / units + size / 2
    center = action.rotation @ (center - 0.5) + 0.5 + action.translation
    size = np.clip(size * action.scale, [1 / width, 1 / height, 1e-6], 1)
    pixels = np.maximum(1, np.rint(size[:2] * units[:2])).astype(int)
    size[:2] = pixels / units[:2]
    origin = np.clip(center - size / 2, 0, 1 - size)
    xy = np.rint(origin[:2] * units[:2]).astype(int)
    return Region(int(xy[0]), int(xy[1]), float(origin[2]),
                  int(pixels[0]), int(pixels[1]), float(size[2]), region.plane)
