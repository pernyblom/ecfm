import numpy as np

from ecfm.data.region_utils import sample_region


class _UpperBoundaryRng:
    def __init__(self) -> None:
        self.integer_calls = []

    def choice(self, values):
        return values[0]

    def integers(self, low, high):
        self.integer_calls.append((low, high))
        return high - 1

    def random(self):
        return 0.5


def test_sample_region_includes_last_spatial_origin() -> None:
    rng = _UpperBoundaryRng()

    region = sample_region(
        rng,
        image_width=10,
        image_height=8,
        region_scales=[4],
        region_scales_x=[4],
        region_scales_y=[3],
        region_time_scales=[0.5],
        plane_types_active=["xt"],
        fixed_region_sizes=True,
    )

    assert rng.integer_calls == [(0, 7), (0, 6)]
    assert region.x == 6
    assert region.y == 5
    assert region.x + region.dx == 10
    assert region.y + region.dy == 8


def test_sample_full_frame_region_has_zero_origin() -> None:
    rng = np.random.default_rng(0)

    region = sample_region(
        rng,
        image_width=10,
        image_height=8,
        region_scales=[10],
        region_scales_x=[10],
        region_scales_y=[8],
        region_time_scales=[1.0],
        plane_types_active=["xy"],
        fixed_region_sizes=True,
    )

    assert (region.x, region.y, region.t) == (0, 0, 0.0)
