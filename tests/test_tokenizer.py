import numpy as np

from ecfm.data.tokenizer import Region, build_patch


def test_build_patch_shape():
    events = np.array(
        [
            [1, 1, 0.1, 1],
            [2, 2, 0.2, 0],
            [3, 3, 0.3, 1],
        ],
        dtype=np.float32,
    )
    region = Region(x=0, y=0, t=0.0, dx=4, dy=4, dt=1.0, plane="xy")
    patch, total = build_patch(events, region, patch_size=8, time_bins=4)
    assert patch.shape == (1, 8, 8)
    assert total == 3.0

