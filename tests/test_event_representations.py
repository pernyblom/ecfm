import numpy as np
import pytest

from ecfm.utils.event_representations import (
    TEMPORAL_IMAGE_REPRESENTATIONS, render_event_representation,
)


def render(events, rep, **kwargs):
    return render_event_representation(np.asarray(events, dtype=np.int64).reshape(-1, 4), rep,
                                       width=4, height=2, start_us=0, end_us=100, **kwargs)


def test_time_surface_uses_latest_per_polarity_and_window_end():
    events = [[1, 0, 80, 1], [1, 0, 20, 1], [1, 0, 50, 0],
              [2, 0, 100, 1], [3, 0, -1, 1]]
    linear = render(events, 'time_surface_linear')
    np.testing.assert_array_equal(linear[0, 1], [204, 204, 127])
    assert not linear[:, 2:].any()
    exponential = render(events, 'time_surface', time_surface_tau_us=50)
    np.testing.assert_array_equal(exponential[0, 1], [170, 170, 93])
    assert not exponential[1].any()


def test_multichannel_time_surface_has_distinct_decay_scales():
    image = render([[1, 0, 50, 0]], 'multi_channel_time_surface', time_surface_tau_us=100)
    np.testing.assert_array_equal(image[0, 1], [34, 93, 154])
    assert not image[1].any()


def test_event_stack_counts_three_slices_with_joint_normalization():
    image = render([[1, 0, 0, 1], [1, 0, 10, 0], [1, 0, 50, 1], [1, 0, 99, 0]], 'event_stack')
    np.testing.assert_array_equal(image[0, 1], [255, 127, 127])


def test_voxel_grid_splats_signed_polarity_and_cancels():
    image = render([[1, 0, 25, 1], [2, 0, 50, 0],
                    [3, 0, 50, 1], [3, 0, 50, 0]], 'voxel_grid')
    np.testing.assert_array_equal(image[0, 1], [192, 192, 128])
    np.testing.assert_array_equal(image[0, 2], [128, 1, 128])
    np.testing.assert_array_equal(image[0, 3], [128, 128, 128])


@pytest.mark.parametrize('rep', sorted(TEMPORAL_IMAGE_REPRESENTATIONS))
def test_empty_resize_and_grid_alias(rep):
    image = render([], rep, output_size=(8, 4))
    assert image.shape == (4, 8, 3) and image.dtype == np.uint8
    assert np.all(image == (128 if rep == 'voxel_grid' else 0))
    events = [[0, 0, 50, 1], [3, 1, 50, 0]]
    np.testing.assert_array_equal(render(events, rep), render(events, rep + '_2x1'))


def test_voxel_grid_normalizes_fractional_splats_and_each_spatial_cell():
    image = render([[1, 0, 25, 1]], 'voxel_grid')
    np.testing.assert_array_equal(image[0, 1], [255, 255, 128])
    split = render([[0, 0, 50, 1], [3, 1, 75, 0]], 'voxel_grid_2x1')
    np.testing.assert_array_equal(split[1, 3], [128, 1, 1])


def test_time_surface_preserves_large_integer_timestamp_differences():
    base = 2**55
    events = np.array([[1, 0, base + 99, 1]], dtype=np.int64)
    result = render_event_representation(events, 'time_surface_linear', width=4, height=2,
                                          start_us=base, end_us=base + 100)
    np.testing.assert_array_equal(result[0, 1], [252, 252, 0])


@pytest.mark.parametrize('tau', [0, -1, float('nan'), float('inf')])
def test_invalid_decay_rejected(tau):
    with pytest.raises(ValueError, match='tau'):
        render([], 'time_surface', time_surface_tau_us=tau)
