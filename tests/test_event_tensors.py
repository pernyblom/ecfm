import numpy as np
import pytest
import torch

from ecfm.utils.event_representations import (
    render_event_tensor, event_representation_channels, representation_options_for,
)
from ecfm.utils.event_visualization import event_tensor_preview, resolve_event_visualization
from experiments.kalman_ml_forecasting.utils.config import resolve_representation_channels
from experiments.kalman_ml_forecasting.models.factory import build_model


def render(rep, options=None, events=None, **kwargs):
    events = [[1, 0, 25, 1], [2, 1, 75, 0]] if events is None else events
    return render_event_tensor(np.asarray(events, dtype=np.int64).reshape(-1, 4), rep,
        width=4, height=2, start_us=0, end_us=100, options=options, **kwargs)


def test_signed_voxels_and_polarity_split_interpolation():
    signed = render('voxel_grid', {'bins': 5, 'normalization': 'none'})
    assert signed.shape == (5, 2, 4) and signed.dtype == np.float32
    assert signed[1, 0, 1] == 1 and signed[3, 1, 2] == -1
    split = render('voxel_grid', {'bins': 5, 'split_polarity': True, 'normalization': 'none'})
    assert split.shape == (10, 2, 4)
    np.testing.assert_array_equal(split[:5] - split[5:], signed)
    fractional = render('voxel_grid', {'bins': 4, 'normalization': 'none'})
    assert fractional[0, 0, 1] == .25 and fractional[1, 0, 1] == .75


def test_stacks_preserve_every_temporal_and_polarity_channel():
    tensor = render('event_stack', {'bins': 4, 'split_polarity': True, 'normalization': 'none'})
    assert tensor.shape == (8, 2, 4)
    assert tensor[1, 0, 1] == 1 and tensor[7, 1, 2] == 1
    assert tensor.sum() == 2


@pytest.mark.parametrize('rep', ['xt_event_stack', 'yt_event_stack'])
def test_projected_stacks_count_slices_with_local_time_and_collapsed_space(rep):
    events = [[1, 0, 0, 1], [1, 1, 0, 0], [1, 0, 29, -1],
              [2, 1, 30, 1], [2, 1, 59, 0], [3, 1, 89, 1],
              [0, 0, -1, 1], [0, 0, 90, 1], [4, 0, 10, 1], [0, 2, 10, 1]]
    tensor = render_event_tensor(np.array(events), rep, width=4, height=2,
        start_us=0, end_us=90, options={'time_bins': 3, 'normalization': 'none'})
    expected = np.zeros((3, 3, 4) if rep.startswith('xt') else (3, 2, 3), np.float32)
    if rep.startswith('xt'):
        expected[0, 0, 1] = 2  # Different y coordinates collapse onto the same XT cell.
        expected[0, 2, 1] = expected[1, 0, 2] = expected[1, 2, 2] = expected[2, 2, 3] = 1
    else:
        expected[0, 0, 0] = expected[0, 1, 0] = expected[0, 0, 2] = 1
        expected[1, 1, 0] = expected[1, 1, 2] = expected[2, 1, 2] = 1
    np.testing.assert_array_equal(tensor, expected)
    reversed_tensor = render_event_tensor(np.array(events)[::-1], rep, width=4, height=2,
        start_us=0, end_us=90, options={'time_bins': 3, 'normalization': 'none'})
    np.testing.assert_array_equal(reversed_tensor, expected)


@pytest.mark.parametrize('rep', ['xt_event_stack', 'yt_event_stack'])
def test_projected_stack_options_polarity_grid_normalization_and_timestamps(rep):
    origin = 2**55
    events = np.array([[0, 0, origin, 1], [0, 0, origin, 1],
                       [0, 0, origin+25, 0], [3, 1, origin+99, -1]], dtype=np.int64)
    raw_opts = {'bins': 4, 'time_bins': 5, 'split_polarity': True, 'normalization': 'none'}
    raw = render_event_tensor(events, rep, width=4, height=2, start_us=origin,
                              end_us=origin+100, options=raw_opts)
    assert raw.shape == ((8, 5, 4) if rep.startswith('xt') else (8, 2, 5))
    assert raw[0].sum() == 2 and raw[5].sum() == raw[7].sum() == 1
    assert raw.sum() == 4 and raw.dtype == np.float32
    normalized = render_event_tensor(events, rep, width=4, height=2, start_us=origin,
        end_us=origin+100, options={**raw_opts, 'normalization': 'max'})
    np.testing.assert_array_equal(normalized, raw/2)
    size = (8, 5) if rep.startswith('xt') else (10, 4)
    grid = render_event_tensor(events, rep+'_2x1', width=4, height=2, start_us=origin,
        end_us=origin+100, output_size=size, options={**raw_opts, 'normalization': 'max'})
    assert grid.shape == (8, size[1], size[0]) and grid[7, :, size[0]//2:].max() == 1
    assert not render(rep, {'bins': 1, 'time_bins': 2}, events=[]).any()
    assert event_representation_channels(rep, raw_opts) == 8


def test_multiscale_surfaces_have_configurable_decays_and_polarities():
    opts = {'decay_ms': [.01, .02, .04, .08], 'split_polarity': True}
    tensor = render('multi_channel_time_surface', opts)
    assert tensor.shape == (8, 2, 4)
    np.testing.assert_allclose(tensor[:4, 0, 1], np.exp(-75/np.array([10, 20, 40, 80])), rtol=1e-6)
    assert not tensor[4:, 0, 1].any()
    np.testing.assert_allclose(tensor[4:, 1, 2], np.exp(-25/np.array([10, 20, 40, 80])), rtol=1e-6)
    assert render('time_surface', {'include_activity': False}).shape[0] == 2


def test_tensor_grid_resize_empty_and_large_integer_timestamps():
    tensor = render('voxel_grid_2x1', {'bins': 7}, output_size=(8, 4))
    assert tensor.shape == (7, 4, 8) and tensor.min() < 0
    assert not render('voxel_grid', {'bins': 7}, events=[]).any()
    origin = 2**55
    events = np.array([[1, 0, origin + 99, 1]], dtype=np.int64)
    tensor = render_event_tensor(events, 'time_surface_linear', width=4, height=2,
                                start_us=origin, end_us=origin+100)
    assert tensor[0, 0, 1] == pytest.approx(.99)
    assert not render('event_stack', {'bins': 1}, events=[[0, 0, -1, 1], [0, 0, 100, 1]]).any()


@pytest.mark.parametrize('rep,opts', [
    ('voxel_grid', {'bins': 0}), ('event_stack', {'bins': 2.5}),
    ('event_stack', {'split_polarity': 'true'}), ('time_surface', {'bins': 5}),
    ('multi_channel_time_surface', {'decay_ms': []}),
    ('multi_channel_time_surface', {'decay_ms': [1, float('nan')]}),
    ('xt_event_stack', {'bins': 0}), ('yt_event_stack', {'bins': True}),
    ('xt_event_stack', {'time_bins': 0}), ('yt_event_stack', {'time_bins': 2.5}),
    ('xt_event_stack', {'normalization': 'invalid'}),
])
def test_invalid_representation_options_fail_before_training(rep, opts):
    with pytest.raises(ValueError):
        event_representation_channels(rep, opts)


def test_channel_resolution_and_grid_option_precedence():
    options = {'event_stack': {'bins': 4}, 'event_stack_2x2': {'bins': 5, 'split_polarity': True}}
    assert representation_options_for('event_stack_2x2', options)['bins'] == 5
    cfg = {'representations': ['event_stack_2x2', 'voxel_grid', 'rgb'],
           'representation_options': options, 'event_image_source': 'indexed'}
    assert resolve_representation_channels(cfg) == {'event_stack_2x2': 10, 'voxel_grid': 3, 'rgb': 3}
    cfg['representation_channels'] = {'event_stack_2x2': 7}
    with pytest.raises(ValueError, match='disagrees'):
        resolve_representation_channels(cfg)


@pytest.mark.parametrize('backbone', ['small_cnn', 'resnet18'])
def test_encoder_branches_accept_different_channels(backbone):
    cfg = {'data': {'representations': ['event_stack', 'time_surface', 'padded_rgb'],
                   'image_sizes': {'event_stack': [32, 32], 'time_surface': [32, 32], 'padded_rgb': [32, 32]},
                   'representation_options': {'event_stack': {'bins': 5, 'split_polarity': True},
                                              'time_surface': {'include_activity': False}},
                   'representation_sequences': {'event_stack': {'length': 2, 'stride': 1}}},
           'model': {'history_steps': 1, 'fusion_hidden_dim': 8, 'state_hidden_dim': 8, 'residual_hidden_dim': 8,
                     'backbone': {'type': backbone, 'channels': [4], 'out_dim': 8}}}
    model = build_model(cfg, torch.device('cpu')).eval()
    assert model.representation_channels == {'event_stack': 10, 'time_surface': 2, 'padded_rgb': 3}
    inputs = {'event_stack': torch.randn(2, 2, 10, 32, 32), 'time_surface': torch.randn(2, 2, 32, 32),
              'padded_rgb': torch.randn(2, 3, 32, 32)}
    past = torch.tensor([[[.4, .4, .2, .2], [.41, .41, .2, .2]]]).expand(2, -1, -1)
    prediction = model(inputs, past, torch.tensor([[0., .1]]).expand(2, -1), torch.tensor([[.2]]).expand(2, -1))
    prediction.sum().backward()
    assert prediction.shape == (2, 1, 4)
    for encoder in model.encoders.values():
        assert any(p.grad is not None for p in encoder.parameters())


def test_visualizations_are_configurable_and_do_not_change_tensors():
    tensor = np.arange(8*2*4, dtype=np.float32).reshape(8, 2, 4)/64
    original = tensor.copy()
    assert event_tensor_preview(tensor).size == (4, 2)
    cfg = {'visualization': {'event_images': {'mode': 'mean', 'normalization': 'none',
           'by_representation': {'event_stack': {'mode': 'grid', 'columns': 2, 'labels': False},
                                 'event_stack_2x2': {'mode': 'rgb', 'channels': [7, 3, 0]}}}}}
    grid = event_tensor_preview(tensor, resolve_event_visualization(cfg, 'event_stack'))
    assert grid.size == (8, 8)
    rgb = np.asarray(event_tensor_preview(tensor, resolve_event_visualization(cfg, 'event_stack_2x2')))
    np.testing.assert_array_equal(rgb, np.rint(tensor[[7, 3, 0]].transpose(1, 2, 0)*255).astype(np.uint8))
    np.testing.assert_array_equal(tensor, original)
    signed = render('voxel_grid', {'bins': 5})
    preview = np.asarray(event_tensor_preview(signed))
    assert preview[0, 1, 0] > 0 and preview[1, 2, 2] > 0


@pytest.mark.parametrize('settings', [{'mode': 'rgb'}, {'channels': [10]}, {'mode': 'unknown'}, {'mode': 'grid', 'columns': 0}])
def test_preview_rejects_invalid_settings(settings):
    with pytest.raises(ValueError):
        event_tensor_preview(np.zeros((5, 2, 4)), settings)


@pytest.mark.parametrize('detector', ['centernet', 'detr_lite'])
def test_shared_detection_encoder_factories_accept_multichannel_inputs(detector):
    from experiments.object_detection.models.factory import build_model as build_detector
    cfg = {'data': {'representations': ['event_stack', 'rgb'], 'frame_size': [32, 32],
                   'image_sizes': {'event_stack': [32, 32], 'rgb': [32, 32]},
                   'representation_options': {'event_stack': {'bins': 5, 'split_polarity': True}}},
           'model': {'detector': detector, 'backbone': {'type': 'small_cnn', 'channels': [4], 'out_dim': 8},
                     'fusion_hidden_dim': 8, 'box_hidden_dim': 8, 'query_hidden_dim': 8,
                     'centernet_hidden_dim': 8, 'topk': 2}}
    model = build_detector(cfg, torch.device('cpu'))
    result = model({'event_stack': torch.randn(2, 10, 32, 32), 'rgb': torch.randn(2, 3, 32, 32)})
    assert result['boxes'].shape[0] == 2 and result['boxes'].shape[-1] == 4
    features = result['centernet_heatmap_logits'] if detector == 'centernet' else result['boxes']
    features.sum().backward()
    assert any(p.grad is not None for p in model.encoders['event_stack'].parameters())


def test_benchmark_inputs_follow_per_representation_channel_counts():
    from experiments.kalman_ml_forecasting.benchmark_inference import _make_inputs
    cfg = {'data': {'representations': ['event_stack', 'rgb'],
                   'image_sizes': {'event_stack': [8, 8], 'rgb': [8, 8]},
                   'representation_options': {'event_stack': {'bins': 5, 'split_polarity': True}},
                   'representation_sequences': {'event_stack': {'length': 4}}}, 'model': {}}
    inputs, _, _, _ = _make_inputs(cfg, torch.device('cpu'), 2, 3, 4, seed=1, step_s=.1)
    assert inputs['event_stack'].shape == (2, 4, 10, 8, 8)
    assert inputs['rgb'].shape == (2, 3, 8, 8)
