"""Inference and HTTP checks for the manual reconstruction inspector."""
from copy import deepcopy
from io import BytesIO
import json
from pathlib import Path
from threading import Thread
from http.server import HTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen
import base64

import numpy as np
from PIL import Image
import pytest
import torch

from experiments.hierarchical_mae.config import load_config
from experiments.hierarchical_mae.model import HierarchicalMAE
from experiments.hierarchical_mae.visualize import Inspector, make_handler, manual_plan


@pytest.fixture
def inspector(tmp_path):
    cfg = load_config(Path(__file__).resolve().parents[1] / 'experiments/hierarchical_mae/configs/smoke.yaml')
    cfg['data'].update(root=str(tmp_path), image_width=8, image_height=8,
                       time_unit=1., crop_fraction=[.5, .9], spatial_crop_fraction=[1., 1.])
    rng = np.random.default_rng(9)
    for i in range(8):
        np.save(tmp_path / f'{i}.npy', np.column_stack((rng.integers(0, 8, (50, 2)),
                 np.linspace(0, 2, 50), rng.integers(0, 2, 50))))
    (tmp_path / 'train.txt').write_text(''.join(f'{i}.npy {i%2}\n' for i in range(8)))
    checkpoint = tmp_path / 'model.pt'
    torch.save(dict(config=cfg, model=HierarchicalMAE(cfg).state_dict(), epoch=3), checkpoint)
    return Inspector(checkpoint, device='cpu')


def test_images_masked_inference_and_crop_reuse(inspector):
    info = inspector.info()
    assert info['tokens'] == 18 and len(info['instances']) == 6
    request = dict(instance=0, targets=[2, 3], crop='training', epoch=4)
    original = inspector.inspect(request)
    view = inspector.view(request)
    assert inspector.view(request) is view
    output = inspector.inspect(request, reconstruct=True)
    assert output['targets'] == 2 and output['visible'] == 16
    assert output['patch_loss'] >= 0
    for before, after in zip(original['tokens'], output['tokens']):
        image = Image.open(BytesIO(base64.b64decode(after['image'].split(',')[1])))
        assert image.mode == 'RGB'
        if not after['target']:
            assert after['image'] == before['image']
    assert inspector.inspect(request, reconstruct=True) == output
    # Target pixels and event counts must not affect decoder predictions.
    plan = manual_plan(view, inspector.model.layout, [2, 3])
    changed = deepcopy(view)
    changed['patches']['l1_xy'][:, :2] = 999
    changed['log_counts'][:, 2:4] = 999
    with torch.inference_mode():
        a, b = inspector.model(view, plan), inspector.model(changed, plan)
    for key in a['predictions']:
        torch.testing.assert_close(a['predictions'][key], b['predictions'][key])
    assert inspector.view(dict(request, epoch=5)) is not view


def test_strict_exclusions_and_invalid_masks(inspector):
    request = dict(instance=0, targets=[2], strict=True)
    result = inspector.inspect(request, True)
    assert result['excluded'] > 0 and result['visible'] > 0
    assert not result['tokens'][0]['visible']  # Root overlaps every fine token.
    assert not result['tokens'][10]['visible']  # Alternate projection of target voxel.
    view = inspector.view(request)
    for ids in ([-1], [18], [True], ['2'], list(range(18))):
        with pytest.raises(ValueError):
            manual_plan(view, inspector.model.layout, ids)
    with pytest.raises(ValueError, match='visible'):
        inspector.inspect(dict(instance=0, targets=[0], strict=True), True)
    assert inspector.inspect(dict(instance=0, targets=[]), True)['patch_loss'] is None


def test_http_ui_and_validation(inspector):
    server = HTTPServer(('127.0.0.1', 0), make_handler(inspector))
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f'http://127.0.0.1:{server.server_port}'
    try:
        with urlopen(url) as response:
            assert b'MAE patch inspector' in response.read()
        with urlopen(url + '/api/info') as response:
            assert json.load(response)['tokens'] == 18
        payload = json.dumps(dict(instance=0, targets=[2])).encode()
        with urlopen(Request(url + '/api/reconstruct', data=payload)) as response:
            assert json.load(response)['targets'] == 1
        with pytest.raises(HTTPError) as error:
            urlopen(Request(url + '/api/reconstruct', data=b'{"instance":-1}'))
        assert error.value.code == 400
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
