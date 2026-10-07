"""Local browser UI for manual hierarchical MAE reconstruction inspection."""
from __future__ import annotations

import argparse
import base64
from copy import deepcopy
from http.server import BaseHTTPRequestHandler, HTTPServer
from io import BytesIO
import json
from pathlib import Path
from urllib.parse import urlparse
import webbrowser

from PIL import Image
import numpy as np
import torch
from torch.utils.data import default_collate

from .config import load_config, validate
from .data import HierarchyDataset, splits
from .loading import to_device
from .masking import TokenPlan, overlaps
from .model import HierarchicalMAE


def patch_image(patch):
    """Use the training inspector's fixed RGB mapping; clip display only."""
    patch = patch.detach().cpu().clamp(0, 1)
    if patch.shape[0] == 2:
        patch = torch.stack((patch[1], torch.zeros_like(patch[0]), patch[0]))
    pixels = (patch.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    stream = BytesIO()
    Image.fromarray(pixels).save(stream, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(stream.getvalue()).decode('ascii')


def manual_plan(view, layout, targets, strict=False):
    if not isinstance(targets, list) or any(type(i) is not int or not 0 <= i < layout.count for i in targets):
        raise ValueError('Targets must be a list of valid integer token IDs')
    if type(strict) is not bool:
        raise ValueError('Strict overlap exclusion must be boolean')
    target = torch.zeros_like(view['valid_mask'])
    target[:, targets] = True
    visible = view['valid_mask'] & ~target
    if strict and targets:
        intersect = overlaps(layout.boxes.to(target.device))[targets].any(0)
        visible &= ~intersect[None]
    plan = TokenPlan(visible, target)
    plan.validate(view['valid_mask'])
    return plan


class Inspector:
    def __init__(self, checkpoint, config=None, device='auto', data_root=None, split='train'):
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        self.cfg = deepcopy(load_config(config) if config else state.get('config'))
        if self.cfg is None:
            raise ValueError('Checkpoint has no config; supply --config')
        if config and 'config' in state:
            # Geometry buffers alone cannot detect changed token order/representations.
            for key in ('hierarchy', 'model'):
                if self.cfg[key] != state['config'][key]:
                    raise ValueError(f'Config {key} differs from checkpoint')
        if data_root:
            self.cfg['data']['root'] = str(data_root)
        validate(self.cfg)
        self.device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if device == 'auto' else device)
        self.model = HierarchicalMAE(self.cfg).to(self.device).eval()
        self.model.load_state_dict(state['model'], strict=True)
        train, val, _ = splits(self.cfg)
        self.entries = train if split == 'train' else val
        if not self.entries:
            raise ValueError(f'Empty {split} split')
        # Inspection should not populate or rebuild large on-disk training caches.
        self.cfg['data']['patch_cache'] = {'enabled': False}
        self.datasets = {mode: HierarchyDataset(self.cfg, self.entries, training=mode == 'training')
                         for mode in ('full', 'training')}
        self.cached_key = self.cached_view = None
        self.checkpoint, self.epoch, self.split = str(checkpoint), state.get('epoch'), split

    def info(self):
        return dict(checkpoint=self.checkpoint, epoch=self.epoch, device=str(self.device), split=self.split,
                    tokens=self.model.layout.count, groups=[dict(vars(g), splits=self.cfg['hierarchy']['levels'][g.level]['splits'])
                    for g in self.model.layout.groups],
                    instances=[dict(id=i, name=p.name, label=label) for i, (p, label) in enumerate(self.entries)])

    def view(self, request):
        index, mode, epoch = request.get('instance', 0), request.get('crop', 'full'), request.get('epoch', 0)
        if type(index) is not int or not 0 <= index < len(self.entries):
            raise ValueError('Invalid instance')
        if mode not in self.datasets or type(epoch) is not int or not 0 <= epoch <= 1_000_000:
            raise ValueError('Invalid crop mode or crop epoch')
        key = (index, mode, epoch if mode == 'training' else 0)
        if key != self.cached_key:
            ds = self.datasets[mode]
            self.cached_view = to_device(default_collate([ds[(index, epoch)]])['source'], self.device)
            self.cached_key = key
        return self.cached_view

    @torch.inference_mode()
    def inspect(self, request, reconstruct=False):
        view = self.view(request)
        layout = self.model.layout
        plan = manual_plan(view, layout, request.get('targets', []) if reconstruct else [], request.get('strict', False))
        result = self.model(view, plan=plan) if plan.target.any() else None
        tokens = []
        for g in layout.groups:
            for local, idx in enumerate(range(g.start, g.stop)):
                original = view['patches'][g.key][0, local]
                target = bool(plan.target[0, idx])
                displayed = result['predictions'][g.key][0, local] if target else original
                token = dict(id=idx, image=patch_image(displayed), visible=bool(plan.visible[0, idx]), target=target)
                if not reconstruct:
                    token.update(count=float(view['log_counts'][0, idx].expm1()),
                                 duration=float(view['metadata'][0, idx, 7].expm1()))
                if target:
                    token.update(mse=float((displayed-original).square().mean()),
                                 predicted_log_count=float(result['count_predictions'][0, idx]))
                tokens.append(token)
        return dict(tokens=tokens, visible=int(plan.visible.sum()), targets=int(plan.target.sum()),
                    excluded=int((view['valid_mask'] & ~plan.visible & ~plan.target).sum()),
                    patch_loss=float(result['patch_loss']) if result else None,
                    count_loss=float(result['count_loss']) if result else None)


def make_handler(inspector):
    class Handler(BaseHTTPRequestHandler):
        def send(self, status, payload, content_type='application/json'):
            body = json.dumps(payload, allow_nan=False).encode() if content_type == 'application/json' else payload
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            route = urlparse(self.path).path
            if route == '/':
                self.send(200, Path(__file__).with_name('visualize.html').read_bytes(), 'text/html; charset=utf-8')
            elif route == '/api/info':
                self.send(200, inspector.info())
            else:
                self.send(404, dict(error='Not found'))

        def do_POST(self):
            route = urlparse(self.path).path
            if route not in ('/api/instance', '/api/reconstruct'):
                self.send(404, dict(error='Not found'))
                return
            try:
                length = int(self.headers.get('Content-Length', 0))
                if not 0 < length <= 1_000_000:
                    raise ValueError('Invalid request size')
                request = json.loads(self.rfile.read(length))
                if not isinstance(request, dict):
                    raise ValueError('Expected a JSON object')
                self.send(200, inspector.inspect(request, route == '/api/reconstruct'))
            except (ValueError, IndexError, TypeError) as exc:
                self.send(400, dict(error=str(exc)))
            except Exception as exc:
                self.log_error('Inspection failed: %s', exc)
                self.send(500, dict(error=str(exc)))

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--config', type=Path, help='Optional config; defaults to the checkpoint config')
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--device', default='auto', help='auto, cpu, cuda, cuda:0, ...')
    parser.add_argument('--split', choices=('train', 'validation'), default='train')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--no-browser', action='store_true')
    args = parser.parse_args()
    inspector = Inspector(args.checkpoint, args.config, args.device, args.data_root, args.split)
    with HTTPServer(('127.0.0.1', args.port), make_handler(inspector)) as server:
        url = f'http://127.0.0.1:{server.server_port}'
        print(f'MAE inspector: {url} ({inspector.device}, {inspector.model.layout.count} tokens). Ctrl+C to stop.', flush=True)
        if not args.no_browser:
            webbrowser.open(url)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == '__main__':
    main()
