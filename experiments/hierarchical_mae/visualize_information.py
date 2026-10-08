"""Local, checkpoint-free browser explorer for patch information scores."""
from __future__ import annotations

import argparse
from copy import deepcopy
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import math
from pathlib import Path
from urllib.parse import urlparse
import webbrowser

import torch
from torch.utils.data import default_collate

from .config import load_config, validate
from .data import HierarchyDataset, Layout, splits
from .information_selection import METRICS, selection_scores, settings, validate_settings
from .visualize import patch_image


class InformationInspector:
    def __init__(self, config, data_root=None, split='train'):
        self.cfg = deepcopy(config if isinstance(config, dict) else load_config(config))
        if data_root is not None:
            self.cfg['data']['root'] = str(data_root)
        self.cfg['data']['information_selection'] = settings(self.cfg) or {
            'bins': [8, 8, 8], 'support_saturation': 3.}
        validate(self.cfg)
        # Preserve the training crop bank, but never read/write disk patch caches.
        self.cfg['data'].setdefault('patch_cache', {})['enabled'] = False
        if split not in ('train', 'validation'):
            raise ValueError('Invalid split')
        train, val, _ = splits(self.cfg)
        self.entries = train if split == 'train' else val
        if not self.entries:
            raise ValueError(f'Empty {split} split')
        self.split, self.layout = split, Layout(self.cfg)
        self.cached_key = self.cached_view = self.cached_tokens = None

    def info(self):
        selection = self.cfg.get('downstream', {}).get('selection', {})
        return dict(split=self.split, tokens=self.layout.count,
                    defaults=dict(self.cfg['data']['information_selection'],
                                  metric=selection.get('metric', 'support')),
                    groups=[dict(vars(g), splits=self.cfg['hierarchy']['levels'][g.level]['splits'])
                            for g in self.layout.groups],
                    instances=[dict(id=i, name=p.name, label=label)
                               for i, (p, label) in enumerate(self.entries)])

    @torch.inference_mode()
    def inspect(self, request):
        index, crop, epoch = request.get('instance', 0), request.get('crop', 'evaluation'), request.get('epoch', 0)
        if type(index) is not int or not 0 <= index < len(self.entries):
            raise ValueError('Invalid instance')
        if crop not in ('evaluation', 'training') or type(epoch) is not int or not 0 <= epoch <= 1_000_000:
            raise ValueError('Invalid crop or epoch')
        options = {k: request.get(k, v) for k, v in self.cfg['data']['information_selection'].items()}
        validate_settings({'data': {'information_selection': options}})
        if any(v > 64 for v in options['bins']):
            raise ValueError('Interactive histogram bins must be between 2 and 64 per axis')
        metric = request.get('metric', 'support')
        mode = request.get('mode', 'information')
        weight = request.get('activity_weight', .5)
        group = request.get('group', 'all')
        if metric not in METRICS or mode not in ('information', 'blend', 'product'):
            raise ValueError('Invalid metric or ranking mode')
        if type(weight) not in (int, float) or not math.isfinite(weight) or not 0 <= weight <= 1:
            raise ValueError('Activity weight must be between 0 and 1')
        if group != 'all' and group not in {g.key for g in self.layout.groups}:
            raise ValueError('Invalid group')
        key = (index, crop, epoch if crop == 'training' else 0,
               tuple(options['bins']), options['support_saturation'])
        if key != self.cached_key:
            cfg = deepcopy(self.cfg)
            cfg['data']['information_selection'] = options
            ds = HierarchyDataset(cfg, self.entries, training=crop == 'training')
            view = default_collate([ds[(index, epoch)]])['source']
            tokens = []
            for g in self.layout.groups:
                nx, ny, _ = cfg['hierarchy']['levels'][g.level]['splits']
                for local, idx in enumerate(range(g.start, g.stop)):
                    tokens.append(dict(id=idx, group=g.key, level=g.level, representation=g.representation,
                        cell=[local % nx, (local // nx) % ny, local // (nx * ny)],
                        image=patch_image(view['patches'][g.key][0, local]),
                        count=round(float(view['log_counts'][0, idx].expm1())),
                        duration=float(view['metadata'][0, idx, 7].expm1()),
                        metrics=dict(zip(METRICS, view['information_scores'][0, idx].tolist()))))
            self.cached_key, self.cached_view, self.cached_tokens = key, view, tokens
        view = self.cached_view
        eligible = view['valid_mask'].clone()
        if group != 'all':
            selected = next(g for g in self.layout.groups if g.key == group)
            eligible[:] = False
            eligible[:, selected.start:selected.stop] = view['valid_mask'][:, selected.start:selected.stop]
        scores = selection_scores(view, eligible, dict(metric=metric,
            strategy='information' if mode == 'information' else 'activity_information',
            combination=mode if mode != 'information' else 'blend', activity_weight=weight))[0]
        tokens = [dict(t, score=float(scores[t['id']])) for t in self.cached_tokens if eligible[0, t['id']]]
        tokens.sort(key=lambda t: (-t['score'], t['id']))
        return dict(tokens=tokens, settings=dict(options, metric=metric, mode=mode,
                    activity_weight=weight, group=group), instance=index, crop=crop, epoch=epoch)


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
                self.send(200, Path(__file__).with_name('visualize_information.html').read_bytes(),
                          'text/html; charset=utf-8')
            elif route == '/api/info':
                self.send(200, inspector.info())
            else:
                self.send(404, dict(error='Not found'))

        def do_POST(self):
            if urlparse(self.path).path != '/api/instance':
                self.send(404, dict(error='Not found'))
                return
            try:
                length = int(self.headers.get('Content-Length', 0))
                if not 0 < length <= 16_384:
                    raise ValueError('Invalid request size')
                request = json.loads(self.rfile.read(length))
                if not isinstance(request, dict):
                    raise ValueError('Expected a JSON object')
                result = inspector.inspect(request)
                self.send(200, result)
            except (ValueError, TypeError, IndexError) as exc:
                self.send(400, dict(error=str(exc)))
            except Exception as exc:
                self.log_error('Information inspection failed: %s', exc)
                self.send(500, dict(error=str(exc)))
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--split', choices=('train', 'validation'), default='train')
    parser.add_argument('--port', type=int, default=8766)
    parser.add_argument('--no-browser', action='store_true')
    args = parser.parse_args()
    inspector = InformationInspector(args.config, args.data_root, args.split)
    with HTTPServer(('127.0.0.1', args.port), make_handler(inspector)) as server:
        url = f'http://127.0.0.1:{server.server_port}'
        print(f'Patch information explorer: {url} ({inspector.layout.count} tokens). Ctrl+C to stop.', flush=True)
        if not args.no_browser:
            webbrowser.open(url)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == '__main__':
    main()
