"""Atomic, content-addressed CPU patch caches with optional reusable training crops."""
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import torch


def implementation_digest():
    from ecfm.data import tokenizer
    from experiments.region_worldmodel import data as event_data
    from . import data, rendering
    digest = hashlib.sha256()
    for path in (__file__, data.__file__, rendering.__file__, tokenizer.__file__, event_data.__file__):
        digest.update(Path(path).read_bytes())
    return digest.hexdigest()


class PatchCache:
    def __init__(self, cfg):
        d = cfg['data']
        self.options = d.get('patch_cache', {})
        self.directory = Path(self.options.get('dir', 'outputs/hierarchical_mae_patches'))
        self.spec = dict(version=1, implementation=implementation_digest(), hierarchy=cfg['hierarchy'],
                         torch_version=str(torch.__version__), numpy_version=str(np.__version__),
                         data={k: d.get(k) for k in ('image_width', 'image_height', 'time_unit',
                                                    'time_bins', 'patch_norm', 'cstr_max_count')})

    def entry(self, path, fraction, start):
        stat = path.stat()
        metadata = dict(self.spec, source=[str(path.resolve()), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns],
                        crop=[fraction, start])
        key = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
        return self.directory / key[:2] / f'{key}.pt', metadata

    def read(self, path, metadata, layout):
        if not path.exists() or self.options.get('rebuild', False):
            return None
        try:
            payload = torch.load(path, map_location='cpu', weights_only=True)
            view = payload['source']
            if payload['metadata'] != metadata:
                raise ValueError('metadata mismatch')
            tensors = [view['metadata'], view['log_counts'], *view['patches'].values()]
            if (view['metadata'].shape != (layout.count, 9) or view['log_counts'].shape != (layout.count,)
                    or view['valid_mask'].shape != (layout.count,) or view['valid_mask'].dtype != torch.bool
                    or not view['valid_mask'].all() or set(view['patches']) != {g.key for g in layout.groups}
                    or any(t.dtype != torch.float32 or not torch.isfinite(t).all() for t in tensors)
                    or any(view['patches'][g.key].shape != (g.stop-g.start, g.channels, g.size, g.size)
                           for g in layout.groups)):
                raise ValueError('invalid tensors')
            return view
        except Exception as error:
            raise ValueError(f'Invalid patch cache {path}; remove this entry or set data.patch_cache.rebuild: true') from error

    def write(self, path, metadata, view):
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(dir=path.parent, suffix='.tmp')
        os.close(fd)
        try:
            torch.save(dict(metadata=metadata, source=view), temporary)
            os.replace(temporary, path)
        finally:
            Path(temporary).unlink(missing_ok=True)
