"""Fingerprint checkpoints, source files, crop/selection configuration and implementation."""
import hashlib
import json
import os
from pathlib import Path
import tempfile

import torch
from torch.utils.data import TensorDataset

from ecfm.data import tokenizer
from experiments.region_worldmodel import data as event_data
from experiments.region_worldmodel.feature_cache import file_digest
from .loading import make_loader, to_device
from . import data, model, masking, rendering, learned_selector
from .masking import TokenPlan, make_plan


def cache_metadata(dataset, checkpoint, checkpoint_digest=None):
    cfg = dataset.cfg
    return dict(version=1, checkpoint=checkpoint_digest or file_digest(checkpoint),
        entries=[[str(p.resolve()), label, p.stat().st_size, p.stat().st_mtime_ns, p.stat().st_ctime_ns]
                 for p, label in dataset.entries],
        data=cfg['data'], hierarchy=cfg['hierarchy'], model=cfg['model'],
        selection=cfg['downstream'].get('selection', {}),
        learned_selector=cfg['downstream'].get('learned_selector'),
        seed=cfg['train']['seed'], torch_version=str(torch.__version__),
        implementation=[file_digest(p) for p in [__file__, data.__file__, model.__file__,
                                                masking.__file__, rendering.__file__, learned_selector.__file__,
                                                tokenizer.__file__, event_data.__file__]])


@torch.no_grad()
def cached_features(backbone, dataset, checkpoint, device, split, checkpoint_digest=None):
    if dataset.training or any(p.requires_grad for p in backbone.parameters()):
        raise ValueError('Feature caches require fixed crops and a frozen encoder')
    metadata = cache_metadata(dataset, checkpoint, checkpoint_digest)
    key = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
    options = dataset.cfg['downstream']['feature_cache']
    directory = Path(options['dir'])
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f'{split}_{key}.pt'
    expected_labels = torch.tensor([label for _, label in dataset.entries])
    if path.exists() and not options.get('rebuild', False):
        payload = torch.load(path, map_location='cpu', weights_only=True)
        features, labels = payload['features'], payload['labels']
        if (payload['metadata'] != metadata or not torch.equal(labels, expected_labels)
                or features.shape != (len(dataset), dataset.cfg['model']['embed_dim'])
                or features.dtype != torch.float32 or not torch.isfinite(features).all()):
            raise ValueError(f'Invalid cache: {path}; set feature_cache.rebuild: true')
        print(f'Features {split}: cache hit {path}', flush=True)
        return TensorDataset(features, labels)
    backbone.eval()
    features, labels = [], []
    t = dataset.cfg['train']
    loader = make_loader(dataset, options.get('batch_size', dataset.cfg['downstream']['batch_size']), t['num_workers'])
    # Fixed per-recording random selection, independent of extraction batch size and cache hits.
    devices = [torch.device(device).index or 0] if torch.device(device).type == 'cuda' else []
    index = 0
    with torch.random.fork_rng(devices=devices):
        for raw in loader:
            view = to_device(raw['source'], device)
            if getattr(backbone, 'is_learned_selector', False):
                features.append(backbone.features(view).float().cpu())
                labels.append(raw['label'])
                continue
            plans = []
            for b in range(len(raw['label'])):
                torch.manual_seed(t['seed']+index)
                single = {k: v[b:b+1] for k, v in view.items() if not isinstance(v, dict)}
                plans.append(make_plan(single, backbone.layout, selection=dataset.cfg['downstream'].get('selection', {})))
                index += 1
            plan = TokenPlan(torch.cat([p.visible for p in plans]), torch.cat([p.target for p in plans]))
            features.append(backbone.features(view, plan=plan).float().cpu())
            labels.append(raw['label'])
    features, labels = torch.cat(features), torch.cat(labels)
    if not torch.isfinite(features).all():
        raise FloatingPointError('Nonfinite features')
    fd, temporary = tempfile.mkstemp(dir=directory, suffix='.tmp')
    os.close(fd)
    try:
        torch.save(dict(metadata=metadata, features=features, labels=labels), temporary)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    print(f'Features {split}: saved {path}', flush=True)
    return TensorDataset(features, labels)
