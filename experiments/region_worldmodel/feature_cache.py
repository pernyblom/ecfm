"""CPU feature caches for fixed observations and a frozen linear-probe encoder."""
from __future__ import annotations

import hashlib
import json
import os
from copy import deepcopy
from pathlib import Path
import tempfile
import time

import torch
from torch.utils.data import Dataset

from ecfm.data import region_utils, tokenizer
from ecfm.models import mae, patch_encoder, rel_attention
from . import data, model, regions
from .train import make_loader, to_device


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def cache_metadata(dataset, checkpoint_digest):
    entries = []
    for path, label in dataset.entries:
        stat = path.stat()
        entries.append([str(path.resolve()), label, stat.st_size, stat.st_mtime_ns,
                        stat.st_ctime_ns])
    return dict(version=1, checkpoint=checkpoint_digest, entries=entries,
                data=deepcopy(dataset.cfg['data']), model=deepcopy(dataset.cfg['model']),
                regions=deepcopy(dataset.cfg['downstream'].get('regions', {})),
                implementation=[file_digest(module.__file__) for module in
                    (data, model, regions, region_utils, tokenizer, mae, patch_encoder, rel_attention)],
                torch_version=str(torch.__version__))


class FeatureDataset(Dataset):
    paired = False

    def __init__(self, features, labels, max_regions, blank_features=None):
        self.features, self.labels = features, labels
        self.max_regions = max_regions
        self.blank_features = blank_features

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        result = dict(features=self.features[index], label=self.labels[index])
        if self.blank_features is not None:
            result['blank_features'] = self.blank_features[index]
        return result


@torch.no_grad()
def cached_features(backbone, dataset, checkpoint_digest, settings, device, workers, split):
    if dataset.training or dataset.paired:
        raise ValueError('Feature caching requires fixed, unpaired observations')
    if any(p.requires_grad for p in backbone.parameters()):
        raise ValueError('Feature caching requires a frozen backbone')
    options = settings.get('feature_cache', {})
    memory_only = options.get('storage', 'disk') == 'memory'
    diagnostic = settings.get('blank_diagnostic', True) and split != 'train'
    path = None
    if not memory_only:
        metadata = cache_metadata(dataset, checkpoint_digest)
        metadata.update(version=2, blank_diagnostic=diagnostic)
        key = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
        directory = Path(options.get('dir', 'outputs/region_worldmodel_features'))
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f'{split}_{key}.pt'
    if path is not None and path.exists() and not options.get('rebuild', False):
        payload = torch.load(path, map_location='cpu', weights_only=True)
        if payload.get('metadata') != metadata:
            raise ValueError(f'Feature cache metadata mismatch: {path}; set feature_cache.rebuild: true')
        features, labels = payload['features'], payload['labels']
        blank_features = payload.get('blank_features')
        expected_labels = torch.tensor([label for _, label in dataset.entries])
        if (features.shape != (len(dataset), dataset.cfg['model']['embed_dim']) or
                features.dtype != torch.float32 or not torch.isfinite(features).all() or
                not torch.equal(labels, expected_labels)):
            raise ValueError(f'Invalid feature cache: {path}; set feature_cache.rebuild: true')
        if diagnostic and (blank_features is None or blank_features.shape != features.shape or
                           not torch.isfinite(blank_features).all()):
            raise ValueError(f'Invalid blank features: {path}; set feature_cache.rebuild: true')
        print(f'Features {split}: cache hit ({len(dataset)} recordings): {path}', flush=True)
        return FeatureDataset(features, labels, dataset.max_regions, blank_features)
    print(f'Features {split}: extracting {len(dataset)} recordings -> {path or "memory"}', flush=True)
    backbone.eval()
    loader = make_loader(dataset, options.get('batch_size', settings['batch_size']), workers)
    features, labels, blanks, count = [], [], [], 0
    last_report = time.monotonic()
    for raw in loader:
        view = to_device(raw['source'], device)
        features.append(backbone.features(view).float().cpu())
        if diagnostic:
            blank = dict(view, patches=torch.zeros_like(view['patches']))
            blanks.append(backbone.features(blank).float().cpu())
        labels.append(raw['label'].cpu())
        count += len(raw['label'])
        if time.monotonic() - last_report >= 20 or count == len(dataset):
            print(f'Features {split}: {count}/{len(dataset)}', flush=True)
            last_report = time.monotonic()
    features, labels = torch.cat(features), torch.cat(labels)
    blank_features = torch.cat(blanks) if blanks else None
    if not torch.isfinite(features).all():
        raise FloatingPointError('Nonfinite extracted features')
    if blank_features is not None and not torch.isfinite(blank_features).all():
        raise FloatingPointError('Nonfinite blank features')
    if memory_only:
        return FeatureDataset(features, labels, dataset.max_regions, blank_features)
    # Readers only ever see a complete file, even with concurrent probe runs.
    descriptor, temporary = tempfile.mkstemp(dir=directory, suffix='.tmp')
    os.close(descriptor)
    try:
        torch.save(dict(metadata=metadata, features=features, labels=labels,
                        blank_features=blank_features), temporary)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return FeatureDataset(features, labels, dataset.max_regions, blank_features)
