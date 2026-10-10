"""Optionally populate reusable patch crops before training; safe to interrupt/resume."""
import argparse
from copy import deepcopy
import json
import time

from .config import load_config
from .data import make_dataset, splits
from .loading import make_loader


def prepare(cfg, split='all', workers=None):
    cfg = deepcopy(cfg)
    if not cfg['data'].get('patch_cache', {}).get('enabled', False):
        raise ValueError('Enable data.patch_cache before preparing patches')
    views = cfg['data']['patch_cache'].get('train_views', 0)
    if split in ('train', 'all') and not views:
        raise ValueError('Training patch caching requires data.patch_cache.train_views > 0')
    cfg['train']['pin_memory'] = False
    train, val, _ = splits(cfg)
    workers = cfg['train']['num_workers'] if workers is None else workers
    for name, entries, training in (('train', train, True), ('validation', val, False)):
        if split not in ('all', name):
            continue
        dataset = make_dataset(cfg, entries, training)
        loader = make_loader(dataset, 1, workers)
        for epoch in range(views if training else 1):
            dataset.epoch = epoch
            start = time.perf_counter()
            for count, _ in enumerate(loader, 1):
                if count % 100 == 0 or count == len(dataset):
                    print(json.dumps(dict(split=name, crop=epoch, recordings=count, total=len(dataset),
                                          seconds=time.perf_counter()-start)), flush=True)
        del loader


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--split', choices=['train', 'validation', 'all'], default='all')
    parser.add_argument('--workers', type=int)
    args = parser.parse_args()
    prepare(load_config(args.config), args.split, args.workers)


if __name__ == '__main__':
    main()
