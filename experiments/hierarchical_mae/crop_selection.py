"""Reproducible crop targets for set selection, plus held-out crop evaluation."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from .data import HierarchyDataset, splits
from .config import load_config, validate
from .downstream import save_checkpoint
from .feature_cache import cache_metadata, file_digest
from .loading import make_loader, to_device
from . import set_selector, swap_oracle
from .set_selector import coarse_inputs, load_policy, selected_features
from .swap_oracle import search_space, token_sets


def with_crop_ranges(cfg, temporal=None, spatial=None):
    """Override only training crop ranges, preserving full-view provenance."""
    result = deepcopy(cfg)
    if temporal is not None:
        result['data']['crop_fraction'] = list(temporal)
    if spatial is not None:
        result['data']['spatial_crop_fraction'] = list(spatial)
    validate(result)
    return result


def training_crop_config(cfg, args):
    temporal = getattr(args, 'temporal_crop', None)
    spatial = getattr(args, 'spatial_crop', None)
    path = getattr(args, 'crop_config', None)
    if path:
        if temporal is not None or spatial is not None:
            raise ValueError('Use --crop-config or explicit crop ranges, not both')
        data = load_config(path)['data']
        temporal, spatial = data['crop_fraction'], data.get('spatial_crop_fraction', [1., 1.])
    return with_crop_ranges(cfg, temporal, spatial)


def crop_dataset(cfg, entries, view_index):
    """One reproducible random spatial/temporal crop per recording and view ID.

    Do not retain rendered crop patches: only compact inputs/targets are cached.
    EpochSampler propagates view_index to Windows persistent worker processes.
    """
    options = deepcopy(cfg)
    options['data']['patch_cache'] = dict(options['data'].get('patch_cache', {}), enabled=False, train_views=0)
    dataset = HierarchyDataset(options, entries, training=True)
    dataset.epoch = view_index
    return dataset


def epoch_indices(records, views, epoch, seed, device):
    """One view per recording per epoch: match fixed-view optimizer step counts.

    Banks are view-major, with the unchanged full recording at view zero. Each
    recording cycles through all views, starting at a seeded random offset.
    """
    if records < 1 or views < 1 or epoch < 0:
        raise ValueError('Require positive records/views and a nonnegative epoch')
    indices = torch.arange(records, device=device)
    if views > 1:
        offset = torch.randint(views, (records,), generator=torch.Generator().manual_seed(seed)).to(device)
        indices = ((offset+epoch) % views)*records + indices
    return indices[torch.randperm(records, device=device)]


@torch.no_grad()
def crop_bank(model, entries, cfg, oracle, args, index):
    options = oracle['arguments']
    states, _ = search_space(options['drops'], options['adds'], options['max_swaps'])
    dataset = crop_dataset(cfg, entries, index)
    metadata = dict(base=cache_metadata(dataset, args.baseline), view_index=index,
                    pool={k: options[k] for k in ('budget', 'drops', 'adds', 'max_swaps')},
                    implementation=[file_digest(__file__), file_digest(set_selector.__file__),
                                    file_digest(swap_oracle.__file__)])
    key = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()[:20]
    directory = Path(args.crop_cache_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory/f'view_{index}_{key}.pt'
    if path.exists():
        saved = torch.load(path, map_location='cpu', weights_only=True)
        if saved['metadata'] != metadata:
            raise ValueError('Crop cache provenance differs')
        print(f'Crop view {index}: cache hit {path}', flush=True)
        return saved['data'], path
    chunks = {k: [] for k in ('roots', 'descriptors', 'labels', 'losses', 'predictions')}
    loader = make_loader(dataset, 8, cfg['train']['num_workers'])
    started = time.monotonic()
    for step, raw in enumerate(loader):
        view = to_device(raw['source'], args.device)
        roots, descriptors = coarse_inputs(model.backbone, view, options['budget'], options['drops'], options['adds'])
        ids, _, _, _ = token_sets(view['log_counts'], view['valid_mask'], options['budget'],
                                  options['drops'], options['adds'], states)
        embeddings = model.backbone.token_embeddings(view)
        flat = ids.flatten(0, 1)
        samples = torch.arange(len(ids), device=args.device).repeat_interleave(len(states))
        logits = []
        for offset in range(0, len(flat), 64):
            packed = embeddings[samples[offset:offset+64, None], flat[offset:offset+64]]
            padding = torch.zeros(packed.shape[:2], dtype=torch.bool, device=args.device)
            logits.append(model.head(model.backbone.encoder(packed, src_key_padding_mask=padding).mean(1)))
        logits = torch.cat(logits).reshape(len(ids), len(states), -1)
        labels = raw['label'].to(args.device)
        losses = F.cross_entropy(logits.flatten(0, 1), labels[:, None].expand(-1, len(states)).flatten(),
                                 reduction='none').reshape(len(ids), len(states))
        for key, value in zip(chunks, (roots, descriptors, labels, losses, logits.argmax(-1))):
            chunks[key].append(value.cpu())
        if step % 10 == 0 or step+1 == len(loader):
            print(f'Crop view {index}: {min((step+1)*8, len(dataset))}/{len(dataset)}, '
                  f'{time.monotonic()-started:.1f}s', flush=True)
    data = {key: torch.cat(values) for key, values in chunks.items()}
    expected_labels = torch.tensor([label for _, label in entries])
    if not torch.equal(data['labels'], expected_labels) or any(not torch.isfinite(v).all() for v in data.values()):
        raise ValueError('Invalid cropped target bank')
    save_checkpoint(path, dict(metadata=metadata, data=data, seconds=time.monotonic()-started))
    return data, path


def augmented_inputs(model, entries, cfg, oracle, args, full):
    cfg = training_crop_config(cfg, args)
    banks, files = [full], []
    for index in range(args.crop_views):
        bank, path = crop_bank(model, entries, cfg, oracle, args, index)
        banks.append(bank)
        files.append(dict(path=str(path.resolve()), digest=file_digest(path)))
    result = {key: torch.cat([bank[key] for bank in banks]) for key in full}
    result['records'] = len(entries)
    result['views'] = len(banks)
    args.augmentation = dict(crop_views=args.crop_views, include_full_view=True, views_per_epoch=1,
        records=len(entries), temporal=cfg['data']['crop_fraction'],
        spatial=cfg['data'].get('spatial_crop_fraction', [1., 1.]),
        region_seed=cfg['data'].get('region_seed', 123), banks=files)
    print(f'Augmentation: {len(entries)} recordings x {len(banks)} views; '
          'one view per recording each epoch', flush=True)
    return result


@torch.no_grad()
def evaluate_crops(args):
    """Fixed policies, same unseen test crops, original activity head for all."""
    torch.set_num_threads(4)
    output = Path(args.output)
    if output.exists():
        raise ValueError('Evaluation output exists')
    output.parent.mkdir(parents=True, exist_ok=True)
    models, hashes, augmented_ranges = {}, {}, []
    for kind, directory in [('fixed', args.fixed_dir), ('augmented', args.augmented_dir)]:
        for limit in args.limits:
            path = Path(directory)/f'policy_swaps{limit}.pt'
            saved = torch.load(path, map_location='cpu', weights_only=True)
            hashes[f'{kind}_{limit}'] = dict(policy=file_digest(path), baseline=saved['baseline_digest'])
            models[f'{kind}_{limit}'] = load_policy(path, device=args.device)
            if kind == 'augmented':
                augmentation = saved.get('augmentation') or {}
                augmented_ranges.append((augmentation.get('temporal'), augmentation.get('spatial')))
    if len({entry['baseline'] for entry in hashes.values()}) != 1:
        raise ValueError('Policies must share one baseline')
    first = next(iter(models.values()))
    if any(ranges != augmented_ranges[0] for ranges in augmented_ranges):
        raise ValueError('Augmented policies use different crop ranges; evaluate separately')
    temporal, spatial = augmented_ranges[0]
    cfg = with_crop_ranges(first.classifier.backbone.cfg,
                           args.temporal_crop if args.temporal_crop is not None else temporal,
                           args.spatial_crop if args.spatial_crop is not None else spatial)
    entries = dict(zip(('train', 'validation', 'test'), splits(cfg, include_test=True)))[args.split]
    totals = {name: dict(loss=0., correct=0, swaps=0., samples=0) for name in ('activity', *models)}
    predictions = {name: [] for name in totals}
    labels_all = []
    for index in range(args.views):
        dataset = crop_dataset(cfg, entries, args.first_view+index)
        for step, raw in enumerate(make_loader(dataset, 16, cfg['train']['num_workers'])):
            view, labels = to_device(raw['source'], args.device), raw['label'].to(args.device)
            logits = first.classifier(view)
            values = {'activity': (logits, torch.zeros(len(labels), device=args.device))}
            for name, model in models.items():
                features, choices = selected_features(model.classifier.backbone, view, model.policy,
                                                       model.threshold, model.budget)
                values[name] = model.classifier.head(features), model.policy.depths[choices]
            labels_all.append(labels.cpu())
            for name, (logits, depths) in values.items():
                totals[name]['loss'] += F.cross_entropy(logits, labels, reduction='sum').item()
                totals[name]['correct'] += int((logits.argmax(1) == labels).sum())
                totals[name]['swaps'] += depths.sum().item()
                totals[name]['samples'] += len(labels)
                predictions[name].append(logits.argmax(1).cpu())
            if step % 10 == 0:
                print(f'Unseen {args.split} crop {index}: {min((step+1)*16, len(dataset))}/{len(dataset)}', flush=True)
    result = dict(arguments=vars(args), checkpoints=hashes, evaluation_ranges=dict(
        temporal=cfg['data']['crop_fraction'], spatial=cfg['data']['spatial_crop_fraction']), metrics={})
    for name, totals in totals.items():
        n = totals['samples']
        result['metrics'][name] = dict(loss=totals['loss']/n, accuracy=totals['correct']/n,
                                     mean_swaps=totals['swaps']/n, samples=n)
    save_checkpoint(output.with_suffix('.pt'), dict(labels=torch.cat(labels_all),
        predictions={name: torch.cat(values) for name, values in predictions.items()}))
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate fixed and augmented policies on identical unseen crops')
    parser.add_argument('--fixed-dir', required=True)
    parser.add_argument('--augmented-dir', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--limits', type=int, nargs='+', default=[1, 2, 4])
    parser.add_argument('--split', choices=['validation', 'test'], default='test')
    parser.add_argument('--views', type=int, default=2)
    parser.add_argument('--first-view', type=int, default=10000)
    parser.add_argument('--temporal-crop', type=float, nargs=2, metavar=('MIN', 'MAX'),
                        help='Evaluation range; defaults to the augmented policy training range')
    parser.add_argument('--spatial-crop', type=float, nargs=2, metavar=('MIN', 'MAX'),
                        help='Evaluation width/height range; defaults to the augmented policy training range')
    args = parser.parse_args()
    if args.views < 1 or args.first_view < 0:
        parser.error('Require positive views and nonnegative first-view')
    evaluate_crops(args)
