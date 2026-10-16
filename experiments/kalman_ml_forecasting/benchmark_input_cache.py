"""Compare uncached, cache-building and warm indexed forecasting data loads.

Example (run from the repository root):
python experiments/kalman_ml_forecasting/benchmark_input_cache.py --folders 0 1 3 \
    --samples 128 --workers 0 2 --repeats 3 --training --output-dir outputs/input_cache_benchmark
"""
from __future__ import annotations

import argparse
from contextlib import closing, redirect_stdout
from copy import deepcopy
import gc
import hashlib
import io
import json
from pathlib import Path
import platform
import sqlite3
import statistics
import sys
import time

import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.kalman_ml_forecasting.models.factory import build_model
from experiments.kalman_ml_forecasting.train import _build_dataset, _make_loader, _run_epoch
from experiments.kalman_ml_forecasting.utils.config import load_config


def consume(loader):
    digest = hashlib.sha256()
    count = 0
    start = time.perf_counter()
    for batch in loader:
        count += len(batch.frame_keys)
        digest.update(json.dumps(batch.frame_keys).encode())
        for rep, tensor in batch.inputs.items():
            digest.update(rep.encode())
            digest.update(tensor.numpy().tobytes())
        for tensor in (batch.past_boxes, batch.future_boxes, batch.past_times_s, batch.future_times_s):
            digest.update(tensor.numpy().tobytes())
    seconds = time.perf_counter() - start
    return dict(seconds=seconds, samples=count, samples_per_second=count / seconds,
                sha256=digest.hexdigest())


def timed_training(model, loader, optimizer, cfg, device):
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    with redirect_stdout(io.StringIO()):
        _run_epoch(model=model, loader=loader, device=device, optimizer=optimizer, cfg=cfg, train=True)
    torch.cuda.synchronize(device)
    return time.perf_counter() - start


def cache_stats(path):
    with closing(sqlite3.connect(path)) as connection:
        connection.execute('PRAGMA wal_checkpoint(TRUNCATE)')
        entries, compressed_bytes = connection.execute(
            'SELECT COUNT(*), SUM(LENGTH(tensor)) FROM cutouts').fetchone()
    return dict(entries=entries, compressed_tensor_bytes=compressed_bytes,
                database_bytes=path.stat().st_size)


def benchmark_case(args, original_cfg, profile, workers):
    cfg = deepcopy(original_cfg)
    data = cfg['data']
    if profile == 'events_only':
        data['representations'] = [rep for rep in data['representations']
                                   if rep.lower() not in {'rgb', 'padded_rgb', 'event_frames', 'event_frame'}]
        data['representation_sequences'] = {
            rep: spec for rep, spec in data.get('representation_sequences', {}).items()
            if rep in data['representations']}
    case_dir = args.output_dir / f'{profile}_workers{workers}'
    case_dir.mkdir()
    data.update(event_image_source='indexed', max_samples_train=args.samples,
                instance_cache={'enabled': False}, cache_dir=str(args.output_dir / 'metadata'),
                box_augmentation={'enabled': False}, decorrelation={'enabled': False})
    cfg['train'].update(num_workers=workers, batch_size=args.batch_size, persistent_workers=True,
                        pin_memory=False, log_every=1_000_000)
    (case_dir / 'config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8')
    print(f'CASE {profile}, workers={workers}', flush=True)
    datasets = {}
    loaders = {}
    for mode in ('uncached', 'cached'):
        data['instance_cache'] = {'enabled': mode == 'cached', 'dir': str(case_dir / 'cache'),
                                  'compression_level': 1}
        dataset = _build_dataset(cfg, 'train', folders_override=args.folders)
        if len(dataset) != args.samples:
            raise ValueError(f'Requested {args.samples} samples, got {len(dataset)}')
        datasets[mode] = dataset
        loaders[mode] = _make_loader(dataset, batch_size=args.batch_size, shuffle=False, train_cfg=cfg['train'])
    result = dict(profile=profile, workers=workers, samples=args.samples,
                  representations=data['representations'], first={}, steady={mode: [] for mode in loaders})
    expected_digest = None
    for repeat in range(args.repeats + 1):
        # Alternate order after the first pass to reduce timing-order bias.
        modes = ('uncached', 'cached') if repeat % 2 == 0 else ('cached', 'uncached')
        for mode in modes:
            measurement = consume(loaders[mode])
            if expected_digest is None:
                expected_digest = measurement['sha256']
            if measurement['sha256'] != expected_digest:
                raise AssertionError(f'Tensor/sample mismatch: {profile}, {mode}, pass {repeat}')
            if repeat == 0:
                result['first'][mode] = measurement
            else:
                result['steady'][mode].append(measurement)
            print(f"  {mode} pass {repeat}: {measurement['seconds']:.3f}s, "
                  f"{measurement['samples_per_second']:.1f} samples/s", flush=True)
    result['median_seconds'] = {
        mode: statistics.median(row['seconds'] for row in measurements)
        for mode, measurements in result['steady'].items()}
    result['warm_speedup'] = result['median_seconds']['uncached'] / result['median_seconds']['cached']
    result['cache'] = cache_stats(datasets['cached']._input_cache.path)

    # One actual training comparison with the base model, after CUDA and loader
    # warm-up. Uses the same epoch runner (forward/backward/metrics) as train.py.
    if args.training and profile == 'mixed' and workers == max(args.workers):
        if not torch.cuda.is_available():
            raise RuntimeError('--training requires CUDA for this benchmark')
        device = torch.device('cuda')
        torch.manual_seed(123)
        model = build_model(cfg, device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg['train'].get('lr', .001)))
        warmup_batch = next(iter(loaders['cached']))
        timed_training(model, [warmup_batch], optimizer, cfg, device)
        measurements = {mode: [] for mode in loaders}
        for repeat in range(args.repeats):
            modes = ('uncached', 'cached') if repeat % 2 == 0 else ('cached', 'uncached')
            for mode in modes:
                elapsed = timed_training(model, loaders[mode], optimizer, cfg, device)
                measurements[mode].append(elapsed)
                print(f'  training {mode} pass {repeat}: {elapsed:.3f}s', flush=True)
        medians = {mode: statistics.median(values) for mode, values in measurements.items()}
        result['training'] = dict(seconds=measurements, median_seconds=medians,
                                  optimizer={'type': 'AdamW', 'lr': optimizer.defaults['lr'],
                                             'weight_decay': optimizer.defaults['weight_decay']},
                                  warm_speedup=medians['uncached'] / medians['cached'])
        del model, optimizer, warmup_batch
        torch.cuda.empty_cache()
    (case_dir / 'results.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    # Dropping the loaders shuts down persistent workers before the next case.
    loaders.clear()
    for dataset in datasets.values():
        if dataset._input_cache is not None:
            dataset._input_cache.close()
    gc.collect()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('experiments/kalman_ml_forecasting/configs/base.yaml'))
    parser.add_argument('--folders', nargs='+', default=['0', '1', '3'])
    parser.add_argument('--samples', type=int, default=128)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--workers', nargs='+', type=int, default=[0, 2])
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--training', action='store_true')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if min(args.samples, args.batch_size, args.repeats) < 1 or min(args.workers) < 0:
        parser.error('samples, batch-size and repeats must be positive; workers must be nonnegative')
    # Refuse reuse so that the reported build pass always starts with no entries.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    cfg = load_config(args.config)
    report = dict(platform=platform.platform(), python=platform.python_version(), torch=torch.__version__,
                  cpu=platform.processor(), torch_threads=torch.get_num_threads(),
                  gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                  config=str(args.config), folders=args.folders, batch_size=args.batch_size,
                  repeats=args.repeats, notes=[
                      'Dataset construction excluded. First passes include worker startup.',
                      'Steady passes reuse persistent workers and the normal 32 MiB event cache.',
                      'OS filesystem cache is not flushed; these are not cold-disk timings.',
                      'Same deterministic sample order; all loading passes verify exact tensor hashes.',
                      'GPU training compares actual train.py epochs after one CUDA warm-up batch.',
                      'No validation, checkpointing, or model convergence measurement.'
                  ], cases=[])
    for profile in ('mixed', 'events_only'):
        for workers in args.workers:
            report['cases'].append(benchmark_case(args, cfg, profile, workers))
            (args.output_dir / 'results.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f'Results: {args.output_dir / "results.json"}', flush=True)


if __name__ == '__main__':
    main()
