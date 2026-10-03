"""Bounded data/model benchmark; never modifies a training checkpoint."""
import argparse
import cProfile
import io
import json
from pathlib import Path
import pstats
import statistics
import time

import torch
from torch.utils.data import default_collate

from experiments.region_worldmodel.data import load_events
from experiments.region_worldmodel.train import to_device
from .config import load_config
from .data import HierarchyDataset, splits
from .model import HierarchicalMAE
from .masking import make_plan
from .loading import make_loader


def benchmark(cfg, samples=8, steps=5, batch_size=None, device=None, loader_batches=0):
    if samples < 1 or steps < 1:
        raise ValueError('samples and steps must be positive')
    device = torch.device(device or cfg['train']['device'])
    train, val, _ = splits(cfg)
    dataset = HierarchyDataset(cfg, train, training=True)
    items, times, loads, rows = [], [], [], []
    for index in range(min(samples, len(dataset))):
        path, _ = dataset.entries[index]
        start = time.perf_counter()
        events, _ = load_events(path, cfg['data']['time_unit'])
        loads.append(time.perf_counter()-start)
        n = len(events)
        del events
        start = time.perf_counter()
        items.append(dataset[index])
        times.append(time.perf_counter()-start)
        rows.append(dict(file=path.name, events=n, load_seconds=loads[-1], sample_seconds=times[-1]))
        print(json.dumps(rows[-1]), flush=True)
    cache_seconds = None
    if dataset.cache is not None and dataset.train_views:
        start = time.perf_counter()
        for index in range(len(items)):
            dataset[index]
        cache_seconds = (time.perf_counter()-start)/len(items)
    profiler = cProfile.Profile()
    profiler.runcall(dataset.__getitem__, 0)
    stream = io.StringIO()
    pstats.Stats(profiler, stream=stream).sort_stats('cumulative').print_stats(22)
    print(stream.getvalue(), flush=True)
    batch_size = batch_size or cfg['train']['batch_size']
    raw = default_collate([items[i % len(items)] for i in range(batch_size)])['source']
    view = to_device(raw, device)
    model = HierarchicalMAE(cfg).to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['train']['lr'])
    def sync():
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
    def step():
        optimizer.zero_grad(set_to_none=True)
        loss = model(view)['loss']
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
    for _ in range(2):
        step()
    sync()
    training, planning = [], []
    for _ in range(steps):
        start = time.perf_counter()
        step()
        sync()
        training.append(time.perf_counter()-start)
        start = time.perf_counter()
        make_plan(view, model.layout, cfg['masking'], cfg.get('selection'))
        sync()
        planning.append(time.perf_counter()-start)
    result = dict(device=str(device), device_name=torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU',
        torch_version=str(torch.__version__), torch_threads=torch.get_num_threads(), batch_size=batch_size,
        train_recordings=len(train), validation_recordings=len(val), samples=rows,
        mean_load_seconds=statistics.mean(loads), mean_sample_seconds=statistics.mean(times),
        estimated_serial_data_batch_seconds=statistics.mean(times)*batch_size,
        mean_training_step_seconds=statistics.mean(training), mean_mask_plan_seconds=statistics.mean(planning),
        mean_warm_cache_sample_seconds=cache_seconds,
        profile=stream.getvalue())
    if loader_batches:
        subset = HierarchyDataset(cfg, train[:loader_batches*batch_size], training=True)
        loader = make_loader(subset, batch_size, cfg['train']['num_workers'])
        result['loader_passes'] = []
        for repeat in range(2):
            waits = []
            previous = time.perf_counter()
            for _ in loader:
                ready = time.perf_counter()
                waits.append(ready-previous)
                previous = ready
            result['loader_passes'].append(dict(pass_index=repeat, workers=cfg['train']['num_workers'],
                batches=len(waits), first_batch_seconds=waits[0], total_seconds=sum(waits),
                subsequent_mean_seconds=statistics.mean(waits[1:]) if len(waits) > 1 else None))
        del loader
    print(json.dumps({k: v for k, v in result.items() if k not in ('profile', 'samples')}, indent=2), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--samples', type=int, default=8)
    parser.add_argument('--steps', type=int, default=5)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--device')
    parser.add_argument('--loader-batches', type=int, default=0,
                        help='Also time two data-only passes over this many batches, including worker startup')
    parser.add_argument('--output', default='outputs/hierarchical_mae_profile.json')
    args = parser.parse_args()
    result = benchmark(load_config(args.config), args.samples, args.steps, args.batch_size, args.device, args.loader_batches)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
