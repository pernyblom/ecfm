"""Pretrain a hierarchical event MAE on random temporal subvolumes."""
import argparse
import json
from itertools import islice
from pathlib import Path
import time

import torch
import yaml

from experiments.region_worldmodel.train import seed_all
from .loading import make_loader, to_device
from .config import load_config, validate
from .data import HierarchyDataset, splits
from .inspect import save_inspection
from .model import HierarchicalMAE


def epoch_pass(model, loader, device, optimizer=None, max_batches=0, log_every=0):
    model.train(optimizer is not None)
    totals, count = {}, 0
    start = previous = time.perf_counter()
    data_seconds, step_seconds, batches = 0., 0., 0
    with torch.set_grad_enabled(optimizer is not None):
        for i, raw in enumerate(islice(loader, max_batches or None)):
            ready = time.perf_counter()
            data_seconds += ready-previous
            view = to_device(raw['source'], device)
            result = model(view)
            if not torch.isfinite(result['loss']):
                raise FloatingPointError('Nonfinite MAE loss')
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                result['loss'].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
            size = len(view['metadata'])
            count += size
            for key in ('loss', 'patch_loss', 'count_loss', 'visible_tokens'):
                totals[key] = totals.get(key, 0.) + float(result[key].detach())*size
            # Scalar metric reads above synchronize CUDA, so this includes completed GPU work.
            previous = time.perf_counter()
            step_seconds += previous-ready
            batches += 1
            if log_every and batches % log_every == 0:
                print(json.dumps(dict(phase='train' if optimizer is not None else 'validation', batch=batches,
                    loss=totals['loss']/count, data_wait_seconds_per_batch=data_seconds/batches,
                    step_seconds_per_batch=step_seconds/batches, samples_per_second=count/(previous-start))), flush=True)
    if not count:
        raise ValueError('Empty epoch')
    return dict({key: value/count for key, value in totals.items()},
                data_wait_seconds_per_batch=data_seconds/batches,
                step_seconds_per_batch=step_seconds/batches, samples_per_second=count/(previous-start))


def run(cfg, resume=None):
    validate(cfg)
    t = cfg['train']
    seed_all(t['seed'])
    device = torch.device(t['device'])
    output = Path(t['output_dir'])
    output.mkdir(parents=True, exist_ok=True)
    train_entries, val_entries, _ = splits(cfg)
    train_ds, val_ds = HierarchyDataset(cfg, train_entries, True), HierarchyDataset(cfg, val_entries)
    train_loader = make_loader(train_ds, t['batch_size'], t['num_workers'], True, t['seed'])
    val_loader = make_loader(val_ds, t['batch_size'], t['num_workers'])
    model = HierarchicalMAE(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=t['lr'], weight_decay=t['weight_decay'])
    start, best = 0, float('inf')
    if resume:
        state = torch.load(resume, map_location=device, weights_only=True)
        for key in ('data', 'hierarchy', 'model', 'masking', 'selection', 'loss'):
            previous, current = state['config'].get(key), cfg.get(key)
            if key == 'data':
                # Runtime caching is safe to enable on existing checkpoints. train_views
                # explicitly opts into a bounded crop schedule for subsequent epochs.
                previous = {k: v for k, v in previous.items() if k != 'patch_cache'}
                current = {k: v for k, v in current.items() if k != 'patch_cache'}
            if previous != current:
                raise ValueError(f'Resume config differs in {key}')
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        start, best = state['epoch']+1, state['best']
    (output / 'config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
    (output / 'splits.json').write_text(json.dumps(dict(train=[str(p) for p, _ in train_entries],
                                                       validation=[str(p) for p, _ in val_entries]), indent=2))
    print(f'{model.layout.count} candidate tokens; {sum(p.numel() for p in model.parameters()):,} parameters', flush=True)
    print(f'Data workers={t["num_workers"]}; training crop bank={train_ds.train_views or "fresh crops"}; '
          f'patch cache={cfg["data"].get("patch_cache", {}).get("enabled", False)}', flush=True)
    for epoch in range(start, t['epochs']):
        seed_all(t['seed']+epoch)
        train_ds.epoch = epoch
        training = epoch_pass(model, train_loader, device, optimizer, t.get('max_batches', 0), t.get('log_every', 0))
        # Validation uses fixed crops AND masks without perturbing training randomness.
        devices = [device.index or 0] if device.type == 'cuda' else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(12345)
            validation = epoch_pass(model, val_loader, device, max_batches=t.get('max_val_batches', 0), log_every=t.get('log_every', 0))
            every = t.get('inspect_every', 0)
            if every and (epoch+1) % every == 0:
                with torch.no_grad():
                    view = to_device(next(iter(val_loader))['source'], device)
                    save_inspection(output, epoch+1, view, model(view), model.layout, t.get('inspect_per_group', 4))
        improved = validation['loss'] < best
        best = min(best, validation['loss'])
        state = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), config=cfg,
                     epoch=epoch, best=best, validation=validation)
        torch.save(state, output / 'last.pt')
        if improved:
            torch.save(state, output / 'best.pt')
        record = dict(epoch=epoch, train=training, validation=validation)
        print(json.dumps(record), flush=True)
        with (output / 'metrics.jsonl').open('a') as stream:
            stream.write(json.dumps(record)+'\n')
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--resume')
    args = parser.parse_args()
    run(load_config(args.config), args.resume)


if __name__ == '__main__':
    main()
