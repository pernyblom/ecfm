"""Run from repository root: python -m experiments.region_worldmodel.train ..."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

from .config import load_config
from .data import RegionActionDataset, partition_entries
from .model import RegionWorldModel, objective, pool


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def to_device(value, device):
    if isinstance(value, dict):
        return {k: to_device(v, device) for k, v in value.items()}
    return value.to(device) if isinstance(value, torch.Tensor) else value


def make_loader(dataset, batch_size, workers=0, shuffle=False, seed=0):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=workers, drop_last=shuffle and dataset.paired,
                      generator=torch.Generator().manual_seed(seed))


def save_checkpoint(path, model, cfg, epoch, **extra):
    torch.save(dict(model=model.state_dict(), config=cfg, epoch=epoch, **extra), path)


@torch.no_grad()
def evaluate(model, loader, cfg, device):
    model.eval()
    totals, count, latents = {}, 0, []
    by_action = {a.name: [] for a in loader.dataset.actions}
    catalog = torch.tensor(np.stack([a.vector() for a in loader.dataset.actions]), device=device)
    for raw in loader:
        batch = to_device(raw, device)
        # Stable numerical sketches make checkpoint comparisons reproducible.
        dev = torch.device(device)
        devices = [dev.index if dev.index is not None else torch.cuda.current_device()] if dev.type == 'cuda' else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(12345)
            _, losses, (source, target, pred) = objective(model, batch, cfg)
        valid = batch['source']['valid_mask']
        def error(value):
            return ((value - target).square().mean(-1) * valid).sum(1) / valid.sum(1)
        errors = error(pred)
        metrics = {k: float(v) for k, v in losses.items()}
        metrics.update(prediction=float(errors.mean()), persistence=float(error(source).mean()),
                       zero_action=float(error(model.predict(source, torch.zeros_like(batch['action']))).mean()),
                       wrong_action=float(error(model.predict(source, catalog[(batch['action_id'] + 1) % len(catalog)])).mean()))
        blank = dict(batch['source'], patches=torch.zeros_like(batch['source']['patches']))
        metrics['blank_source'] = float(error(model.predict(model.encode(blank), batch['action'])).mean())
        for aid, err in zip(raw['action_id'].tolist(), errors.cpu().tolist()):
            by_action[loader.dataset.actions[aid].name].append(err)
        latents.append(pool(target, valid).cpu())
        size = len(errors)
        count += size
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0) + value * size
    result = {k: v / count for k, v in totals.items()}
    z = torch.cat(latents)
    result['latent_std'] = float(z.std(0, unbiased=False).mean())
    result['score'] = result['prediction'] + cfg['loss']['regularizer_weight'] * result['regularizer']
    result['per_action_prediction'] = {k: sum(v) / len(v) for k, v in by_action.items() if v}
    return result


def run(cfg, resume=None):
    seed_all(cfg['train']['seed'])
    t = cfg['train']
    device = torch.device(t['device'])
    output = Path(t['output_dir'])
    output.mkdir(parents=True, exist_ok=True)
    train_entries, val_entries = partition_entries(cfg)
    train_ds = RegionActionDataset(cfg, train_entries, training=True)
    val_ds = RegionActionDataset(cfg, val_entries)
    if len(train_ds) < t['batch_size']:
        raise ValueError('Training set is smaller than batch_size; reduce batch_size')
    model = RegionWorldModel(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=t['lr'], weight_decay=t['weight_decay'])
    start, best = 0, float('inf')
    if resume:
        state = torch.load(resume, map_location='cpu', weights_only=True)
        for key in ('model', 'data', 'actions', 'loss'):
            if state['config'][key] != cfg[key]:
                raise ValueError(f'Resume config differs in {key}')
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        start, best = state['epoch'] + 1, state['best']
    (output / 'config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
    (output / 'splits.json').write_text(json.dumps({
        'train': [str(p) for p, _ in train_entries],
        'validation': [str(p) for p, _ in val_entries]}, indent=2))
    val_loader = make_loader(val_ds, t['batch_size'], t['num_workers'])
    for epoch in range(start, t['epochs']):
        seed_all(t['seed'] + epoch)
        train_ds.epoch = epoch
        loader = make_loader(train_ds, t['batch_size'], t['num_workers'], True, t['seed'] + epoch)
        model.train()
        total, steps = 0.0, 0
        for i, raw in enumerate(loader):
            if t.get('max_batches', 0) and i >= t['max_batches']:
                break
            batch = to_device(raw, device)
            valid = batch['source']['valid_mask']
            mask = (torch.rand(valid.shape, device=device) < t['mask_ratio']) & valid
            optimizer.zero_grad(set_to_none=True)
            loss, _, _ = objective(model, batch, cfg, mask)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite SSL loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item()
            steps += 1
        metrics = evaluate(model, val_loader, cfg, device)
        improved = metrics['score'] < best
        best = min(best, metrics['score'])
        record = dict(epoch=epoch, train_loss=total / steps, validation=metrics)
        print(json.dumps(record), flush=True)
        with (output / 'metrics.jsonl').open('a') as file:
            file.write(json.dumps(record) + '\n')
        extra = dict(optimizer=optimizer.state_dict(), best=best, validation=metrics)
        save_checkpoint(output / 'last.pt', model, cfg, epoch, **extra)
        if improved:
            save_checkpoint(output / 'best.pt', model, cfg, epoch, **extra)
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--resume')
    args = parser.parse_args()
    run(load_config(args.config), args.resume)


if __name__ == '__main__':
    main()
