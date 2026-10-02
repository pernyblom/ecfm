"""Pretrain a hierarchical event MAE on random temporal subvolumes."""
import argparse
import json
from pathlib import Path

import torch
import yaml

from experiments.region_worldmodel.train import make_loader, seed_all, to_device
from .config import load_config, validate
from .data import HierarchyDataset, splits
from .inspect import save_inspection
from .model import HierarchicalMAE


def epoch_pass(model, loader, device, optimizer=None, max_batches=0):
    model.train(optimizer is not None)
    totals, count = {}, 0
    with torch.set_grad_enabled(optimizer is not None):
        for i, raw in enumerate(loader):
            if max_batches and i >= max_batches:
                break
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
    if not count:
        raise ValueError('Empty epoch')
    return {key: value/count for key, value in totals.items()}


def run(cfg, resume=None):
    validate(cfg)
    t = cfg['train']
    seed_all(t['seed'])
    device = torch.device(t['device'])
    output = Path(t['output_dir'])
    output.mkdir(parents=True, exist_ok=True)
    train_entries, val_entries, _ = splits(cfg)
    train_ds, val_ds = HierarchyDataset(cfg, train_entries, True), HierarchyDataset(cfg, val_entries)
    val_loader = make_loader(val_ds, t['batch_size'], t['num_workers'])
    model = HierarchicalMAE(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=t['lr'], weight_decay=t['weight_decay'])
    start, best = 0, float('inf')
    if resume:
        state = torch.load(resume, map_location=device, weights_only=True)
        for key in ('data', 'hierarchy', 'model', 'masking', 'selection', 'loss'):
            if state['config'].get(key) != cfg.get(key):
                raise ValueError(f'Resume config differs in {key}')
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        start, best = state['epoch']+1, state['best']
    (output / 'config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
    (output / 'splits.json').write_text(json.dumps(dict(train=[str(p) for p, _ in train_entries],
                                                       validation=[str(p) for p, _ in val_entries]), indent=2))
    print(f'{model.layout.count} candidate tokens; {sum(p.numel() for p in model.parameters()):,} parameters', flush=True)
    for epoch in range(start, t['epochs']):
        seed_all(t['seed']+epoch)
        train_ds.epoch = epoch
        loader = make_loader(train_ds, t['batch_size'], t['num_workers'], True, t['seed']+epoch)
        training = epoch_pass(model, loader, device, optimizer, t.get('max_batches', 0))
        # Validation uses fixed crops AND masks without perturbing training randomness.
        devices = [device.index or 0] if device.type == 'cuda' else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(12345)
            validation = epoch_pass(model, val_loader, device, max_batches=t.get('max_val_batches', 0))
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
