"""Saved-feature linear probing and supervised encoder finetuning."""
import argparse
import json
from itertools import islice
from pathlib import Path

import torch
from torch import nn

from experiments.region_worldmodel.train import seed_all
from .loading import make_loader, to_device
from .config import load_config, validate
from .data import HierarchyDataset, splits
from .feature_cache import cached_features
from .model import HierarchicalMAE


class Classifier(nn.Module):
    def __init__(self, backbone, classes, frozen):
        super().__init__()
        self.backbone, self.frozen = backbone, frozen
        backbone.requires_grad_(False)
        if not frozen:
            for p in backbone.encoder_parameters():
                p.requires_grad_(True)
        self.head = nn.Linear(backbone.cfg['model']['embed_dim'], classes)

    def train(self, mode=True):
        super().train(mode)
        if self.frozen:
            self.backbone.eval()
        return self

    def forward(self, view):
        return self.head(self.backbone.features(view))


def classification_epoch(model, loader, device, optimizer=None, max_batches=0):
    model.train(optimizer is not None)
    count, total, correct = 0, 0., 0
    with torch.set_grad_enabled(optimizer is not None):
        for raw in islice(loader, max_batches or None):
            if isinstance(raw, (tuple, list)):
                features, labels = [v.to(device) for v in raw]
                logits = model.head(features)
            else:
                batch = to_device(raw, device)
                labels = batch['label']
                logits = model(batch['source'])
            loss = nn.functional.cross_entropy(logits, labels)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite classification loss')
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            count += len(labels)
            total += float(loss.detach())*len(labels)
            correct += int((logits.argmax(-1) == labels).sum())
    if count == 0:
        raise ValueError('Empty classification epoch')
    return dict(loss=total/count, accuracy=correct/count, samples=count)


def run(cfg, checkpoint, mode, output_dir=None):
    validate(cfg)
    if mode not in ('linear_probe', 'finetune'):
        raise ValueError('mode must be linear_probe or finetune')
    seed_all(cfg['train']['seed'])
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    for key in ('hierarchy', 'model'):
        if cfg[key] != state['config'][key]:
            raise ValueError(f'Checkpoint {key} differs')
    for key in ('image_width', 'image_height', 'time_unit', 'time_bins', 'patch_norm', 'cstr_max_count'):
        if cfg['data'].get(key) != state['config']['data'].get(key):
            raise ValueError(f'Checkpoint data.{key} differs')
    # Preserve the pretraining validation holdout when comparing downstream runs.
    for key in ('root', 'train_split', 'val_fraction', 'split_seed', 'max_samples'):
        if cfg['data'].get(key) != state['config']['data'].get(key):
            raise ValueError(f'Changing data.{key} would change the pretraining holdout')
    device = torch.device(cfg['train']['device'])
    settings = cfg['downstream']
    backbone = HierarchicalMAE(cfg)
    backbone.load_state_dict(state['model'])
    frozen = mode == 'linear_probe'
    model = Classifier(backbone, settings['num_classes'], frozen).to(device)
    train, val, test = splits(cfg, include_test=True)
    if settings.get('max_test_samples', 0):
        test = test[:settings['max_test_samples']]
    train_ds = HierarchyDataset(cfg, train, training=not frozen)
    val_ds, test_ds = HierarchyDataset(cfg, val), HierarchyDataset(cfg, test)
    if frozen:
        train_ds = cached_features(backbone, train_ds, checkpoint, device, 'train')
        val_ds = cached_features(backbone, val_ds, checkpoint, device, 'validation')
    workers = 0 if frozen else cfg['train']['num_workers']
    def loader(ds, shuffle=False, seed=0):
        return make_loader(ds, settings['batch_size'], workers, shuffle, seed)
    train_loader = loader(train_ds, True, cfg['train']['seed'])
    val_loader = loader(val_ds)
    groups = [dict(params=model.head.parameters(), lr=settings['lr'])]
    if not frozen:
        groups.append(dict(params=backbone.encoder_parameters(), lr=settings['encoder_lr']))
    optimizer = torch.optim.AdamW(groups, weight_decay=settings['weight_decay'])
    output = Path(output_dir or Path(cfg['train']['output_dir']) / mode)
    output.mkdir(parents=True, exist_ok=True)
    best, best_epoch = float('inf'), None
    for epoch in range(settings['epochs']):
        seed_all(cfg['train']['seed']+epoch)
        if not frozen:
            train_ds.epoch = epoch
        training = classification_epoch(model, train_loader,
                                        device, optimizer, settings.get('max_batches', 0))
        devices = [device.index or 0] if device.type == 'cuda' else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(12345)
            validation = classification_epoch(model, val_loader, device)
        record = dict(epoch=epoch, train=training, validation=validation)
        print(json.dumps(record), flush=True)
        with (output / 'metrics.jsonl').open('a') as stream:
            stream.write(json.dumps(record)+'\n')
        if validation['loss'] < best:
            best, best_epoch = validation['loss'], epoch
            torch.save(dict(model=model.state_dict(), config=cfg, mode=mode, epoch=epoch,
                            validation=validation, pretrained_checkpoint=str(checkpoint)), output / 'best.pt')
    best_state = torch.load(output / 'best.pt', map_location=device, weights_only=True)
    model.load_state_dict(best_state['model'])
    if frozen:
        test_ds = cached_features(backbone, test_ds, checkpoint, device, 'test')
    torch.manual_seed(12345)
    result = dict(mode=mode, best_epoch=best_epoch, validation=best_state['validation'],
                  test=classification_epoch(model, loader(test_ds), device),
                  selection=settings.get('selection', {}), candidate_tokens=backbone.layout.count)
    (output / 'results.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--mode', choices=['linear_probe', 'finetune'], required=True)
    parser.add_argument('--output-dir')
    args = parser.parse_args()
    run(load_config(args.config), args.checkpoint, args.mode, args.output_dir)


if __name__ == '__main__':
    main()
