"""THU classification with a frozen linear probe or encoder finetuning."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn

from .config import load_config
from .data import RegionActionDataset, partition_entries, read_entries
from .model import RegionWorldModel
from .train import make_loader, seed_all, to_device


class Classifier(nn.Module):
    def __init__(self, backbone, num_classes, frozen):
        super().__init__()
        self.backbone, self.frozen = backbone, frozen
        # Predictor/decoder have no role in classification.
        backbone.requires_grad_(False)
        if not frozen:
            for name, parameter in backbone.encoder.named_parameters():
                if not name.startswith(('decoder', 'patch_decoder', 'count_decoder')):
                    parameter.requires_grad_(True)
        self.head = nn.Linear(backbone.predictor[-1].out_features, num_classes)

    def train(self, mode=True):
        super().train(mode)
        if self.frozen:
            self.backbone.eval()
        return self

    def forward(self, view):
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.frozen):
            features = self.backbone.features(view)
        return self.head(features)


def classification_epoch(model, loader, device, optimizer=None, max_batches=0):
    model.train(optimizer is not None)
    total, correct, count = 0.0, 0, 0
    with torch.set_grad_enabled(optimizer is not None):
        for i, raw in enumerate(loader):
            if max_batches and i >= max_batches:
                break
            batch = to_device(raw, device)
            logits = model(batch['source'])
            loss = nn.functional.cross_entropy(logits, batch['label'])
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite classification loss')
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            size = len(logits)
            total += loss.item() * size
            correct += int((logits.argmax(-1) == batch['label']).sum())
            count += size
    return dict(loss=total / count, accuracy=correct / count, samples=count)


def run(cfg, checkpoint, mode, output_dir=None):
    seed_all(cfg['train']['seed'])
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    # Enforce identical feature semantics, including plane ID ordering and
    # timestamp/patch normalization. Split/subset limits may change downstream.
    if cfg['model'] != state['config']['model']:
        raise ValueError('Model config differs from pretrained checkpoint')
    for key in ('image_width', 'image_height', 'time_unit', 'time_bins',
                'plane_types', 'num_regions_choices', 'patch_norm'):
        if cfg['data'].get(key) != state['config']['data'].get(key):
            raise ValueError(f'Data feature config differs in {key}')
    backbone = RegionWorldModel(cfg)
    backbone.load_state_dict(state['model'])
    settings = cfg['downstream']
    device = torch.device(cfg['train']['device'])
    model = Classifier(backbone, settings['num_classes'], mode == 'linear_probe').to(device)
    train_entries, val_entries = partition_entries(cfg)
    test_entries = read_entries(Path(cfg['data']['root']), cfg['data'].get('test_split', 'test'))
    if {p for p, _ in train_entries + val_entries} & {p for p, _ in test_entries}:
        raise ValueError('Train/validation overlap with test recordings')
    if settings.get('max_test_samples', 0):
        test_entries = test_entries[:settings['max_test_samples']]
    for _, label in train_entries + val_entries + test_entries:
        if not 0 <= label < settings['num_classes']:
            raise ValueError('Labels must be zero-based class indices within num_classes')
    # Fixed regions for the probe keep the feature extraction protocol stable.
    train_ds = RegionActionDataset(cfg, train_entries, training=mode == 'finetune', paired=False)
    val_ds = RegionActionDataset(cfg, val_entries, paired=False)
    test_ds = RegionActionDataset(cfg, test_entries, paired=False)
    workers = cfg['train']['num_workers']
    val_loader = make_loader(val_ds, settings['batch_size'], workers)
    test_loader = make_loader(test_ds, settings['batch_size'], workers)
    groups = [{'params': model.head.parameters(), 'lr': settings['lr']}]
    if mode == 'finetune':
        groups.append({'params': [p for p in backbone.parameters() if p.requires_grad],
                       'lr': settings['encoder_lr']})
    optimizer = torch.optim.AdamW(groups, weight_decay=settings['weight_decay'])
    output = Path(output_dir or Path(cfg['train']['output_dir']) / mode)
    output.mkdir(parents=True, exist_ok=True)
    best = float('inf')
    for epoch in range(settings['epochs']):
        train_ds.epoch = epoch
        loader = make_loader(train_ds, settings['batch_size'], workers, True, cfg['train']['seed'] + epoch)
        training = classification_epoch(model, loader, device, optimizer, settings.get('max_batches', 0))
        validation = classification_epoch(model, val_loader, device)
        print(json.dumps(dict(epoch=epoch, train=training, validation=validation)), flush=True)
        with (output / 'metrics.jsonl').open('a') as file:
            file.write(json.dumps(dict(epoch=epoch, train=training, validation=validation)) + '\n')
        if validation['loss'] < best:
            best = validation['loss']
            torch.save(dict(model=model.state_dict(), config=cfg, mode=mode,
                            epoch=epoch, validation=validation,
                            pretrained_checkpoint=str(checkpoint)), output / 'best.pt')
    best_state = torch.load(output / 'best.pt', map_location=device, weights_only=True)
    model.load_state_dict(best_state['model'])
    test = classification_epoch(model, test_loader, device)
    result = dict(mode=mode, checkpoint=str(checkpoint), best_epoch=best_state['epoch'],
                  validation=best_state['validation'], test=test)
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
