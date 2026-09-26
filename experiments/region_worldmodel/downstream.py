"""THU classification with a frozen linear probe or encoder finetuning."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import tempfile

import torch
from torch import nn

from .config import load_config
from .data import RegionActionDataset, partition_entries, read_entries
from .feature_cache import cached_features, file_digest
from .model import RegionWorldModel
from .diagnostics import probe_blank_diagnostic
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
            logits = model.head(batch['features']) if 'features' in batch else model(batch['source'])
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


def run(cfg, checkpoint, mode, output_dir=None, *, evaluate_test=True, save_weights=True):
    if mode == 'finetune' and not save_weights:
        raise ValueError('Finetuning requires saving the best encoder weights')
    options = cfg['downstream'].get('feature_cache', {})
    if mode == 'linear_probe' and options.get('enabled', True) and options.get('storage') == 'temporary':
        with tempfile.TemporaryDirectory(prefix='region_probe_') as directory:
            recorded_cfg = cfg
            cfg = deepcopy(cfg)
            cfg['downstream']['feature_cache'].update(storage='disk', dir=directory)
            return _run(cfg, checkpoint, mode, output_dir, evaluate_test, save_weights, recorded_cfg)
    return _run(cfg, checkpoint, mode, output_dir, evaluate_test, save_weights)


def _run(cfg, checkpoint, mode, output_dir, evaluate_test, save_weights, recorded_cfg=None):
    seed_all(cfg['train']['seed'])
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    # Enforce identical feature semantics, including plane ID ordering and
    # timestamp/patch normalization. Split/subset limits may change downstream.
    if cfg['model'] != state['config']['model']:
        raise ValueError('Model config differs from pretrained checkpoint')
    if cfg['data'].get('include_absolute_duration', True) != state['config']['data'].get('include_absolute_duration', True):
        raise ValueError('include_absolute_duration differs from pretrained checkpoint')
    for key in ('image_width', 'image_height', 'time_unit', 'time_bins',
                'plane_types', 'patch_norm'):
        if cfg['data'].get(key) != state['config']['data'].get(key):
            raise ValueError(f'Data feature config differs in {key}')
    # Keep unused absolute embedding shapes compatible with the checkpoint;
    # relative attention supports downstream sequences of a different length.
    backbone = RegionWorldModel(state['config'])
    backbone.load_state_dict(state['model'])
    settings = cfg['downstream']
    device = torch.device(cfg['train']['device'])
    model = Classifier(backbone, settings['num_classes'], mode == 'linear_probe').to(device)
    train_entries, val_entries = partition_entries(cfg)
    test_entries = read_entries(Path(cfg['data']['root']), cfg['data'].get('test_split', 'test')) if evaluate_test else []
    if {p for p, _ in train_entries + val_entries} & {p for p, _ in test_entries}:
        raise ValueError('Train/validation overlap with test recordings')
    if settings.get('max_test_samples', 0):
        test_entries = test_entries[:settings['max_test_samples']]
    for _, label in train_entries + val_entries + test_entries:
        if not 0 <= label < settings['num_classes']:
            raise ValueError('Labels must be zero-based class indices within num_classes')
    # Explicit grid/multiscale layouts are fixed even during finetuning.
    train_ds = RegionActionDataset(cfg, train_entries, training=mode == 'finetune', paired=False)
    val_ds = RegionActionDataset(cfg, val_entries, paired=False)
    test_ds = RegionActionDataset(cfg, test_entries, paired=False)
    workers = cfg['train']['num_workers']
    use_cache = mode == 'linear_probe' and settings.get('feature_cache', {}).get('enabled', True)
    if use_cache:
        checkpoint_digest = file_digest(checkpoint) if settings.get('feature_cache', {}).get('storage', 'disk') != 'memory' else ''
        train_ds = cached_features(backbone, train_ds, checkpoint_digest, settings, device, workers, 'train')
        val_ds = cached_features(backbone, val_ds, checkpoint_digest, settings, device, workers, 'validation')
    # In-memory tensors do not benefit from Windows worker process startup.
    probe_workers = 0 if use_cache else workers
    val_loader = make_loader(val_ds, settings['batch_size'], probe_workers)
    groups = [{'params': model.head.parameters(), 'lr': settings['lr']}]
    if mode == 'finetune':
        groups.append({'params': [p for p in backbone.parameters() if p.requires_grad],
                       'lr': settings['encoder_lr']})
    optimizer = torch.optim.AdamW(groups, weight_decay=settings['weight_decay'])
    region_spec = settings.get('regions', {})
    layout_name = region_spec.get('mode', 'random')
    run_name = mode if not region_spec else f'{mode}_{layout_name}_{train_ds.max_regions}'
    output = Path(output_dir or Path(cfg['train']['output_dir']) / run_name)
    output.mkdir(parents=True, exist_ok=True)
    best = float('inf')
    best_head, best_validation, best_epoch = None, None, None
    for epoch in range(settings['epochs']):
        train_ds.epoch = epoch
        loader = make_loader(train_ds, settings['batch_size'], probe_workers, True, cfg['train']['seed'] + epoch)
        training = classification_epoch(model, loader, device, optimizer, settings.get('max_batches', 0))
        validation = classification_epoch(model, val_loader, device)
        print(json.dumps(dict(epoch=epoch, train=training, validation=validation)), flush=True)
        with (output / 'metrics.jsonl').open('a') as file:
            file.write(json.dumps(dict(epoch=epoch, train=training, validation=validation)) + '\n')
        if validation['loss'] < best:
            best = validation['loss']
            best_head = deepcopy(model.head.state_dict())
            best_validation, best_epoch = validation, epoch
            if save_weights:
                torch.save(dict(model=model.state_dict(), config=recorded_cfg or cfg, mode=mode,
                            backbone_config=state['config'],
                            epoch=epoch, validation=validation,
                            pretrained_epoch=state.get('epoch'),
                            pretrained_checkpoint=str(checkpoint)), output / 'best.pt')
    if save_weights:
        best_state = torch.load(output / 'best.pt', map_location=device, weights_only=True)
        model.load_state_dict(best_state['model'])
    else:
        model.head.load_state_dict(best_head)
    result = dict(mode=mode, checkpoint=str(checkpoint), best_epoch=best_epoch,
                  pretrained_epoch=state.get('epoch'),
                  regions=region_spec, max_region_tokens=train_ds.max_regions,
                  validation=best_validation)
    if settings.get('blank_diagnostic', True):
        result['blank_validation'] = probe_blank_diagnostic(model, val_loader, device)
    if evaluate_test:
        if use_cache:
            test_ds = cached_features(backbone, test_ds, checkpoint_digest, settings, device, workers, 'test')
        test_loader = make_loader(test_ds, settings['batch_size'], probe_workers)
        result['test'] = classification_epoch(model, test_loader, device)
        if settings.get('blank_diagnostic', True):
            result['blank_test'] = probe_blank_diagnostic(model, test_loader, device)
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
