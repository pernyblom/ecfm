"""Saved-feature linear probing and supervised encoder finetuning."""
import argparse
from copy import deepcopy
import json
from itertools import islice
import os
from pathlib import Path
import tempfile
import warnings

import torch
from torch import nn

from experiments.region_worldmodel.train import seed_all
from .loading import make_loader, to_device
from .config import load_config, validate
from .data import HierarchyDataset, splits
from .feature_cache import cached_features, file_digest
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


def save_checkpoint(path, state):
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, suffix='.tmp')
    os.close(descriptor)
    try:
        torch.save(state, temporary)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def validate_resume(cfg, state, mode):
    if state.get('mode') != mode:
        raise ValueError('Resume mode differs from downstream checkpoint')
    previous = state['config']
    for key in ('hierarchy', 'model'):
        if cfg[key] != previous[key]:
            raise ValueError(f'Resume {key} differs')
    for section, excluded in (('data', {'patch_cache'}),
                              ('downstream', {'epochs', 'feature_cache', 'max_test_samples'})):
        if ({k: v for k, v in cfg[section].items() if k not in excluded} !=
                {k: v for k, v in previous[section].items() if k not in excluded}):
            raise ValueError(f'Resume {section} differs; preserve training and observation settings')
    if (cfg['train']['seed'] != previous['train']['seed'] or
            cfg['data'].get('patch_cache', {}).get('train_views', 0) !=
            previous['data'].get('patch_cache', {}).get('train_views', 0)):
        raise ValueError('Resume seed or training crop bank differs')


def run(cfg, checkpoint=None, mode=None, output_dir=None, resume=None):
    validate(cfg)
    resumed = torch.load(resume, map_location='cpu', weights_only=True) if resume else None
    if resumed is not None:
        mode = mode or resumed.get('mode')
        validate_resume(cfg, resumed, mode)
        if checkpoint and resumed.get('pretrained_digest') and file_digest(checkpoint) != resumed['pretrained_digest']:
            raise ValueError('Pretrained checkpoint differs from resumed run')
        checkpoint = checkpoint or resumed.get('pretrained_checkpoint')
    if mode not in ('linear_probe', 'finetune'):
        raise ValueError('mode must be linear_probe or finetune')
    seed_all(cfg['train']['seed'])
    if resumed is None:
        if checkpoint is None:
            raise ValueError('A new downstream run requires a pretrained checkpoint')
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
        checkpoint_digest = file_digest(checkpoint)
    else:
        checkpoint_digest = resumed.get('pretrained_digest') or file_digest(
            checkpoint if checkpoint and Path(checkpoint).is_file() else resume)
    device = torch.device(cfg['train']['device'])
    settings = cfg['downstream']
    backbone = HierarchicalMAE(cfg)
    if resumed is None:
        backbone.load_state_dict(state['model'])
    frozen = mode == 'linear_probe'
    model = Classifier(backbone, settings['num_classes'], frozen).to(device)
    if resumed is not None:
        model.load_state_dict(resumed['model'])
    train, val, test = splits(cfg, include_test=True)
    if settings.get('max_test_samples', 0):
        test = test[:settings['max_test_samples']]
    train_ds = HierarchyDataset(cfg, train, training=not frozen)
    val_ds, test_ds = HierarchyDataset(cfg, val), HierarchyDataset(cfg, test)
    if frozen:
        train_ds = cached_features(backbone, train_ds, checkpoint, device, 'train', checkpoint_digest)
        val_ds = cached_features(backbone, val_ds, checkpoint, device, 'validation', checkpoint_digest)
    workers = 0 if frozen else cfg['train']['num_workers']
    def loader(ds, shuffle=False, seed=0):
        return make_loader(ds, settings['batch_size'], workers, shuffle, seed)
    train_loader = loader(train_ds, True, cfg['train']['seed'])
    val_loader = loader(val_ds)
    groups = [dict(params=model.head.parameters(), lr=settings['lr'])]
    if not frozen:
        groups.append(dict(params=backbone.encoder_parameters(), lr=settings['encoder_lr']))
    optimizer = torch.optim.AdamW(groups, weight_decay=settings['weight_decay'])
    output = Path(output_dir or (Path(resume).parent if resume else Path(cfg['train']['output_dir']) / mode))
    output.mkdir(parents=True, exist_ok=True)
    start, best, best_epoch, best_state = 0, float('inf'), None, None
    if resumed is not None:
        start = resumed['epoch']+1
        best_state = resumed.get('best_state', {k: v for k, v in resumed.items() if k != 'best_state'})
        best, best_epoch = best_state['validation']['loss'], best_state['epoch']
        if 'optimizer' in resumed:
            optimizer.load_state_dict(resumed['optimizer'])
        else:
            warnings.warn('Legacy downstream checkpoint has no optimizer state; restoring weights and epoch '
                          'with a fresh optimizer. Exact continuation is unavailable.', UserWarning)
        if 'loader_generator_state' in resumed:
            train_loader.generator.set_state(resumed['loader_generator_state'])
        # A resumed last.pt carries its best weights, even when copied to another directory.
        save_checkpoint(output / 'best.pt', best_state)
        print(f'Resuming {mode} at epoch {start}; configured total={settings["epochs"]}', flush=True)
    for epoch in range(start, settings['epochs']):
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
        current_state = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), config=cfg,
            mode=mode, epoch=epoch, validation=validation, pretrained_checkpoint=str(checkpoint),
            pretrained_digest=checkpoint_digest, loader_generator_state=train_loader.generator.get_state())
        improved = validation['loss'] < best
        if improved:
            best, best_epoch = validation['loss'], epoch
            best_state = deepcopy(current_state)
            save_checkpoint(output / 'best.pt', best_state)
        save_checkpoint(output / 'last.pt', dict(current_state, best_state=best_state))
    model.load_state_dict(best_state['model'])
    if frozen:
        test_ds = cached_features(backbone, test_ds, checkpoint, device, 'test', checkpoint_digest)
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
    parser.add_argument('--checkpoint', help='Pretrained MAE checkpoint for a new run')
    parser.add_argument('--mode', choices=['linear_probe', 'finetune'], help='Inferred from --resume when omitted')
    parser.add_argument('--resume', help='Downstream last.pt or best.pt to continue')
    parser.add_argument('--output-dir')
    args = parser.parse_args()
    if not args.resume and (not args.checkpoint or not args.mode):
        parser.error('New runs require --checkpoint and --mode; continuing runs require --resume')
    run(load_config(args.config), args.checkpoint, args.mode, args.output_dir, args.resume)


if __name__ == '__main__':
    main()
