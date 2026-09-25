"""Evaluate latent predictions, optionally with an unseen action catalog."""
import argparse
import json
from pathlib import Path

import torch
import yaml

from .config import validate
from .data import RegionActionDataset, partition_entries, read_entries
from .model import RegionWorldModel
from .train import evaluate, make_loader


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--actions-config', help='YAML containing an actions list; other fields are ignored')
    parser.add_argument('--split', choices=['validation', 'test'], default='validation')
    parser.add_argument('--device')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    cfg = state['config']
    if args.actions_config:
        cfg['actions'] = yaml.safe_load(Path(args.actions_config).read_text())['actions']
    validate(cfg)
    train, validation = partition_entries(cfg)
    entries = validation
    if args.split == 'test':
        entries = read_entries(Path(cfg['data']['root']), cfg['data'].get('test_split', 'test'))
        if {p for p, _ in train + validation} & {p for p, _ in entries}:
            raise ValueError('Train/validation overlap with test recordings')
    ds = RegionActionDataset(cfg, entries)
    loader = make_loader(ds, cfg['train']['batch_size'], cfg['train']['num_workers'])
    device = args.device or cfg['train']['device']
    model = RegionWorldModel(cfg).to(device)
    model.load_state_dict(state['model'])
    result = dict(checkpoint=args.checkpoint, split=args.split, actions=cfg['actions'],
                  metrics=evaluate(model, loader, cfg, device))
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
