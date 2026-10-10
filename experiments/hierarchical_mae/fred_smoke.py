"""Bounded real-FRED pretraining/checkpoint and downstream training-loop checks."""
import argparse
import json
from pathlib import Path

import torch
import yaml

from .config import load_config
from .train import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--output', type=Path, default=Path('outputs/fred_loader_smoke'))
    args = parser.parse_args()
    pretraining = load_config('experiments/hierarchical_mae/configs/fred_smoke.yaml')
    pretraining['train'].update(num_workers=args.workers, output_dir=str(args.output/'pretraining'))
    args.output.mkdir(parents=True, exist_ok=True)
    run(pretraining)
    checkpoint = args.output/'pretraining/best.pt'
    from experiments.object_detection import train as detection
    from experiments.kalman_ml_forecasting import train as forecasting
    results = {}
    for task, runner in [('object_detection', detection), ('kalman_ml_forecasting', forecasting)]:
        cfg = yaml.safe_load((Path('experiments')/task/'configs/fred_hierarchical.yaml').read_text())
        cfg['model']['backbone'].update(mae_config='experiments/hierarchical_mae/configs/fred_smoke.yaml',
                                        checkpoint=str(checkpoint), out_dim=16)
        cfg['model'].update(centernet_hidden_dim=16, topk=10, fusion_hidden_dim=16,
                            state_hidden_dim=16, residual_hidden_dim=16, history_steps=5)
        cfg['data'].update(max_samples_train=2, max_samples_val=2, image_sizes={'hierarchy': [64, 36]},
                           hierarchy_lookback_s=.033333, history_steps=4, future_steps=4)
        cfg['train'].update(device='cpu', num_workers=args.workers, batch_size=2, accumulation_steps=1,
                            log_every=1, compute_train_epoch_map=False, compute_val_epoch_map=False)
        # Two different sequences from canonical TRAIN; test recordings remain reserved.
        train = runner._build_dataset(cfg, 'train', folders_override=['13'])
        val = runner._build_dataset(cfg, 'val', folders_override=['10'])
        model = runner.build_model(cfg, torch.device('cpu'))
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        train_loader = runner._make_loader(train, batch_size=2, shuffle=False, train_cfg=cfg['train'])
        val_loader = runner._make_loader(val, batch_size=2, shuffle=False, train_cfg=cfg['train'])
        results[task] = {
            'train': runner._run_epoch(model=model, loader=train_loader, device=torch.device('cpu'),
                                      optimizer=optimizer, cfg=cfg, train=True),
            'validation': runner._run_epoch(model=model, loader=val_loader, device=torch.device('cpu'),
                                           optimizer=None, cfg=cfg, train=False),
        }
        if not all(torch.isfinite(torch.tensor(v)) for phase in results[task].values() for v in phase.values()):
            raise ValueError(f'Nonfinite metrics in {task} smoke run')
        torch.save(dict(model=model.state_dict(), config=cfg), args.output/f'{task}.pt')
        print(task, json.dumps(results[task]), flush=True)
        del train_loader, val_loader
    (args.output/'results.json').write_text(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
