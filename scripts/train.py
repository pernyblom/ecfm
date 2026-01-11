from __future__ import annotations

import argparse
from pathlib import Path

import torch

from ecfm.training.train import build_dataloader, build_model, train_one_epoch
from ecfm.utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(Path(args.config))

    torch.manual_seed(cfg.train.seed)
    device = torch.device(cfg.train.device)

    loader = build_dataloader(cfg)
    model = build_model(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr)

    for step in range(cfg.train.num_steps):
        metrics = train_one_epoch(
            model,
            loader,
            optimizer,
            device,
            mask_ratio=cfg.model.mask_ratio,
        )
        print(f"step={step} loss={metrics['loss']:.4f}")


if __name__ == "__main__":
    main()

