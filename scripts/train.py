from __future__ import annotations

import argparse
from pathlib import Path

import torch

from ecfm.training.train import (
    build_dataloader,
    build_model,
    dump_reconstructions,
    train_one_epoch,
)
from ecfm.utils.config import load_config
from ecfm.utils.masking import random_mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(Path(args.config))

    torch.manual_seed(cfg.train.seed)
    if cfg.train.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA requested but not available, falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(cfg.train.device)

    loader = build_dataloader(cfg)
    model = build_model(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr)

    ema_loss = None
    ema_alpha = 0.9

    for step in range(cfg.train.num_steps):
        metrics = train_one_epoch(
            model,
            loader,
            optimizer,
            device,
            mask_ratio=cfg.model.mask_ratio,
        )
        loss = metrics["loss"]
        if ema_loss is None:
            ema_loss = loss
        else:
            ema_loss = ema_alpha * ema_loss + (1.0 - ema_alpha) * loss
        mem_msg = ""
        if device.type == "cuda":
            allocated = torch.cuda.max_memory_allocated(device=device) / (1024**2)
            reserved = torch.cuda.max_memory_reserved(device=device) / (1024**2)
            mem_msg = f" max_mem_mb={allocated:.1f} reserved_mb={reserved:.1f}"
        print(f"step={step} loss={loss:.4f} ema_loss={ema_loss:.4f}{mem_msg}")

        if cfg.train.recon_every > 0 and step % cfg.train.recon_every == 0:
            model.eval()
            batch = next(iter(loader))
            patches = batch["patches"].to(device)
            metadata = batch["metadata"].to(device)
            plane_ids = batch["plane_ids"].to(device)
            mask_list = []
            for _ in range(patches.shape[0]):
                mask_list.append(
                    random_mask(patches.shape[1], cfg.model.mask_ratio).to(device)
                )
            mask = torch.stack(mask_list, dim=0)
            with torch.no_grad():
                pred_patches, _, _ = model(patches, metadata, plane_ids, mask=mask)
            dump_reconstructions(
                step=step,
                patches=patches,
                pred_patches=pred_patches,
                mask=mask,
                plane_ids=plane_ids,
                plane_types=cfg.data.plane_types,
                out_dir=cfg.train.recon_out_dir,
                num_patches=cfg.train.recon_num_patches,
                upscale=cfg.train.recon_upscale,
            )


if __name__ == "__main__":
    main()
