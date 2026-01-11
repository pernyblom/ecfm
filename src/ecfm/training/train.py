from __future__ import annotations

from typing import Dict

import torch
from torch import nn
from torch.utils.data import DataLoader

from ecfm.data.event_dataset import SyntheticEventDataset, THUEACTDataset
from ecfm.models.mae import EventMAE
from ecfm.utils.config import Config
from ecfm.utils.masking import random_mask


def build_dataloader(cfg: Config) -> DataLoader:
    if cfg.data.dataset_name == "synthetic":
        dataset = SyntheticEventDataset(
            num_samples=cfg.data.num_samples,
            max_events=cfg.data.max_events,
            image_width=cfg.data.image_width,
            image_height=cfg.data.image_height,
            time_bins=cfg.data.time_bins,
            region_scales=cfg.data.region_scales,
            plane_types=cfg.data.plane_types,
            num_regions=cfg.data.num_regions,
            patch_size=cfg.model.patch_size,
            rng_seed=cfg.train.seed,
        )
    elif cfg.data.dataset_name == "thu-eact":
        dataset = THUEACTDataset(
            root=cfg.data.dataset_path,
            split=cfg.data.split,
            image_width=cfg.data.image_width,
            image_height=cfg.data.image_height,
            time_bins=cfg.data.time_bins,
            region_scales=cfg.data.region_scales,
            plane_types=cfg.data.plane_types,
            num_regions=cfg.data.num_regions,
            patch_size=cfg.model.patch_size,
            max_samples=cfg.data.max_samples,
        )
    else:
        raise ValueError(f"Unknown dataset_name: {cfg.data.dataset_name}")
    return DataLoader(dataset, batch_size=cfg.train.batch_size, shuffle=True)


def build_model(cfg: Config) -> EventMAE:
    return EventMAE(
        patch_size=cfg.model.patch_size,
        embed_dim=cfg.model.embed_dim,
        num_heads=cfg.model.num_heads,
        num_layers=cfg.model.num_layers,
        mlp_ratio=cfg.model.mlp_ratio,
        plane_embed_dim=cfg.model.plane_embed_dim,
        metadata_dim=cfg.model.metadata_dim,
        num_tokens=cfg.data.num_regions,
        num_planes=len(cfg.data.plane_types),
    )


def train_one_epoch(
    model: EventMAE,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    mask_ratio: float,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_steps = 0
    loss_fn = nn.MSELoss(reduction="sum")
    l1 = nn.L1Loss(reduction="sum")

    for batch in loader:
        patches = batch["patches"].to(device)
        metadata = batch["metadata"].to(device)
        plane_ids = batch["plane_ids"].to(device)
        counts = batch["event_counts"].to(device)

        mask_list = []
        for _ in range(patches.shape[0]):
            mask_list.append(random_mask(patches.shape[1], mask_ratio).to(device))
        mask = torch.stack(mask_list, dim=0)

        pred_patches, pred_counts, _ = model(patches, metadata, plane_ids, mask=mask)

        mask_patch = mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        masked_patch_loss = loss_fn(pred_patches * mask_patch, patches * mask_patch)
        masked_count_loss = l1(pred_counts * mask.unsqueeze(-1), counts * mask.unsqueeze(-1))
        denom = mask.sum().clamp(min=1.0)
        loss = (masked_patch_loss + masked_count_loss) / denom

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        total_steps += 1

    return {"loss": total_loss / max(1, total_steps)}
