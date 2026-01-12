from __future__ import annotations

from typing import Dict, List

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from ecfm.data.event_dataset import SyntheticEventDataset, THUEACTDataset
from ecfm.models.mae import EventMAE
from ecfm.utils.config import Config
from ecfm.utils.masking import random_mask

import numpy as np
from pathlib import Path
from PIL import Image


def build_dataloader(cfg: Config) -> DataLoader:
    if cfg.data.dataset_name == "synthetic":
        if cfg.data.cache_token_variants_per_sample > 0 and cfg.data.fixed_region_seed < 0:
            print(
                "Warning: cache_token_variants_per_sample is enabled but fixed_region_seed < 0; "
                "token variants will not be reproducible across runs."
            )
        if cfg.data.cache_token_max_samples > 0 and cfg.data.fixed_region_seed < 0:
            print(
                "Warning: cache_token_max_samples is enabled but fixed_region_seed < 0; "
                "token caching will freeze randomly sampled regions."
            )
        dataset = SyntheticEventDataset(
            num_samples=cfg.data.num_samples,
            max_events=cfg.data.max_events,
            image_width=cfg.data.image_width,
            image_height=cfg.data.image_height,
            time_bins=cfg.data.time_bins,
            region_scales=cfg.data.region_scales,
            region_scales_x=cfg.data.region_scales_x,
            region_scales_y=cfg.data.region_scales_y,
            region_time_scales=cfg.data.region_time_scales,
            region_sampling=cfg.data.region_sampling,
            grid_x=cfg.data.grid_x,
            grid_y=cfg.data.grid_y,
            grid_t=cfg.data.grid_t,
            grid_plane_mode=cfg.data.grid_plane_mode,
            plane_types=cfg.data.plane_types,
            num_regions=cfg.data.num_regions,
            patch_size=cfg.model.patch_size,
            rng_seed=cfg.train.seed,
            fixed_region_seed=cfg.data.fixed_region_seed,
            patch_divider=cfg.data.patch_divider,
            fixed_region_sizes=cfg.data.fixed_region_sizes,
            fixed_region_positions_global=cfg.data.fixed_region_positions_global,
            fixed_single_region=cfg.data.fixed_single_region,
            cache_token_max_samples=cfg.data.cache_token_max_samples,
            cache_token_variants_per_sample=cfg.data.cache_token_variants_per_sample,
            cache_token_dir=cfg.data.cache_token_dir,
            cache_token_variant_mode=cfg.data.cache_token_variant_mode,
            cache_token_clear_on_start=cfg.data.cache_token_clear_on_start,
        )
    elif cfg.data.dataset_name == "thu-eact":
        if cfg.data.cache_token_variants_per_sample > 0 and cfg.data.fixed_region_seed < 0:
            print(
                "Warning: cache_token_variants_per_sample is enabled but fixed_region_seed < 0; "
                "token variants will not be reproducible across runs."
            )
        if cfg.data.cache_token_max_samples > 0 and cfg.data.fixed_region_seed < 0:
            print(
                "Warning: cache_token_max_samples is enabled but fixed_region_seed < 0; "
                "token caching will freeze randomly sampled regions."
            )
        dataset = THUEACTDataset(
            root=cfg.data.dataset_path,
            split=cfg.data.split,
            image_width=cfg.data.image_width,
            image_height=cfg.data.image_height,
            time_unit=cfg.data.time_unit,
            time_bins=cfg.data.time_bins,
            region_scales=cfg.data.region_scales,
            region_scales_x=cfg.data.region_scales_x,
            region_scales_y=cfg.data.region_scales_y,
            region_time_scales=cfg.data.region_time_scales,
            region_sampling=cfg.data.region_sampling,
            grid_x=cfg.data.grid_x,
            grid_y=cfg.data.grid_y,
            grid_t=cfg.data.grid_t,
            grid_plane_mode=cfg.data.grid_plane_mode,
            plane_types=cfg.data.plane_types,
            num_regions=cfg.data.num_regions,
            patch_size=cfg.model.patch_size,
            max_samples=cfg.data.max_samples,
            max_events=cfg.data.max_events,
            subset_seed=cfg.data.subset_seed,
            fixed_region_seed=cfg.data.fixed_region_seed,
            patch_divider=cfg.data.patch_divider,
            fixed_region_sizes=cfg.data.fixed_region_sizes,
            fixed_region_positions_global=cfg.data.fixed_region_positions_global,
            fixed_single_region=cfg.data.fixed_single_region,
            cache_max_samples=cfg.data.cache_max_samples,
            cache_token_max_samples=cfg.data.cache_token_max_samples,
            cache_token_variants_per_sample=cfg.data.cache_token_variants_per_sample,
            cache_token_dir=cfg.data.cache_token_dir,
            cache_token_variant_mode=cfg.data.cache_token_variant_mode,
            cache_token_clear_on_start=cfg.data.cache_token_clear_on_start,
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
        decoder_embed_dim=cfg.model.decoder_embed_dim,
        decoder_num_heads=cfg.model.decoder_num_heads,
        decoder_num_layers=cfg.model.decoder_num_layers,
        mlp_ratio=cfg.model.mlp_ratio,
        plane_embed_dim=cfg.model.plane_embed_dim,
        metadata_dim=cfg.model.metadata_dim,
        num_tokens=cfg.data.num_regions,
        num_planes=len(cfg.data.plane_types),
        use_pos_embedding=cfg.model.use_pos_embedding,
    )


def train_one_epoch(
    model: EventMAE,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    mask_ratio: float,
    patch_loss_blur_radius: int = 0,
    count_loss_weight: float = 1.0,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_steps = 0
    loss_fn = nn.MSELoss(reduction="sum")
    l1 = nn.L1Loss(reduction="sum")

    blur_kernel = None
    if patch_loss_blur_radius > 0:
        radius = int(patch_loss_blur_radius)
        ksize = radius * 2 + 1
        coords = torch.arange(ksize, device=device) - radius
        sigma = max(1e-6, radius / 2.0)
        gauss_1d = torch.exp(-(coords**2) / (2 * sigma * sigma))
        gauss_1d = gauss_1d / gauss_1d.sum()
        gauss_2d = gauss_1d[:, None] * gauss_1d[None, :]
        blur_kernel = gauss_2d.view(1, 1, ksize, ksize)

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
        if blur_kernel is not None:
            c = patches.shape[2]
            kernel = blur_kernel.expand(c, 1, -1, -1)
            pred_blur = F.conv2d(
                pred_patches.view(-1, c, pred_patches.shape[-2], pred_patches.shape[-1]),
                kernel,
                padding=patch_loss_blur_radius,
                groups=c,
            ).view_as(pred_patches)
            patch_blur = F.conv2d(
                patches.view(-1, c, patches.shape[-2], patches.shape[-1]),
                kernel,
                padding=patch_loss_blur_radius,
                groups=c,
            ).view_as(patches)
            masked_patch_loss = loss_fn(pred_blur * mask_patch, patch_blur * mask_patch)
        else:
            masked_patch_loss = loss_fn(pred_patches * mask_patch, patches * mask_patch)
        masked_count_loss = torch.tensor(0.0, device=device)
        if count_loss_weight > 0:
            pred_counts_pos = F.softplus(pred_counts)
            count_target = torch.log1p(counts)
            count_pred = torch.log1p(pred_counts_pos)
            masked_count_loss = l1(
                count_pred * mask.unsqueeze(-1), count_target * mask.unsqueeze(-1)
            )
        denom = mask.sum().clamp(min=1.0)
        loss = (masked_patch_loss + count_loss_weight * masked_count_loss) / denom

        if not torch.isfinite(loss):
            optimizer.zero_grad()
            continue

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += float(loss.item())
        total_steps += 1

    if total_steps == 0:
        return {"loss": float("nan")}
    return {"loss": total_loss / total_steps}


def dump_reconstructions(
    step: int,
    patches: torch.Tensor,
    pred_patches: torch.Tensor,
    mask: torch.Tensor,
    plane_ids: torch.Tensor,
    plane_types: List[str],
    out_dir: str,
    num_patches: int,
    upscale: int,
    patch_loss_blur_radius: int = 0,
) -> None:
    out_path = Path(out_dir) / f"step_{step:06d}"
    out_path.mkdir(parents=True, exist_ok=True)

    if patch_loss_blur_radius > 0:
        radius = int(patch_loss_blur_radius)
        ksize = radius * 2 + 1
        coords = torch.arange(ksize, device=patches.device) - radius
        sigma = max(1e-6, radius / 2.0)
        gauss_1d = torch.exp(-(coords**2) / (2 * sigma * sigma))
        gauss_1d = gauss_1d / gauss_1d.sum()
        gauss_2d = gauss_1d[:, None] * gauss_1d[None, :]
        blur_kernel = gauss_2d.view(1, 1, ksize, ksize)
        c = patches.shape[2]
        kernel = blur_kernel.expand(c, 1, -1, -1)
        pred_patches = torch.nn.functional.conv2d(
            pred_patches.view(-1, c, pred_patches.shape[-2], pred_patches.shape[-1]),
            kernel,
            padding=radius,
            groups=c,
        ).view_as(pred_patches)
        patches = torch.nn.functional.conv2d(
            patches.view(-1, c, patches.shape[-2], patches.shape[-1]),
            kernel,
            padding=radius,
            groups=c,
        ).view_as(patches)

    patches = patches.detach().cpu().numpy()
    pred_patches = pred_patches.detach().cpu().numpy()
    plane_ids = plane_ids.detach().cpu().numpy()
    mask = mask.detach().cpu().numpy()

    count = min(num_patches, patches.shape[1])
    if upscale < 1:
        raise ValueError("upscale must be >= 1")

    saved = 0
    for i in range(patches.shape[1]):
        if not mask[0, i]:
            continue
        plane = plane_types[int(plane_ids[0, i])]
        for ch in range(2):
            gt = patches[0, i, ch]
            pred = np.nan_to_num(pred_patches[0, i, ch], nan=0.0, posinf=0.0, neginf=0.0)
            shared_max = float(max(gt.max(initial=0.0), pred.max(initial=0.0)))
            if shared_max > 0:
                gt = gt / shared_max
                pred = pred / shared_max
            gt = np.clip(gt * 255.0, 0, 255).astype(np.uint8)
            pred = np.clip(pred * 255.0, 0, 255).astype(np.uint8)
            if upscale > 1:
                gt = np.repeat(np.repeat(gt, upscale, axis=0), upscale, axis=1)
                pred = np.repeat(np.repeat(pred, upscale, axis=0), upscale, axis=1)
            Image.fromarray(gt, mode="L").save(out_path / f"token_{i:03d}_{plane}_p{ch}_gt.png")
            Image.fromarray(pred, mode="L").save(out_path / f"token_{i:03d}_{plane}_p{ch}_pred.png")
        saved += 1
        if saved >= count:
            break
