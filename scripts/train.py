from __future__ import annotations

import argparse
from pathlib import Path

import torch

from ecfm.training.train import (
    build_dataloader,
    build_model,
    dump_reconstructions,
    resolve_num_tokens,
    train_one_epoch,
)
from ecfm.training.linear_probe import (
    build_token_cache,
    compute_embeddings,
    load_split,
    train_linear_probe,
)
from ecfm.utils.config import config_hash, load_config
from ecfm.utils.masking import random_mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.config)
    cfg = load_config(cfg_path)
    cfg.data.cache_token_config_id = config_hash(cfg_path)
    ckpt_tag = f"{cfg_path.stem}_{cfg.data.cache_token_config_id}"

    torch.manual_seed(cfg.train.seed)
    if cfg.train.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA requested but not available, falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(cfg.train.device)

    loader = build_dataloader(cfg)
    model = build_model(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr)
    start_step = 0

    if cfg.train.resume_path:
        ckpt = torch.load(cfg.train.resume_path, map_location=device)
        state_dict = ckpt["model"]
        model_state = model.state_dict()
        for key in ("pos_embedding", "decoder_pos_embedding"):
            if key in state_dict and key in model_state:
                if state_dict[key].shape != model_state[key].shape:
                    print(
                        f"Warning: dropping {key} from checkpoint due to shape mismatch "
                        f"{tuple(state_dict[key].shape)} vs {tuple(model_state[key].shape)}"
                    )
                    state_dict.pop(key)
        model.load_state_dict(state_dict, strict=False)
        if not cfg.train.load_model_only and "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
            start_step = int(ckpt.get("step", 0)) + 1
        print(f"Loaded checkpoint {cfg.train.resume_path} (start_step={start_step})")

    ema_loss = None
    ema_alpha = 0.9
    probe_cache = None
    probe_enabled = cfg.train.probe_every > 0

    if probe_enabled and cfg.data.dataset_name != "thu-eact":
        print("Linear probe only supports thu-eact right now; disabling.")
        probe_enabled = False
    if probe_enabled and cfg.train.probe_regions_per_sample != resolve_num_tokens(cfg):
        raise ValueError("probe_regions_per_sample must match the model num_tokens")

    for step in range(start_step, cfg.train.num_steps):
        metrics = train_one_epoch(
            model,
            loader,
            optimizer,
            device,
            mask_ratio=cfg.model.mask_ratio,
            patch_loss_blur_radius=cfg.train.patch_loss_blur_radius,
            count_loss_weight=cfg.train.count_loss_weight,
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
            valid_mask = batch.get("valid_mask")
            if valid_mask is not None:
                valid_mask = valid_mask.to(device)
            mask_list = []
            if valid_mask is None:
                for _ in range(patches.shape[0]):
                    mask_list.append(
                        random_mask(patches.shape[1], cfg.model.mask_ratio).to(device)
                    )
            else:
                for i in range(patches.shape[0]):
                    valid = valid_mask[i].bool()
                    if valid.sum() == 0:
                        mask_list.append(torch.zeros_like(valid, dtype=torch.bool))
                    else:
                        sampled = random_mask(int(valid.sum()), cfg.model.mask_ratio).to(device)
                        full_mask = torch.zeros_like(valid, dtype=torch.bool)
                        full_mask[valid] = sampled
                        mask_list.append(full_mask)
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
                patch_loss_blur_radius=cfg.train.patch_loss_blur_radius,
            )

        if cfg.train.checkpoint_every > 0 and step % cfg.train.checkpoint_every == 0:
            ckpt_dir = Path(cfg.train.checkpoint_dir)
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = ckpt_dir / f"{ckpt_tag}_step_{step:06d}.pt"
            torch.save(
                {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step},
                ckpt_path,
            )
            print(f"Saved checkpoint {ckpt_path}")
            last_path = ckpt_dir / f"{ckpt_tag}_last.pt"
            torch.save(
                {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step},
                last_path,
            )
            print(f"Saved checkpoint {last_path}")

        if probe_enabled and step % cfg.train.probe_every == 0:
            if probe_cache is None:
                root = Path(cfg.data.dataset_path)
                train_entries = load_split(root, "train")
                test_entries = load_split(root, "test")
                cache_dir = Path(cfg.train.probe_cache_dir)
                probe_tag = cfg.data.cache_token_config_id
                train_cache = cache_dir / (
                    f"train_h{probe_tag}_n{cfg.train.probe_subset_size}_s{cfg.train.probe_subset_seed}"
                    f"_r{cfg.train.probe_region_seed}.npz"
                )
                test_cache = cache_dir / (
                    f"test_h{probe_tag}_n{cfg.train.probe_subset_size}_s{cfg.train.probe_subset_seed}"
                    f"_r{cfg.train.probe_region_seed}.npz"
                )
                train_patches, train_metadata, train_plane_ids, train_labels = build_token_cache(
                    cfg,
                    train_entries,
                    cfg.train.probe_subset_size,
                    cfg.train.probe_subset_seed,
                    cfg.train.probe_region_seed,
                    cfg.train.probe_regions_per_sample,
                    train_cache,
                )
                test_patches, test_metadata, test_plane_ids, test_labels = build_token_cache(
                    cfg,
                    test_entries,
                    cfg.train.probe_subset_size,
                    cfg.train.probe_subset_seed,
                    cfg.train.probe_region_seed,
                    cfg.train.probe_regions_per_sample,
                    test_cache,
                )
                probe_cache = {
                    "train": (train_patches, train_metadata, train_plane_ids, train_labels),
                    "test": (test_patches, test_metadata, test_plane_ids, test_labels),
                }

            train_patches, train_metadata, train_plane_ids, train_labels = probe_cache["train"]
            test_patches, test_metadata, test_plane_ids, test_labels = probe_cache["test"]

            train_emb = compute_embeddings(
                model,
                train_patches,
                train_metadata,
                train_plane_ids,
                cfg.train.probe_batch_size,
                device,
            )
            test_emb = compute_embeddings(
                model,
                test_patches,
                test_metadata,
                test_plane_ids,
                cfg.train.probe_batch_size,
                device,
            )
            train_acc, test_acc = train_linear_probe(
                train_emb=train_emb,
                train_labels=train_labels,
                test_emb=test_emb,
                test_labels=test_labels,
                epochs=cfg.train.probe_epochs,
                batch_size=cfg.train.probe_batch_size,
                lr=cfg.train.probe_lr,
                device=device,
            )
            print(f"probe step={step} train_acc={train_acc:.4f} test_acc={test_acc:.4f}")

    if cfg.train.checkpoint_every >= 0:
        ckpt_dir = Path(cfg.train.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_dir / f"{ckpt_tag}_last.pt"
        torch.save(
            {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": cfg.train.num_steps - 1},
            ckpt_path,
        )
        print(f"Saved checkpoint {ckpt_path}")


if __name__ == "__main__":
    main()
