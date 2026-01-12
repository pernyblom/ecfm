from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from ecfm.models.mae import EventMAE
from ecfm.data.event_dataset import THUEACTDataset
from ecfm.training.linear_probe import load_split
from ecfm.utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=str, default="outputs/finetune")
    parser.add_argument("--train-split", type=str, default="train")
    parser.add_argument("--test-split", type=str, default="test")
    return parser.parse_args()


def build_model(cfg, device: torch.device) -> EventMAE:
    model = EventMAE(
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
    ).to(device)
    return model


def load_checkpoint(model: EventMAE, path: str, device: torch.device) -> None:
    state = torch.load(path, map_location=device, weights_only=True)
    state_dict = state["model"] if isinstance(state, dict) and "model" in state else state
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print("Warning: checkpoint model keys mismatched.")
        if missing:
            print(f"  Missing keys: {len(missing)}")
        if unexpected:
            print(f"  Unexpected keys: {len(unexpected)}")


def build_datasets(cfg, train_split: str, test_split: str, seed: int, val_split: float):
    load_split(Path(cfg.data.dataset_path), train_split)
    load_split(Path(cfg.data.dataset_path), test_split)

    train_ds = THUEACTDataset(
        root=cfg.data.dataset_path,
        split=train_split,
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
        plane_types=cfg.data.plane_types,
        num_regions=cfg.data.num_regions,
        patch_size=cfg.model.patch_size,
        max_samples=0,
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
        return_label=True,
    )

    test_ds = THUEACTDataset(
        root=cfg.data.dataset_path,
        split=test_split,
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
        plane_types=cfg.data.plane_types,
        num_regions=cfg.data.num_regions,
        patch_size=cfg.model.patch_size,
        max_samples=0,
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
        return_label=True,
    )

    labels = train_ds.labels
    rng = torch.Generator().manual_seed(seed)
    label_to_indices = {}
    for idx, label in enumerate(labels):
        label_to_indices.setdefault(int(label), []).append(idx)
    train_idx = []
    val_idx = []
    for label, idxs in label_to_indices.items():
        idxs_t = torch.tensor(idxs)
        perm = idxs_t[torch.randperm(len(idxs_t), generator=rng)].tolist()
        val_count = max(1, int(round(len(perm) * val_split))) if len(perm) > 1 else 0
        val_idx.extend(perm[:val_count])
        train_idx.extend(perm[val_count:])
    return Subset(train_ds, train_idx), Subset(train_ds, val_idx), test_ds


def run_epoch(
    model: EventMAE,
    head: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> Tuple[float, float]:
    train_mode = optimizer is not None
    model.train(train_mode)
    head.train(train_mode)
    total = 0
    correct = 0
    total_loss = 0.0
    loss_fn = nn.CrossEntropyLoss()

    for batch in loader:
        patches = batch["patches"].to(device)
        metadata = batch["metadata"].to(device)
        plane_ids = batch["plane_ids"].to(device)
        labels = batch["label"].to(device)
        with torch.set_grad_enabled(train_mode):
            encoded = model.encode(patches, metadata, plane_ids, mask=None)
            pooled = encoded.mean(dim=1)
            logits = head(pooled)
            loss = loss_fn(logits, labels)
            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        total_loss += float(loss.item()) * labels.size(0)
        total += labels.size(0)
        correct += int((logits.argmax(dim=1) == labels).sum().item())

    return total_loss / max(1, total), correct / max(1, total)


def main() -> None:
    args = parse_args()
    cfg = load_config(Path(args.config))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg, device)
    load_checkpoint(model, args.checkpoint, device)

    train_ds, val_ds, test_ds = build_datasets(
        cfg, args.train_split, args.test_split, args.seed, args.val_split
    )
    print(
        f"train={len(train_ds)} val={len(val_ds)} test={len(test_ds)} "
        f"classes={len(set(train_ds.dataset.labels))}"
    )
    num_classes = len(set(train_ds.dataset.labels))
    head = nn.Linear(cfg.model.embed_dim, num_classes).to(device)
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(head.parameters()), lr=args.lr)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_acc = -1.0
    best_path = out_dir / "best.pt"

    for epoch in range(args.epochs):
        train_loss, train_acc = run_epoch(model, head, train_loader, optimizer, device)
        val_loss, val_acc = run_epoch(model, head, val_loader, None, device)
        print(
            f"epoch={epoch} train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(
                {
                    "model": model.state_dict(),
                    "head": head.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "best_val_acc": best_acc,
                },
                best_path,
            )
            print(f"Saved best checkpoint {best_path}")

    last_path = out_dir / "last.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "head": head.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": args.epochs - 1,
            "best_val_acc": best_acc,
        },
        last_path,
    )
    print(f"Saved last checkpoint {last_path}")

    ckpt = torch.load(best_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"], strict=False)
    head.load_state_dict(ckpt["head"], strict=False)
    test_loss, test_acc = run_epoch(model, head, test_loader, None, device)
    print(f"test_loss={test_loss:.4f} test_acc={test_acc:.4f}")


if __name__ == "__main__":
    main()
