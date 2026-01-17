from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision.models import ResNet18_Weights, resnet18
import yaml

from ecfm.data.dvs_lip_dataset import DVSLipDataset
from ecfm.data.thu_eact_dataset import THUEACTDataset
from ecfm.data.region_utils import grid_region_count
from ecfm.utils.config import config_hash, load_config


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_datasets(cfg, train_split: str, test_split: str, seed: int, val_split: float):
    name = cfg.data.dataset_name.lower()
    data_kwargs = {
        "image_width": cfg.data.image_width,
        "image_height": cfg.data.image_height,
        "time_unit": cfg.data.time_unit,
        "time_bins": cfg.data.time_bins,
        "region_scales": cfg.data.region_scales,
        "region_scales_x": cfg.data.region_scales_x,
        "region_scales_y": cfg.data.region_scales_y,
        "region_time_scales": cfg.data.region_time_scales,
        "region_sampling": cfg.data.region_sampling,
        "grid_x": cfg.data.grid_x,
        "grid_y": cfg.data.grid_y,
        "grid_t": cfg.data.grid_t,
        "grid_plane_mode": cfg.data.grid_plane_mode,
        "plane_types": cfg.data.plane_types,
        "plane_types_active": cfg.data.plane_types_active,
        "num_regions": cfg.data.num_regions,
        "num_regions_choices": cfg.data.num_regions_choices,
        "patch_size": cfg.model.patch_size,
        "max_samples": cfg.data.max_samples,
        "max_events": cfg.data.max_events,
        "subset_seed": cfg.data.subset_seed,
        "fixed_region_seed": cfg.data.fixed_region_seed,
        "patch_divider": cfg.data.patch_divider,
        "patch_norm": cfg.data.patch_norm,
        "patch_norm_eps": cfg.data.patch_norm_eps,
        "augmentations": cfg.data.augmentations,
        "rotation_max_deg": cfg.data.rotation_max_deg,
        "augmentation_invalidate_prob": cfg.data.augmentation_invalidate_prob,
        "fixed_region_sizes": cfg.data.fixed_region_sizes,
        "fixed_region_positions_global": cfg.data.fixed_region_positions_global,
        "fixed_single_region": cfg.data.fixed_single_region,
        "cache_max_samples": cfg.data.cache_max_samples,
        "cache_token_max_samples": cfg.data.cache_token_max_samples,
        "cache_token_variants_per_sample": cfg.data.cache_token_variants_per_sample,
        "cache_token_dir": cfg.data.cache_token_dir,
        "cache_token_variant_mode": cfg.data.cache_token_variant_mode,
        "cache_token_clear_on_start": cfg.data.cache_token_clear_on_start,
        "cache_token_config_id": cfg.data.cache_token_config_id,
        "cache_token_drop_prob": cfg.data.cache_token_drop_prob,
        "return_label": True,
    }
    if name == "thu-eact":
        train_ds = THUEACTDataset(
            root=cfg.data.dataset_path,
            split=train_split,
            apply_augmentations=True,
            **data_kwargs,
        )
        val_base_ds = THUEACTDataset(
            root=cfg.data.dataset_path,
            split=train_split,
            apply_augmentations=False,
            **data_kwargs,
        )
        test_ds = THUEACTDataset(
            root=cfg.data.dataset_path,
            split=test_split,
            apply_augmentations=False,
            **data_kwargs,
        )
    elif name == "dvs-lip":
        train_ds = DVSLipDataset(
            root=cfg.data.dataset_path,
            split=train_split,
            apply_augmentations=True,
            **data_kwargs,
        )
        val_base_ds = DVSLipDataset(
            root=cfg.data.dataset_path,
            split=train_split,
            apply_augmentations=False,
            class_to_idx=train_ds.class_to_idx,
            **data_kwargs,
        )
        test_ds = DVSLipDataset(
            root=cfg.data.dataset_path,
            split=test_split,
            apply_augmentations=False,
            class_to_idx=train_ds.class_to_idx,
            **data_kwargs,
        )
    else:
        raise ValueError(f"Unsupported dataset_name for baseline: {name}")

    rng = np.random.default_rng(seed)
    train_idx = []
    val_idx = []
    labels: Dict[int, list[int]] = {}
    if not hasattr(train_ds, "labels"):
        raise ValueError("Dataset does not expose labels for stratified split")
    for idx in range(len(train_ds)):
        label = int(train_ds.labels[idx])
        labels.setdefault(label, []).append(idx)
    for _, idxs in labels.items():
        perm = rng.permutation(idxs)
        val_count = max(1, int(round(len(perm) * val_split))) if len(perm) > 1 else 0
        val_idx.extend(perm[:val_count])
        train_idx.extend(perm[val_count:])
    return Subset(train_ds, train_idx), Subset(val_base_ds, val_idx), test_ds


def ensure_single_region(cfg) -> None:
    grid_count = grid_region_count(
        cfg.data.grid_x,
        cfg.data.grid_y,
        cfg.data.grid_t,
        cfg.data.grid_plane_mode,
        cfg.data.plane_types_active,
    )
    max_regions = max(
        0,
        cfg.data.num_regions,
        max(cfg.data.num_regions_choices, default=0),
        grid_count if cfg.data.region_sampling == "grid" and cfg.data.num_regions <= 0 else 0,
    )
    if max_regions != 1:
        raise ValueError(
            f"ResNet baseline expects exactly one region; got max_regions={max_regions}. "
            "Set num_regions=1 or grid_x=grid_y=grid_t=1 with a single plane."
        )


def make_inputs(batch: Dict[str, torch.Tensor], device: torch.device) -> torch.Tensor:
    patches = batch["patches"].to(device)
    if patches.ndim != 5 or patches.shape[1] < 1 or patches.shape[2] != 2:
        raise ValueError(f"Unexpected patch shape: {patches.shape}")
    patch = patches[:, 0]
    img = torch.zeros((patch.shape[0], 3, patch.shape[2], patch.shape[3]), device=device)
    img[:, 0] = patch[:, 0]
    img[:, 2] = patch[:, 1]
    return img


def maybe_imagenet_norm(img: torch.Tensor, enable: bool) -> torch.Tensor:
    if not enable:
        return img
    mean = torch.tensor([0.485, 0.456, 0.406], device=img.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=img.device).view(1, 3, 1, 1)
    return (img - mean) / std


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    imagenet_norm: bool,
) -> Tuple[float, float]:
    is_train = optimizer is not None
    if is_train:
        model.train()
    else:
        model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    for batch in loader:
        labels = batch["label"].to(device)
        inputs = make_inputs(batch, device)
        inputs = maybe_imagenet_norm(inputs, imagenet_norm)
        logits = model(inputs)
        loss = loss_fn(logits, labels)
        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        total_loss += float(loss.item()) * labels.size(0)
        preds = torch.argmax(logits, dim=1)
        total_correct += int((preds == labels).sum().item())
        total += int(labels.size(0))
    if total == 0:
        return 0.0, 0.0
    return total_loss / total, total_correct / total


def main() -> None:
    parser = argparse.ArgumentParser(description="ResNet-18 baseline on event projections")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=str, default="outputs/resnet_baseline")
    parser.add_argument("--train-split", type=str, default="train")
    parser.add_argument("--test-split", type=str, default="test")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--imagenet-norm", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    cfg_path = Path(args.config)
    cfg = load_config(cfg_path)
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    cfg.data.cache_token_config_id = config_hash(cfg_path)

    baseline_cfg = raw.get("baseline", {})
    pretrained = bool(baseline_cfg.get("pretrained", args.pretrained))
    has_imagenet_flag = "imagenet_norm" in baseline_cfg
    imagenet_norm = bool(baseline_cfg.get("imagenet_norm", args.imagenet_norm))
    if pretrained and not has_imagenet_flag and not args.imagenet_norm:
        imagenet_norm = True

    set_seed(args.seed)
    ensure_single_region(cfg)

    train_ds, val_ds, test_ds = build_datasets(
        cfg, args.train_split, args.test_split, args.seed, args.val_split
    )
    print(
        f"Dataset sizes: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)} "
        f"plane={cfg.data.plane_types_active}"
    )

    if pretrained:
        weights = ResNet18_Weights.DEFAULT
        model = resnet18(weights=weights)
    else:
        model = resnet18(weights=None)

    num_classes = 0
    if hasattr(train_ds.dataset, "labels"):
        num_classes = int(max(train_ds.dataset.labels) + 1)
    elif hasattr(train_ds.dataset, "class_to_idx"):
        num_classes = len(train_ds.dataset.class_to_idx)
    if num_classes <= 0:
        raise ValueError("Unable to infer number of classes from dataset labels")
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_best = out_dir / f"{cfg_path.stem}_resnet18_best.pt"
    ckpt_last = out_dir / f"{cfg_path.stem}_resnet18_last.pt"

    best_val_acc = -1.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(
            model, train_loader, device, loss_fn, optimizer, imagenet_norm
        )
        val_loss, val_acc = run_epoch(model, val_loader, device, loss_fn, None, imagenet_norm)
        print(
            f"epoch={epoch} train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "val_acc": val_acc,
                    "val_loss": val_loss,
                    "config": str(cfg_path),
                },
                ckpt_best,
            )
        torch.save(
            {
                "model": model.state_dict(),
                "epoch": epoch,
                "val_acc": val_acc,
                "val_loss": val_loss,
                "config": str(cfg_path),
            },
            ckpt_last,
        )

    test_loss, test_acc = run_epoch(model, test_loader, device, loss_fn, None, imagenet_norm)
    print(f"test_loss={test_loss:.4f} test_acc={test_acc:.4f}")


if __name__ == "__main__":
    main()
