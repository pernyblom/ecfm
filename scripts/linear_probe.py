from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ecfm.models.mae import EventMAE
from ecfm.utils.config import config_hash, load_config
from ecfm.training.linear_probe import build_token_cache, compute_embeddings, load_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--train-split", type=str, default="train")
    parser.add_argument("--test-split", type=str, default="test")
    parser.add_argument("--subset-size", type=int, default=0)
    parser.add_argument("--subset-seed", type=int, default=0)
    parser.add_argument("--region-seed", type=int, default=0)
    parser.add_argument("--regions-per-sample", type=int, default=None)
    parser.add_argument("--cache-dir", type=str, default="outputs/linear_probe_cache")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--show-cache-progress", action="store_true")
    parser.add_argument("--val-split", type=float, default=0.0)
    parser.add_argument("--val-seed", type=int, default=0)
    parser.add_argument("--save-best", action="store_true")
    parser.add_argument("--out-dir", type=str, default="outputs/linear_probe")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.config)
    cfg = load_config(cfg_path)
    cfg_hash = config_hash(cfg_path)

    if cfg.data.dataset_name != "thu-eact":
        raise ValueError("linear_probe currently supports thu-eact only")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    regions_per_sample = args.regions_per_sample
    choices_max = max(cfg.data.num_regions_choices, default=0)
    if regions_per_sample is None:
        if choices_max > 0:
            regions_per_sample = choices_max
        elif cfg.data.region_sampling == "grid" and cfg.data.num_regions <= 0:
            grid_count = cfg.data.grid_x * cfg.data.grid_y * cfg.data.grid_t
            if cfg.data.grid_plane_mode == "all":
                grid_count *= len(cfg.data.plane_types)
            regions_per_sample = grid_count
        elif cfg.data.num_regions > 0:
            regions_per_sample = cfg.data.num_regions
        else:
            raise ValueError("regions_per_sample must be set unless region_sampling=grid")
    elif cfg.data.region_sampling == "grid" and regions_per_sample <= 0:
        grid_count = cfg.data.grid_x * cfg.data.grid_y * cfg.data.grid_t
        if cfg.data.grid_plane_mode == "all":
            grid_count *= len(cfg.data.plane_types)
        regions_per_sample = grid_count
    elif regions_per_sample <= 0:
        raise ValueError("regions_per_sample must be > 0 unless region_sampling=grid")

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
        num_tokens=regions_per_sample,
        num_planes=len(cfg.data.plane_types),
        use_pos_embedding=cfg.model.use_pos_embedding,
        use_relative_bias=cfg.model.use_relative_bias,
        rel_bias_hidden_dim=cfg.model.rel_bias_hidden_dim,
    ).to(device)

    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location=device, weights_only=True)
        state_dict = state["model"] if isinstance(state, dict) and "model" in state else state
        model_state = model.state_dict()
        for key in ("pos_embedding", "decoder_pos_embedding"):
            if key in state_dict and key in model_state:
                if state_dict[key].shape != model_state[key].shape:
                    print(
                        f"Warning: dropping {key} from checkpoint due to shape mismatch "
                        f"{tuple(state_dict[key].shape)} vs {tuple(model_state[key].shape)}"
                    )
                    state_dict.pop(key)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            print("Warning: checkpoint model keys mismatched.")
            if missing:
                print(f"  Missing keys: {len(missing)}")
            if unexpected:
                print(f"  Unexpected keys: {len(unexpected)}")
    else:
        print("Warning: no checkpoint provided, using random weights.")

    root = Path(cfg.data.dataset_path)
    train_entries = load_split(root, args.train_split)
    test_entries = load_split(root, args.test_split)

    if args.val_split > 0.0:
        full_train_entries = train_entries
        rng = np.random.default_rng(args.val_seed)
        labels = np.array([label for _, label in full_train_entries], dtype=np.int64)
        train_idx = []
        val_idx = []
        for label in np.unique(labels):
            idxs = np.where(labels == label)[0]
            rng.shuffle(idxs)
            val_count = max(1, int(round(len(idxs) * args.val_split))) if len(idxs) > 1 else 0
            val_idx.extend(idxs[:val_count].tolist())
            train_idx.extend(idxs[val_count:].tolist())
        train_entries = [full_train_entries[i] for i in train_idx]
        val_entries = [full_train_entries[i] for i in val_idx]
    else:
        val_entries = []

    cache_dir = Path(args.cache_dir)
    ckpt_tag = Path(args.checkpoint).stem if args.checkpoint else "random"
    train_cache = (
        cache_dir
        / f"{args.train_split}_h{cfg_hash}_{ckpt_tag}_n{args.subset_size}_s{args.subset_seed}_r{args.region_seed}_k{regions_per_sample}.npz"
    )
    test_cache = (
        cache_dir
        / f"{args.test_split}_h{cfg_hash}_{ckpt_tag}_n{args.subset_size}_s{args.subset_seed}_r{args.region_seed}_k{regions_per_sample}.npz"
    )

    train_patches, train_metadata, train_plane_ids, train_labels = build_token_cache(
        cfg,
        train_entries,
        args.subset_size,
        args.subset_seed,
        args.region_seed,
        regions_per_sample,
        train_cache,
        show_progress=args.show_cache_progress,
    )
    test_patches, test_metadata, test_plane_ids, test_labels = build_token_cache(
        cfg,
        test_entries,
        args.subset_size,
        args.subset_seed,
        args.region_seed,
        regions_per_sample,
        test_cache,
        show_progress=args.show_cache_progress,
    )

    train_emb = compute_embeddings(
        model, train_patches, train_metadata, train_plane_ids, args.batch_size, device
    )
    test_emb = compute_embeddings(
        model, test_patches, test_metadata, test_plane_ids, args.batch_size, device
    )
    if val_entries:
        val_cache = (
            cache_dir
            / f"val_h{cfg_hash}_{ckpt_tag}_n{args.subset_size}_s{args.subset_seed}_r{args.region_seed}_k{regions_per_sample}.npz"
        )
        val_patches, val_metadata, val_plane_ids, val_labels = build_token_cache(
            cfg,
            val_entries,
            args.subset_size,
            args.subset_seed,
            args.region_seed,
            regions_per_sample,
            val_cache,
            show_progress=args.show_cache_progress,
        )
        val_emb = compute_embeddings(
            model, val_patches, val_metadata, val_plane_ids, args.batch_size, device
        )

    num_classes = int(max(train_labels.max(), test_labels.max()) + 1)
    head = nn.Linear(train_emb.shape[1], num_classes).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    train_ds = TensorDataset(
        torch.from_numpy(train_emb).float(),
        torch.from_numpy(train_labels).long(),
    )
    test_ds = TensorDataset(
        torch.from_numpy(test_emb).float(),
        torch.from_numpy(test_labels).long(),
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    if val_entries:
        val_ds = TensorDataset(
            torch.from_numpy(val_emb).float(),
            torch.from_numpy(val_labels).long(),
        )
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    else:
        val_loader = None

    best_val = -1.0
    best_state = None
    for epoch in range(args.epochs):
        head.train()
        total = 0
        correct = 0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            logits = head(x)
            loss = loss_fn(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += x.size(0)
            correct += int((logits.argmax(dim=1) == y).sum().item())
        train_acc = correct / max(1, total)

        if val_loader is not None:
            head.eval()
            total = 0
            correct = 0
            val_loss = 0.0
            with torch.no_grad():
                for x, y in val_loader:
                    x = x.to(device)
                    y = y.to(device)
                    logits = head(x)
                    val_loss += float(loss_fn(logits, y).item()) * x.size(0)
                    total += x.size(0)
                    correct += int((logits.argmax(dim=1) == y).sum().item())
            val_acc = correct / max(1, total)
            val_loss = val_loss / max(1, total)
            if val_acc > best_val:
                best_val = val_acc
                best_state = head.state_dict()
            print(
                f"epoch={epoch} train_acc={train_acc:.4f} val_acc={val_acc:.4f} val_loss={val_loss:.4f}"
            )
        else:
            print(f"epoch={epoch} train_acc={train_acc:.4f}")

    if best_state is not None:
        head.load_state_dict(best_state)
    head.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)
            logits = head(x)
            total += x.size(0)
            correct += int((logits.argmax(dim=1) == y).sum().item())
    test_acc = correct / max(1, total)

    if val_entries:
        print(f"linear_probe best_val_acc={best_val:.4f} test_acc={test_acc:.4f}")
        if args.save_best:
            out_dir = Path(args.out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            torch.save(best_state, out_dir / "best_head.pt")
            np.savez(
                out_dir / "val_split_indices.npz",
                train_indices=np.array(train_idx, dtype=np.int64),
                val_indices=np.array(val_idx, dtype=np.int64),
            )
    else:
        print(f"linear_probe test_acc={test_acc:.4f}")


if __name__ == "__main__":
    main()
