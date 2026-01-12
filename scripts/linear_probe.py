from __future__ import annotations

import argparse
from pathlib import Path

import torch

from ecfm.models.mae import EventMAE
from ecfm.utils.config import load_config
from ecfm.training.linear_probe import (
    build_token_cache,
    compute_embeddings,
    load_split,
    train_linear_probe,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--train-split", type=str, default="train")
    parser.add_argument("--test-split", type=str, default="test")
    parser.add_argument("--subset-size", type=int, default=0)
    parser.add_argument("--subset-seed", type=int, default=0)
    parser.add_argument("--region-seed", type=int, default=0)
    parser.add_argument("--regions-per-sample", type=int, default=32)
    parser.add_argument("--cache-dir", type=str, default="outputs/linear_probe_cache")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--show-cache-progress", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(Path(args.config))

    if cfg.data.dataset_name != "thu-eact":
        raise ValueError("linear_probe currently supports thu-eact only")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        num_tokens=args.regions_per_sample,
        num_planes=len(cfg.data.plane_types),
        use_pos_embedding=cfg.model.use_pos_embedding,
    ).to(device)

    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location=device, weights_only=True)
        state_dict = state["model"] if isinstance(state, dict) and "model" in state else state
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

    cache_dir = Path(args.cache_dir)
    ckpt_tag = Path(args.checkpoint).stem if args.checkpoint else "random"
    train_cache = cache_dir / f"{args.train_split}_{ckpt_tag}_n{args.subset_size}_s{args.subset_seed}_r{args.region_seed}.npz"
    test_cache = cache_dir / f"{args.test_split}_{ckpt_tag}_n{args.subset_size}_s{args.subset_seed}_r{args.region_seed}.npz"

    train_patches, train_metadata, train_plane_ids, train_labels = build_token_cache(
        cfg,
        train_entries,
        args.subset_size,
        args.subset_seed,
        args.region_seed,
        args.regions_per_sample,
        train_cache,
        show_progress=args.show_cache_progress,
    )
    test_patches, test_metadata, test_plane_ids, test_labels = build_token_cache(
        cfg,
        test_entries,
        args.subset_size,
        args.subset_seed,
        args.region_seed,
        args.regions_per_sample,
        test_cache,
        show_progress=args.show_cache_progress,
    )

    train_emb = compute_embeddings(
        model, train_patches, train_metadata, train_plane_ids, args.batch_size, device
    )
    test_emb = compute_embeddings(
        model, test_patches, test_metadata, test_plane_ids, args.batch_size, device
    )

    train_acc, test_acc = train_linear_probe(
        train_emb=train_emb,
        train_labels=train_labels,
        test_emb=test_emb,
        test_labels=test_labels,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
    )
    print(f"linear_probe train_acc={train_acc:.4f} test_acc={test_acc:.4f}")


if __name__ == "__main__":
    main()
