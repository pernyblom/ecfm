from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ecfm.data.tokenizer import Region, build_patch


def load_split(root: Path, split: str) -> List[Tuple[Path, int]]:
    list_path = root / f"{split}.txt"
    if not list_path.exists():
        raise FileNotFoundError(f"Missing split list: {list_path}")
    entries = []
    for line in list_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 2:
            raise ValueError(f"Missing label in split list: {line}")
        name = Path(parts[0]).name
        label = int(parts[1])
        candidate = root / name
        if candidate.exists():
            entries.append((candidate, label))
    if not entries:
        raise FileNotFoundError("No dataset files found for split list")
    return entries


def sample_region(
    rng: np.random.Generator,
    image_width: int,
    image_height: int,
    region_scales: List[int],
    region_scales_x: List[int],
    region_scales_y: List[int],
    region_time_scales: List[float],
    plane_types: List[str],
) -> Region:
    if region_scales_x:
        dx = int(rng.choice(region_scales_x))
    else:
        dx = int(rng.choice(region_scales))
    if region_scales_y:
        dy = int(rng.choice(region_scales_y))
    else:
        dy = int(rng.choice(region_scales))
    dx = min(dx, image_width)
    dy = min(dy, image_height)
    dt = float(rng.choice(region_time_scales))
    x = int(rng.integers(0, max(1, image_width - dx)))
    y = int(rng.integers(0, max(1, image_height - dy)))
    t = float(rng.random() * max(1e-6, 1.0 - dt))
    plane = str(rng.choice(plane_types))
    return Region(x=x, y=y, t=t, dx=dx, dy=dy, dt=dt, plane=plane)


def grid_regions(
    image_width: int,
    image_height: int,
    grid_x: int,
    grid_y: int,
    grid_t: int,
    plane_types: List[str],
    grid_plane_mode: str,
) -> List[Region]:
    if grid_x <= 0 or grid_y <= 0 or grid_t <= 0:
        raise ValueError("grid_x, grid_y, grid_t must be > 0 for grid sampling")
    xs = np.linspace(0, image_width, grid_x + 1, dtype=int)
    ys = np.linspace(0, image_height, grid_y + 1, dtype=int)
    ts = np.linspace(0.0, 1.0, grid_t + 1, dtype=np.float32)
    regions: List[Region] = []
    for ti in range(grid_t):
        t = float(ts[ti])
        dt = float(ts[ti + 1] - ts[ti])
        for yi in range(grid_y):
            y = int(ys[yi])
            dy = int(max(1, ys[yi + 1] - ys[yi]))
            if y + dy > image_height:
                dy = image_height - y
            for xi in range(grid_x):
                x = int(xs[xi])
                dx = int(max(1, xs[xi + 1] - xs[xi]))
                if x + dx > image_width:
                    dx = image_width - x
                if grid_plane_mode == "all":
                    for plane in plane_types:
                        regions.append(Region(x=x, y=y, t=t, dx=dx, dy=dy, dt=dt, plane=plane))
                else:
                    plane = plane_types[len(regions) % len(plane_types)]
                    regions.append(Region(x=x, y=y, t=t, dx=dx, dy=dy, dt=dt, plane=plane))
    return regions


def load_events(path: Path, time_unit: float) -> Tuple[np.ndarray, float]:
    events = np.load(path).astype(np.float32)
    if events.ndim != 2 or events.shape[1] != 4:
        raise ValueError(f"Unexpected event shape in {path}: {events.shape}")
    t = events[:, 2]
    t_min = float(t.min())
    t_max = float(t.max())
    denom = max(1e-6, t_max - t_min)
    events[:, 2] = (t - t_min) / denom
    return events, denom * time_unit


def build_token_cache(
    cfg,
    entries: List[Tuple[Path, int]],
    subset_size: int,
    subset_seed: int,
    region_seed: int,
    regions_per_sample: int,
    cache_path: Path,
    show_progress: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if cache_path.exists():
        cached = np.load(cache_path, allow_pickle=True)
        return (
            cached["patches"],
            cached["metadata"],
            cached["plane_ids"],
            cached["labels"],
        )

    rng = np.random.default_rng(subset_seed)
    if subset_size > 0 and subset_size < len(entries):
        idx = rng.choice(len(entries), size=subset_size, replace=False)
        entries = [entries[i] for i in idx]

    patches_list = []
    metadata_list = []
    plane_ids_list = []
    labels = []

    grid_list = None
    if cfg.data.region_sampling == "grid":
        grid_list = grid_regions(
            cfg.data.image_width,
            cfg.data.image_height,
            cfg.data.grid_x,
            cfg.data.grid_y,
            cfg.data.grid_t,
            cfg.data.plane_types,
            cfg.data.grid_plane_mode,
        )

    total = len(entries)
    for idx, (path, label) in enumerate(entries):
        if show_progress and (idx == 0 or (idx + 1) % 50 == 0 or idx + 1 == total):
            print(f"caching {idx + 1}/{total} -> {cache_path.name}")
        events, seq_len_sec = load_events(path, cfg.data.time_unit)
        sample_rng = np.random.default_rng(region_seed + idx)

        patches = []
        metadata = []
        plane_ids = []

        if grid_list is not None:
            regions = grid_list
            if regions_per_sample > 0 and regions_per_sample < len(grid_list):
                idxs = sample_rng.choice(len(grid_list), size=regions_per_sample, replace=False)
                regions = [grid_list[i] for i in idxs]
        else:
            regions = [
                sample_region(
                    sample_rng,
                    cfg.data.image_width,
                    cfg.data.image_height,
                    cfg.data.region_scales,
                    cfg.data.region_scales_x,
                    cfg.data.region_scales_y,
                    cfg.data.region_time_scales,
                    cfg.data.plane_types,
                )
                for _ in range(regions_per_sample)
            ]

        for region in regions:
            patch, _ = build_patch(
                events,
                region,
                cfg.model.patch_size,
                cfg.data.time_bins,
                patch_divider=cfg.data.patch_divider,
            )
            patches.append(patch.numpy())
            plane_ids.append(cfg.data.plane_types.index(region.plane))
            meta = np.array(
                [
                    region.x / cfg.data.image_width,
                    region.y / cfg.data.image_height,
                    region.dx / cfg.data.image_width,
                    region.dy / cfg.data.image_height,
                    region.t,
                    region.dt,
                    region.t * seq_len_sec,
                    region.dt * seq_len_sec,
                    seq_len_sec,
                ],
                dtype=np.float32,
            )
            metadata.append(meta)

        patches_list.append(np.stack(patches, axis=0))
        metadata_list.append(np.stack(metadata, axis=0))
        plane_ids_list.append(np.array(plane_ids, dtype=np.int64))
        labels.append(label)

    patches_arr = np.stack(patches_list, axis=0).astype(np.float32)
    metadata_arr = np.stack(metadata_list, axis=0).astype(np.float32)
    plane_ids_arr = np.stack(plane_ids_list, axis=0).astype(np.int64)
    labels_arr = np.array(labels, dtype=np.int64)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        cache_path,
        patches=patches_arr,
        metadata=metadata_arr,
        plane_ids=plane_ids_arr,
        labels=labels_arr,
    )
    return patches_arr, metadata_arr, plane_ids_arr, labels_arr


def compute_embeddings(
    model,
    patches: np.ndarray,
    metadata: np.ndarray,
    plane_ids: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    ds = TensorDataset(
        torch.from_numpy(patches),
        torch.from_numpy(metadata),
        torch.from_numpy(plane_ids),
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    model.eval()
    reps = []
    with torch.no_grad():
        for batch in loader:
            p, m, pid = [x.to(device) for x in batch]
            encoded = model.encode(p, m, pid, mask=None)
            rep = encoded.mean(dim=1).cpu().numpy()
            reps.append(rep)
    return np.concatenate(reps, axis=0)


def train_linear_probe(
    train_emb: np.ndarray,
    train_labels: np.ndarray,
    test_emb: np.ndarray,
    test_labels: np.ndarray,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
) -> Tuple[float, float]:
    num_classes = int(max(train_labels.max(), test_labels.max()) + 1)
    model = nn.Linear(train_emb.shape[1], num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    train_ds = TensorDataset(
        torch.from_numpy(train_emb).float(),
        torch.from_numpy(train_labels).long(),
    )
    test_ds = TensorDataset(
        torch.from_numpy(test_emb).float(),
        torch.from_numpy(test_labels).long(),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    last_train_acc = 0.0
    last_test_acc = 0.0
    for _ in range(epochs):
        model.train()
        total = 0
        correct = 0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += x.size(0)
            correct += int((logits.argmax(dim=1) == y).sum().item())
        last_train_acc = correct / max(1, total)

        model.eval()
        total = 0
        correct = 0
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device)
                y = y.to(device)
                logits = model(x)
                total += x.size(0)
                correct += int((logits.argmax(dim=1) == y).sum().item())
        last_test_acc = correct / max(1, total)

    return last_train_acc, last_test_acc
