from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.pose_dynamics.data.track_projection_dataset import (
    ProjectionSample,
    TrackProjectionDataset,
)
from experiments.pose_dynamics.losses import compute_losses
from experiments.pose_dynamics.models.factory import build_model
from experiments.pose_dynamics.utils.config import load_config
from experiments.pose_dynamics.visualization import export_batch_visualizations


def _collate_samples(batch: List[ProjectionSample]):
    if not batch:
        return batch
    reps = batch[0].inputs.keys()
    inputs = {rep: torch.stack([b.inputs[rep] for b in batch], dim=0) for rep in reps}
    past_centers = torch.stack([b.past_centers for b in batch], dim=0)
    future_centers = torch.stack([b.future_centers for b in batch], dim=0)
    dt = torch.stack([b.dt for b in batch], dim=0)
    intrinsics = torch.stack([b.intrinsics for b in batch], dim=0)
    camera_pose = torch.stack([b.camera_pose for b in batch], dim=0)
    frame_keys = [b.frame_key for b in batch]
    frame_times = [b.frame_time_s for b in batch]
    track_ids = [b.track_id for b in batch]
    return type(
        "Batch",
        (),
        {
            "inputs": inputs,
            "past_centers": past_centers,
            "future_centers": future_centers,
            "dt": dt,
            "intrinsics": intrinsics,
            "camera_pose": camera_pose,
            "frame_keys": frame_keys,
            "frame_times": frame_times,
            "track_ids": track_ids,
        },
    )


def _read_split_file(path: Path) -> List[str]:
    items: List[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        items.append(line.strip("/"))
    return items


def _build_dataset(data_cfg: Dict, folders: List[str], *, max_samples_key: str, seed_offset: int):
    return TrackProjectionDataset(
        images_root=Path(data_cfg["images_root"]),
        labels_root=Path(data_cfg["labels_root"]),
        representations=data_cfg["representations"],
        image_size=tuple(data_cfg["image_size"]),
        history_steps=int(data_cfg.get("history_steps", 12)),
        future_steps=int(data_cfg["future_steps"]),
        stride=int(data_cfg["stride"]),
        frame_size=tuple(data_cfg["frame_size"]),
        intrinsics=tuple(data_cfg["camera"]["intrinsics"]),
        camera_pose=tuple(data_cfg["camera"]["pose"]),
        folders=folders,
        labels_subdir=data_cfg.get("labels_subdir", "Event_YOLO"),
        tracks_file=data_cfg.get("tracks_file", "cleaned_tracks.txt"),
        label_time_unit=float(data_cfg.get("label_time_unit", 1e-6)),
        track_time_unit=float(data_cfg.get("track_time_unit", 1.0)),
        time_align=data_cfg.get("time_align", "start"),
        max_tracks=data_cfg.get("max_tracks"),
        max_samples=data_cfg.get(max_samples_key),
        seed=int(data_cfg.get("seed", 1234)) + seed_offset,
        cache_dir=Path(data_cfg["cache_dir"]) if data_cfg.get("cache_dir") else None,
    )


def _init_metric_sums(future_steps: int):
    return {
        "loss": 0.0,
        "center_l1": 0.0,
        "pose_reg": 0.0,
        "intr_reg": 0.0,
        "acc_reg": 0.0,
        "endpoint_l2": 0.0,
        "oob_step_rate": 0.0,
        "oob_traj_rate": 0.0,
        "horizon_l1": torch.zeros(future_steps, dtype=torch.float64),
        "horizon_l2": torch.zeros(future_steps, dtype=torch.float64),
    }


def _update_metric_sums(sums, metrics: Dict[str, float], pred, target):
    for key in ("loss", "center_l1", "pose_reg", "intr_reg", "acc_reg"):
        sums[key] += metrics[key]

    abs_err = (pred["pred_centers"] - target).abs()
    l1_per_step = abs_err.mean(dim=(0, 2)).detach().cpu().to(torch.float64)
    l2_per_step = (
        (pred["pred_centers"] - target).pow(2).sum(dim=-1).sqrt().mean(dim=0).detach().cpu().to(torch.float64)
    )
    endpoint_l2 = float(l2_per_step[-1].item())

    oob_mask = ((pred["pred_centers"] < 0.0) | (pred["pred_centers"] > 1.0)).any(dim=-1)
    oob_step_rate = float(oob_mask.float().mean().item())
    oob_traj_rate = float(oob_mask.any(dim=1).float().mean().item())

    sums["horizon_l1"] += l1_per_step
    sums["horizon_l2"] += l2_per_step
    sums["endpoint_l2"] += endpoint_l2
    sums["oob_step_rate"] += oob_step_rate
    sums["oob_traj_rate"] += oob_traj_rate


def _finalize_metric_sums(sums, count: int):
    if count == 0:
        out = {k: float("nan") for k in ("loss", "center_l1", "pose_reg", "intr_reg", "acc_reg", "endpoint_l2", "oob_step_rate", "oob_traj_rate")}
        out["horizon_l1"] = []
        out["horizon_l2"] = []
        return out

    out = {
        "loss": sums["loss"] / count,
        "center_l1": sums["center_l1"] / count,
        "pose_reg": sums["pose_reg"] / count,
        "intr_reg": sums["intr_reg"] / count,
        "acc_reg": sums["acc_reg"] / count,
        "endpoint_l2": sums["endpoint_l2"] / count,
        "oob_step_rate": sums["oob_step_rate"] / count,
        "oob_traj_rate": sums["oob_traj_rate"] / count,
        "horizon_l1": (sums["horizon_l1"] / count).tolist(),
        "horizon_l2": (sums["horizon_l2"] / count).tolist(),
    }
    return out


def _summarize_horizons(values: List[float]) -> str:
    if not values:
        return "h[n/a]"
    first = values[0]
    mid = values[len(values) // 2]
    last = values[-1]
    return f"h1 {first:.5f} h{len(values)//2 + 1} {mid:.5f} h{len(values)} {last:.5f}"


def _sequence_key(frame_key: str) -> str:
    if "/" not in frame_key:
        return frame_key
    return frame_key.split("/", 1)[0]


def _collect_sequence_metrics(seq_sums, frame_keys: List[str], pred_centers: torch.Tensor, future_centers: torch.Tensor):
    seq_err = (pred_centers - future_centers).abs().mean(dim=(1, 2)).detach().cpu().tolist()
    for frame_key, err in zip(frame_keys, seq_err):
        seq_key = _sequence_key(frame_key)
        seq_sums[seq_key][0] += float(err)
        seq_sums[seq_key][1] += 1


def _format_sequence_summary(seq_sums, limit: int = 3) -> str:
    if not seq_sums:
        return "seq[n/a]"
    ranked = sorted(
        ((total / count, key, count) for key, (total, count) in seq_sums.items() if count > 0),
        reverse=True,
    )
    parts = [f"{key}:{err:.5f} ({count})" for err, key, count in ranked[:limit]]
    return "worst_seq " + " | ".join(parts)


def _train_epoch(model, loader, optimizer, device, train_cfg):
    model.train()
    future_steps = int(train_cfg["future_steps"])
    sums = _init_metric_sums(future_steps)
    count = 0
    for step, batch in enumerate(loader):
        inputs = {k: v.to(device) for k, v in batch.inputs.items()}
        past_centers = batch.past_centers.to(device)
        future_centers = batch.future_centers.to(device)
        dt = batch.dt.to(device)
        intrinsics = batch.intrinsics.to(device)
        camera_pose = batch.camera_pose.to(device)

        pred = model(inputs, past_centers, intrinsics, camera_pose, dt)
        loss, metrics = compute_losses(
            pred,
            future_centers,
            center_weight=float(train_cfg.get("center_weight", 1.0)),
            pose_reg_weight=float(train_cfg.get("pose_reg_weight", 1.0e-3)),
            intr_reg_weight=float(train_cfg.get("intr_reg_weight", 1.0e-3)),
            acc_reg_weight=float(train_cfg.get("acc_reg_weight", 1.0e-4)),
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _update_metric_sums(sums, metrics, pred, future_centers)
        count += 1

        if step % int(train_cfg.get("log_every", 20)) == 0:
            print(
                f"step {step} "
                f"loss {metrics['loss']:.5f} "
                f"center {metrics['center_l1']:.5f} "
                f"pose_reg {metrics['pose_reg']:.5f} "
                f"intr_reg {metrics['intr_reg']:.5f} "
                f"acc_reg {metrics['acc_reg']:.5f} "
                f"end_l2 {float(((pred['pred_centers'][:, -1] - future_centers[:, -1]).pow(2).sum(dim=-1).sqrt().mean()).item()):.5f} "
                f"oob {float((((pred['pred_centers'] < 0.0) | (pred['pred_centers'] > 1.0)).any(dim=-1).float().mean()).item()):.3f}"
            )

    return _finalize_metric_sums(sums, count)


@torch.no_grad()
def _eval_epoch(model, loader, device, train_cfg):
    model.eval()
    future_steps = int(train_cfg["future_steps"])
    sums = _init_metric_sums(future_steps)
    seq_sums = defaultdict(lambda: [0.0, 0])
    count = 0
    for batch in loader:
        inputs = {k: v.to(device) for k, v in batch.inputs.items()}
        past_centers = batch.past_centers.to(device)
        future_centers = batch.future_centers.to(device)
        dt = batch.dt.to(device)
        intrinsics = batch.intrinsics.to(device)
        camera_pose = batch.camera_pose.to(device)

        pred = model(inputs, past_centers, intrinsics, camera_pose, dt)
        _, metrics = compute_losses(
            pred,
            future_centers,
            center_weight=float(train_cfg.get("center_weight", 1.0)),
            pose_reg_weight=float(train_cfg.get("pose_reg_weight", 1.0e-3)),
            intr_reg_weight=float(train_cfg.get("intr_reg_weight", 1.0e-3)),
            acc_reg_weight=float(train_cfg.get("acc_reg_weight", 1.0e-4)),
        )
        _update_metric_sums(sums, metrics, pred, future_centers)
        _collect_sequence_metrics(seq_sums, batch.frame_keys, pred["pred_centers"], future_centers)
        count += 1
    out = _finalize_metric_sums(sums, count)
    out["sequence_summary"] = _format_sequence_summary(seq_sums)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    train_cfg = cfg["train"]
    train_cfg = dict(train_cfg)
    train_cfg["future_steps"] = int(data_cfg["future_steps"])

    split_files = data_cfg.get("split_files")
    if not split_files:
        raise ValueError("data.split_files is required for this experiment.")
    train_folders = _read_split_file(Path(split_files["train"]))
    val_folders = _read_split_file(Path(split_files["val"]))

    train_set = _build_dataset(data_cfg, train_folders, max_samples_key="max_samples_train", seed_offset=0)
    val_set = _build_dataset(data_cfg, val_folders, max_samples_key="max_samples_val", seed_offset=1)
    print(f"Train samples: {len(train_set)}")
    print(f"Val samples: {len(val_set)}")

    train_loader = DataLoader(
        train_set,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=True,
        num_workers=int(train_cfg.get("num_workers", 0)),
        collate_fn=_collate_samples,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=False,
        num_workers=int(train_cfg.get("num_workers", 0)),
        collate_fn=_collate_samples,
    )

    device = torch.device(train_cfg.get("device", "cpu") if torch.cuda.is_available() else "cpu")
    model = build_model(cfg, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )

    ckpt_dir = Path(train_cfg.get("checkpoint_dir", "outputs/pose_dynamics_ckpt"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_every = int(train_cfg.get("checkpoint_every", 1))
    best_val = None
    start_epoch = 0

    resume_from = train_cfg.get("resume_from")
    if resume_from:
        resume_path = Path(resume_from)
        if resume_path.exists():
            state = torch.load(resume_path, map_location=device)
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optim"])
            start_epoch = int(state.get("epoch", 0)) + 1
            best_val = state.get("best_val")
            print(f"Resumed checkpoint {resume_path} at epoch {start_epoch}")

    epochs = int(train_cfg["epochs"])
    for epoch in range(start_epoch, epochs):
        print(f"Epoch {epoch}/{epochs - 1}")
        train_metrics = _train_epoch(model, train_loader, optimizer, device, train_cfg)
        val_metrics = _eval_epoch(model, val_loader, device, train_cfg)
        print(
            "train "
            f"loss {train_metrics['loss']:.5f} center {train_metrics['center_l1']:.5f} "
            f"pose_reg {train_metrics['pose_reg']:.5f} intr_reg {train_metrics['intr_reg']:.5f} "
            f"acc_reg {train_metrics['acc_reg']:.5f} end_l2 {train_metrics['endpoint_l2']:.5f} "
            f"oob_step {train_metrics['oob_step_rate']:.3f} oob_traj {train_metrics['oob_traj_rate']:.3f}"
        )
        print(
            "val   "
            f"loss {val_metrics['loss']:.5f} center {val_metrics['center_l1']:.5f} "
            f"pose_reg {val_metrics['pose_reg']:.5f} intr_reg {val_metrics['intr_reg']:.5f} "
            f"acc_reg {val_metrics['acc_reg']:.5f} end_l2 {val_metrics['endpoint_l2']:.5f} "
            f"oob_step {val_metrics['oob_step_rate']:.3f} oob_traj {val_metrics['oob_traj_rate']:.3f}"
        )
        print(f"train { _summarize_horizons(train_metrics['horizon_l1']) }")
        print(f"val   { _summarize_horizons(val_metrics['horizon_l1']) }")
        print(f"val   {val_metrics['sequence_summary']}")

        val_loss = val_metrics["loss"]
        if val_loss == val_loss and (best_val is None or val_loss < best_val):
            best_val = val_loss
            torch.save(
                {
                    "model": model.state_dict(),
                    "optim": optimizer.state_dict(),
                    "epoch": epoch,
                    "best_val": best_val,
                    "config": cfg,
                },
                ckpt_dir / "best.pt",
            )

        if ckpt_every > 0 and (epoch + 1) % ckpt_every == 0:
            torch.save(
                {
                    "model": model.state_dict(),
                    "optim": optimizer.state_dict(),
                    "epoch": epoch,
                    "best_val": best_val,
                    "config": cfg,
                },
                ckpt_dir / f"epoch_{epoch:03d}.pt",
            )

        vis_every = int(train_cfg.get("vis_every", 0))
        if vis_every > 0 and (epoch + 1) % vis_every == 0:
            vis_dir = Path(train_cfg.get("vis_output_dir", "outputs/pose_dynamics_vis"))
            export_batch_visualizations(
                model=model,
                loader=val_loader,
                device=device,
                output_dir=vis_dir / f"epoch_{epoch:03d}",
                rep=train_cfg.get("vis_rep", data_cfg["representations"][0]),
                max_samples=int(train_cfg.get("vis_samples", 8)),
            )


if __name__ == "__main__":
    main()
