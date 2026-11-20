"""Fit the scene optics and pose model to 2D tracklet reprojection."""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.nn import functional as F
from torch.utils.data import DataLoader

from experiments.kalman_ml_forecasting.models.kalman_residual import last_four_constant_velocity_forecast
from experiments.kalman_ml_forecasting.utils.config import load_config

from .data import PoseTracklets, load_scenes, make_splits, to_device
from .model import RelativePoseModel, project_body


DEFAULT_CONFIG = Path(__file__).with_name("configs") / "base.yaml"


def reprojection_loss(result: dict, batch: dict, frame: torch.Tensor, config: dict) -> torch.Tensor:
    target = torch.cat((batch["past"][:, -1:], batch["future"]), dim=1)
    target = target * frame.repeat(2)
    projected = result["boxes_px"]
    center = F.smooth_l1_loss(projected[..., :2] / config["center_scale_px"],
                              target[..., :2] / config["center_scale_px"], reduction="none").mean(-1)
    size = F.smooth_l1_loss(projected[..., 2:].clamp_min(1).log(), target[..., 2:].clamp_min(1).log(),
                           beta=0.2, reduction="none").mean(-1)
    lo = target[..., :2] - target[..., 2:] / 2
    hi = target[..., :2] + target[..., 2:] / 2
    visible = ((lo > 1) & (hi < frame - 1)).all(-1)
    future_size = (size[:, 1:] * visible[:, 1:]).sum() / visible[:, 1:].sum().clamp_min(1)
    anchor = center[:, 0].mean() + config["size_weight"] * size[:, 0].mean()
    prior = result["camera_offsets"].square().mean()
    controls = result["controls"].tanh().square().mean()
    tilt = (1 - (result["trajectory"].up * result["gravity_up"][:, None]).sum(-1)).mean()
    near_penalty = F.relu(1 - result["rollout"][..., 2]).square().mean()
    return (center[:, 1:].mean() + config["size_weight"] * future_size + config["anchor_weight"] * anchor
            + config["camera_prior_weight"] * prior + config["control_prior_weight"] * controls
            + config.get("tilt_prior_weight", 0) * tilt
            + config["behind_camera_weight"] * near_penalty)


def metrics(result: dict, batch: dict, model: RelativePoseModel, *, compare: bool) -> dict[str, torch.Tensor]:
    frame = model.cameras.frame_size
    target = batch["future"] * frame.repeat(2)
    boxes = result["boxes_px"][:, 1:]
    error = torch.linalg.vector_norm(boxes[..., :2] - target[..., :2], dim=-1)
    tilt = torch.acos((result["up"] * result["gravity_up"]).sum(-1).clamp(-1, 1)).rad2deg()
    output = {"ade_px": error.mean(-1), "fde_px": error[:, -1], "tilt_deg": tilt,
              "tilt_rate_deg_s": result["angular_velocity"].norm(dim=-1).rad2deg(),
              "speed_diameters_s": result["velocity"].norm(dim=-1),
              "accel_diameters_s2": result["acceleration"].norm(dim=-1),
              "log_extent_mae": (boxes[..., 2:].clamp_min(1).log() - target[..., 2:].clamp_min(1).log()).abs().mean((1, 2)),
              "min_depth_diameters": result["rollout"][..., 2].amin(-1)}
    if compare:
        last4 = last_four_constant_velocity_forecast(batch["past"], batch["past_time"], batch["future_time"])
        linear = torch.linalg.vector_norm(last4[..., :2] * frame - target[..., :2], dim=-1)
        cv = project_body(result["cv_rollout"][:, 1:], result["up"], result, model.half_height)
        cv = torch.linalg.vector_norm(cv[..., :2] - target[..., :2], dim=-1)
        output.update(last4_ade_px=linear.mean(-1), last4_fde_px=linear[:, -1], cv_3d_ade_px=cv.mean(-1))
        if model.encoders and model.event_dropout:
            masked = model(batch, mask_events=True)["boxes_px"][:, 1:, :2]
            output["masked_event_ade_px"] = torch.linalg.vector_norm(masked - target[..., :2], dim=-1).mean(-1)
    return output


def epoch(model, loader, device, config, optimizer=None, *, number: int):
    training = optimizer is not None
    model.train(training)
    train_cfg = config["train"]
    totals = defaultdict(float)
    scene_totals = defaultdict(lambda: defaultdict(float))
    count = 0
    started = time.perf_counter()
    for step, raw in enumerate(loader, 1):
        batch = to_device(raw, device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            result = model(batch)
            loss = reprojection_loss(result, batch, model.cameras.frame_size, config["loss"])
        if training:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg["grad_clip"]), error_if_nonfinite=True)
            optimizer.step()
        with torch.no_grad():
            values = metrics(result, batch, model, compare=not training)
            names = list(values)
            values_cpu = torch.stack(list(values.values()), dim=1).cpu().numpy()
            for name, total in zip(names, values_cpu.sum(0, dtype=np.float64)):
                totals[name] += float(total)
            if not training:
                ids = raw["scene"].numpy()
                for scene in np.unique(ids):
                    selected = values_cpu[ids == scene]
                    rows = scene_totals[int(scene)]
                    rows["samples"] += len(selected)
                    for name, total in zip(names, selected.sum(0, dtype=np.float64)):
                        rows[name] += float(total)
            total_loss = float(loss.detach())
            totals["loss"] += total_loss * len(values_cpu)
            count += len(values_cpu)
        if step == 1 or step % train_cfg["log_every"] == 0:
            split = "train" if training else "val"
            print(f"epoch {number:02d} {split} {step}/{len(loader)} "
                  f"loss={totals['loss'] / count:.4f} ADE={totals['ade_px'] / count:.2f}px "
                  f"{count / (time.perf_counter() - started):.1f} tracklets/s", flush=True)
    summary = {key: value / count for key, value in totals.items()}
    if scene_totals:
        summary["by_scene"] = {index: {key: value / rows["samples"] for key, value in rows.items() if key != "samples"}
                               | {"samples": int(rows["samples"])} for index, rows in scene_totals.items()}
    summary["seconds"] = time.perf_counter() - started
    return summary


def loader_for(dataset, config, device, *, training: bool, seed: int):
    workers = int(config["num_workers"])
    return DataLoader(dataset, shuffle=training, batch_size=int(config["batch_size"]), num_workers=workers,
                      persistent_workers=workers > 0, pin_memory=device.type == "cuda",
                      generator=torch.Generator().manual_seed(seed))


@torch.no_grad()
def camera_report(model, scene_names, train_counts, val_counts) -> dict:
    fitted = model.cameras(torch.arange(len(scene_names), device=model.cameras.offsets.device))
    return {scene: {"focal_px": fitted["focal"][i].tolist(), "principal_point_px": fitted["center"][i].tolist(),
                    "camera_up": fitted["gravity_up"][i].tolist(), "fitted": bool(train_counts.get(scene)),
                    "train_samples": train_counts.get(scene, 0), "val_samples": val_counts.get(scene, 0)}
            for i, scene in enumerate(scene_names)}


def write_json(path: Path, value) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def save_checkpoint(path: Path, checkpoint: dict) -> None:
    temp = path.with_suffix(".tmp")
    torch.save(checkpoint, temp)
    temp.replace(path)


def train(config: dict) -> None:
    data, settings = config["data"], config["train"]
    output = Path(settings["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    history_path = output / "metrics.jsonl"
    if history_path.exists() or (output / "last.pt").exists() or (output / "best.pt").exists():
        raise FileExistsError(f"Choose a fresh train.output_dir instead of replacing the run in {output}")
    seed = int(settings["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(int(settings.get("torch_threads", 2)))
    torch.set_float32_matmul_precision("high")
    device = torch.device(settings["device"])
    scenes = load_scenes(Path(data["scene_manifest"]))
    splits = make_splits(data, scenes, seed)
    (output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    write_json(output / "splits.json", splits)
    print(f"{device} | {len(splits['train'])} train sequences, {len(splits['val'])} validation sequences, "
          f"{len(splits['excluded'])} excluded", flush=True)

    use_events = bool(config["model"]["use_events"])
    train_data = PoseTracklets(data, splits["train"], scenes, split="train", seed=seed, use_events=use_events)
    val_data = PoseTracklets(data, splits["val"], scenes, split="val", seed=seed + 1, use_events=use_events)
    unseen = val_data.count_by_scene.keys() - train_data.count_by_scene.keys()
    if unseen:
        raise ValueError(f"Validation scenes lack usable training tracklets: {sorted(unseen)}")
    splits["tracklets_by_sequence"] = {"train": train_data.count_by_folder, "val": val_data.count_by_folder}
    splits["tracklet_sha256"] = {"train": train_data.sample_digest, "val": val_data.sample_digest}
    write_json(output / "splits.json", splits)
    train_loader = loader_for(train_data, settings, device, training=True, seed=seed)
    val_loader = loader_for(val_data, settings, device, training=False, seed=seed + 1)
    model = RelativePoseModel(config, scenes["up_priors"]).to(device)
    camera_parameters = list(model.cameras.parameters())
    network_parameters = [p for name, p in model.named_parameters() if not name.startswith("cameras.")]
    optimizer = torch.optim.AdamW([
        {"params": network_parameters, "lr": settings["lr"], "weight_decay": settings["weight_decay"]},
        {"params": camera_parameters, "lr": settings["camera_lr"], "weight_decay": 0.0},
    ])
    epochs = int(settings["epochs"])
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=settings["lr"] * 0.08)
    best = float("inf")
    for number in range(1, epochs + 1):
        train_row = epoch(model, train_loader, device, config, optimizer, number=number)
        val_row = epoch(model, val_loader, device, config, number=number)
        val_row["by_scene"] = {scenes["names"][i]: v for i, v in val_row["by_scene"].items()}
        report = camera_report(model, scenes["names"], train_data.count_by_scene, val_data.count_by_scene)
        row = {"epoch": number, "train": train_row, "val": val_row, "lr": optimizer.param_groups[0]["lr"]}
        with history_path.open("a", encoding="utf-8") as history:
            history.write(json.dumps(row, allow_nan=False) + "\n")
        checkpoint = {"model": model.state_dict(), "epoch": number, "config": config, "scenes": scenes,
                      "splits": splits, "metrics": row, "cameras": report}
        save_checkpoint(output / "last.pt", checkpoint)
        if val_row["ade_px"] < best:
            best = val_row["ade_px"]
            save_checkpoint(output / "best.pt", checkpoint)
            write_json(output / "cameras.json", report)
        print(f"epoch {number:02d} train_loss={train_row['loss']:.4f} val_loss={val_row['loss']:.4f} "
              f"val_ADE={val_row['ade_px']:.2f}px val_FDE={val_row['fde_px']:.2f}px "
              f"last4_ADE={val_row['last4_ade_px']:.2f}px best={best:.2f}px", flush=True)
        schedule.step()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--forecast-ms", type=int, choices=(400, 800))
    parser.add_argument("--event-window-ms", type=float, help="Trailing event window, independent of box history")
    parser.add_argument("--cutout-px", type=int, help="Square crop size in representation pixels")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--device")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--samples-per-sequence", type=int, help="Override both sample caps for a short run")
    parser.add_argument("--output-dir")
    parser.add_argument("--history-only", action="store_true", help="Train the box-only control on the same tracklets")
    parser.add_argument("--anchor-only", action="store_true", help="Withhold box motion history from the event model")
    parser.add_argument("--all-sequences", action="store_true", help="Use every reviewed sequence and derive missing tracks from FRED coordinates")
    args = parser.parse_args()
    config = load_config(args.config)
    for key, value in (("epochs", args.epochs), ("device", args.device), ("num_workers", args.workers),
                       ("output_dir", args.output_dir)):
        if value is not None:
            config["train"][key] = value
    if args.forecast_ms is not None:
        config["data"]["forecast_ms"] = args.forecast_ms
    if args.event_window_ms is not None:
        config["data"]["event_window_ms"] = args.event_window_ms
    if args.cutout_px is not None:
        if args.cutout_px < 1:
            parser.error("Cutout size must be positive")
        config["data"].setdefault("spatial_cutout", {}).update(mode="fixed_pixels", size_px=[args.cutout_px] * 2)
    if args.samples_per_sequence is not None:
        config["data"]["max_samples_per_sequence"] = {"train": args.samples_per_sequence, "val": args.samples_per_sequence}
    if args.history_only:
        config["model"]["use_events"] = False
    if args.anchor_only:
        config["model"]["use_box_motion"] = False
        config["model"]["velocity_limit_px_s"] = 650
    if args.all_sequences:
        config["data"]["all_sequences"] = True
        config["data"].setdefault("coordinate_track_cache", "outputs/fred_relative_pose/coordinate_tracks")
        config["data"].setdefault("fallback_frame_timestamps", True)
    if config["train"]["epochs"] < 1 or args.samples_per_sequence is not None and args.samples_per_sequence < 1:
        parser.error("Epoch and sample counts must be positive")
    train(config)


if __name__ == "__main__":
    main()
