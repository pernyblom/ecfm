from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.pose_dynamics.data.track_projection_dataset import (
    ProjectionSample,
    TrackProjectionDataset,
)
from experiments.pose_dynamics.models.factory import build_model
from experiments.pose_dynamics.utils.config import load_config


def _collate_samples(batch: List[ProjectionSample]):
    if not batch:
        return batch
    reps = batch[0].inputs.keys()
    inputs = {rep: torch.stack([b.inputs[rep] for b in batch], dim=0) for rep in reps}
    past_centers = torch.stack([b.past_centers for b in batch], dim=0)
    past_sizes = torch.stack([b.past_sizes for b in batch], dim=0)
    future_centers = torch.stack([b.future_centers for b in batch], dim=0)
    future_sizes = torch.stack([b.future_sizes for b in batch], dim=0)
    dt = torch.stack([b.dt for b in batch], dim=0)
    intrinsics = torch.stack([b.intrinsics for b in batch], dim=0)
    camera_pose = torch.stack([b.camera_pose for b in batch], dim=0)
    frame_keys = [b.frame_key for b in batch]
    track_ids = [b.track_id for b in batch]
    return type(
        "Batch",
        (),
        {
            "inputs": inputs,
            "past_centers": past_centers,
            "past_sizes": past_sizes,
            "future_centers": future_centers,
            "future_sizes": future_sizes,
            "dt": dt,
            "intrinsics": intrinsics,
            "camera_pose": camera_pose,
            "frame_keys": frame_keys,
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


def _build_dataset(data_cfg: Dict, folders: List[str], *, max_samples: int | None, seed_offset: int):
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
        max_samples=max_samples,
        seed=int(data_cfg.get("seed", 1234)) + seed_offset,
        cache_dir=Path(data_cfg["cache_dir"]) if data_cfg.get("cache_dir") else None,
    )


def _tensor_to_image(t: torch.Tensor) -> Image.Image:
    arr = t.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    arr = (arr * 255.0).astype(np.uint8)
    return Image.fromarray(arr)


def _draw_polyline(draw: ImageDraw.ImageDraw, points: np.ndarray, size: tuple[int, int], color: tuple[int, int, int], width: int) -> None:
    if len(points) == 0:
        return
    xy = []
    for x, y in points:
        xy.append((float(x) * size[0], float(y) * size[1]))
    if len(xy) == 1:
        px, py = xy[0]
        r = max(2, width + 1)
        draw.ellipse((px - r, py - r, px + r, py + r), fill=color)
        return
    draw.line(xy, fill=color, width=width)
    for px, py in xy:
        r = max(2, width)
        draw.ellipse((px - r, py - r, px + r, py + r), fill=color)


def render_trajectory_image(
    backdrop: torch.Tensor,
    past_centers: torch.Tensor,
    future_centers: torch.Tensor,
    pred_centers: torch.Tensor,
) -> Image.Image:
    img = _tensor_to_image(backdrop).convert("RGB")
    draw = ImageDraw.Draw(img)
    size = img.size

    past_np = past_centers.detach().cpu().numpy()
    future_np = future_centers.detach().cpu().numpy()
    pred_np = pred_centers.detach().cpu().numpy()

    _draw_polyline(draw, past_np, size, (64, 160, 255), 3)
    _draw_polyline(draw, future_np, size, (0, 220, 80), 3)
    _draw_polyline(draw, pred_np, size, (255, 210, 0), 3)

    if len(past_np) > 0:
        x, y = past_np[-1]
        px = float(x) * size[0]
        py = float(y) * size[1]
        r = 5
        draw.ellipse((px - r, py - r, px + r, py + r), outline=(255, 255, 255), width=2)

    return img


@torch.no_grad()
def export_batch_visualizations(
    *,
    model,
    loader,
    device: torch.device,
    output_dir: Path,
    rep: str,
    max_samples: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    batch = next(iter(loader), None)
    if batch is None:
        return

    inputs = {k: v.to(device) for k, v in batch.inputs.items()}
    pred = model(
        inputs,
        batch.past_centers.to(device),
        batch.intrinsics.to(device),
        batch.camera_pose.to(device),
        batch.dt.to(device),
    )
    pred_centers = pred["pred_centers"].cpu()

    n = min(max_samples, pred_centers.shape[0])
    for idx in range(n):
        backdrop = batch.inputs[rep][idx]
        img = render_trajectory_image(
            backdrop=backdrop,
            past_centers=batch.past_centers[idx],
            future_centers=batch.future_centers[idx],
            pred_centers=pred_centers[idx],
        )
        frame_key = batch.frame_keys[idx].replace("/", "_")
        track_id = batch.track_ids[idx]
        out_path = output_dir / f"{idx:03d}_track_{track_id}_{frame_key}.png"
        img.save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/pose_dynamics_vis_manual"))
    parser.add_argument("--rep", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    split_files = data_cfg.get("split_files")
    if not split_files:
        raise ValueError("data.split_files is required.")

    split_key = "train" if args.split == "train" else "val"
    folders = _read_split_file(Path(split_files[split_key]))
    dataset = _build_dataset(data_cfg, folders, max_samples=args.num_samples, seed_offset=7)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=_collate_samples,
    )

    device = torch.device(cfg["train"].get("device", "cpu") if torch.cuda.is_available() else "cpu")
    model = build_model(cfg, device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()

    rep = args.rep or data_cfg["representations"][0]
    export_batch_visualizations(
        model=model,
        loader=loader,
        device=device,
        output_dir=args.output_dir,
        rep=rep,
        max_samples=args.num_samples,
    )


if __name__ == "__main__":
    main()
