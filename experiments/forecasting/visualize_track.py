from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict

import torch
import PIL

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.forecasting.data.track_dataset import TrackForecastDataset
from experiments.forecasting.models.fusion import MultiRepForecast
from experiments.forecasting.models.transformer import MultiRepTransformer
from experiments.forecasting.utils.config import load_config
from experiments.forecasting.train import _render_forecast_image


def _build_model(cfg: dict, device: torch.device) -> torch.nn.Module:
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    model_type = model_cfg.get("type", "gru")
    if model_type == "transformer":
        model = MultiRepTransformer(
            reps=data_cfg["representations"],
            cnn_channels=model_cfg["cnn_channels"],
            feature_dim=model_cfg["feature_dim"],
            d_model=model_cfg.get("d_model", 256),
            nhead=model_cfg.get("nhead", 4),
            num_encoder_layers=model_cfg.get("num_encoder_layers", 4),
            num_decoder_layers=model_cfg.get("num_decoder_layers", 4),
            dim_feedforward=model_cfg.get("dim_feedforward", 512),
            dropout=model_cfg.get("dropout", 0.1),
            use_past_boxes=model_cfg["use_past_boxes"],
            past_steps=data_cfg["past_steps"],
            future_steps=data_cfg["future_steps"],
            predict_past=bool(model_cfg.get("predict_past", False)),
            pos_encoding=model_cfg.get("pos_encoding", "learned"),
        ).to(device)
    else:
        model = MultiRepForecast(
            reps=data_cfg["representations"],
            cnn_channels=model_cfg["cnn_channels"],
            feature_dim=model_cfg["feature_dim"],
            use_past_boxes=model_cfg["use_past_boxes"],
            rnn_hidden=model_cfg["rnn_hidden"],
            rnn_layers=model_cfg["rnn_layers"],
            future_steps=data_cfg["future_steps"],
        ).to(device)
    return model


def _read_track_ids(path: Path) -> list[int]:
    if not path.exists():
        return []
    ids: set[int] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            ids.add(int(parts[1]))
        except ValueError:
            continue
    return sorted(ids)


def _to_gif(frames, out_path: Path, duration_ms: int) -> None:
    from PIL import Image

    if not frames:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
    )


def _parse_frame_key(key: str) -> tuple[str, str]:
    if "/" in key:
        folder, stem = key.split("/", 1)
        return folder, stem
    return "", key


def _load_backdrop(
    images_root: Path,
    frame_key: str,
    rep: str,
    frame_size: tuple[int, int],
) -> "PIL.Image.Image | None":
    from PIL import Image

    folder, stem = _parse_frame_key(frame_key)
    base = images_root / folder if folder else images_root
    for ext in (".png", ".jpg", ".jpeg"):
        img_path = base / f"{stem}_{rep}{ext}"
        if img_path.exists():
            img = Image.open(img_path).convert("RGB")
            return img.resize(frame_size, resample=Image.BILINEAR)
    return None

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize forecasting for a specific track ID as a GIF."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--track-id", type=int, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/track_vis.gif"))
    parser.add_argument("--max-frames", type=int, default=200)
    parser.add_argument("--duration-ms", type=int, default=120)
    parser.add_argument(
        "--backdrop-rep",
        type=str,
        default=None,
        help="Optional representation to use as backdrop (e.g., rgb, cstr2). Overrides config.",
    )
    parser.add_argument(
        "--backdrop-index",
        type=int,
        default=None,
        help="Index into the window frame_keys for the backdrop (default: last past).",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "val", "all"],
        default="all",
        help="Which split folders to use when split_files are configured (default: all).",
    )
    parser.add_argument(
        "--track-folder",
        type=str,
        default=None,
        help="Optional folder override for the track ID (e.g., '4' for track 4003).",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    frame_size = data_cfg.get("frame_size")
    if frame_size is None:
        raise ValueError("data.frame_size must be set for visualization.")
    frame_size_t = (int(frame_size[0]), int(frame_size[1]))

    folders = None
    split_files = data_cfg.get("split_files")
    if args.track_folder is not None:
        folders = [args.track_folder]
    elif split_files:
        if args.split == "all":
            train_path = Path(split_files["train"])
            val_path = Path(split_files["val"])
            train_folders = [
                line.strip("/").strip()
                for line in train_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            val_folders = [
                line.strip("/").strip()
                for line in val_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            folders = sorted(set(train_folders + val_folders))
        else:
            split_path = Path(split_files["train" if args.split == "train" else "val"])
            folders = [
                line.strip("/").strip()
                for line in split_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

    dataset = TrackForecastDataset(
        images_root=Path(data_cfg["images_root"]),
        labels_root=Path(data_cfg["labels_root"]),
        representations=data_cfg["representations"],
        past_steps=data_cfg["past_steps"],
        future_steps=data_cfg["future_steps"],
        stride=data_cfg["stride"],
        image_size=tuple(data_cfg["image_size"]),
        folders=folders,
        labels_subdir=data_cfg.get("labels_subdir", "Event_YOLO"),
        tracks_file=data_cfg.get("tracks_file", "tracks.txt"),
        label_time_unit=float(data_cfg.get("label_time_unit", 1e-6)),
        track_time_unit=float(data_cfg.get("track_time_unit", 1.0)),
        time_align=data_cfg.get("time_align", "start"),
        frame_size=tuple(data_cfg["frame_size"]) if data_cfg.get("frame_size") else None,
        max_tracks=None,
        max_samples=None,
        seed=int(data_cfg.get("seed", 123)),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _build_model(cfg, device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()

    frames = []
    count = 0
    with torch.no_grad():
        for sample in dataset:
            print(count, sample.track_id)
            if sample.track_id != args.track_id:
                print()
                continue
            inputs = {r: sample.inputs[r].unsqueeze(0).to(device) for r in sample.inputs}
            past_boxes = sample.past_boxes.unsqueeze(0).to(device)
            future = sample.future_boxes.unsqueeze(0).to(device)
            pred = model(inputs, past_boxes)
            pred_future = (
                pred[:, data_cfg["past_steps"] :]
                if pred.shape[1] == data_cfg["past_steps"] + data_cfg["future_steps"]
                else pred
            )
            if args.backdrop_index is None:
                idx = data_cfg["past_steps"] - 1
            else:
                idx = int(args.backdrop_index)
            if idx < 0:
                idx = len(sample.frame_keys) + idx
            idx = max(0, min(idx, len(sample.frame_keys) - 1))
            frame_key = sample.frame_keys[idx]
            backdrop_rep = args.backdrop_rep or cfg.get("train", {}).get("vis_backdrop_rep")
            backdrop = (
                _load_backdrop(
                    Path(data_cfg["images_root"]),
                    frame_key,
                    backdrop_rep,
                    frame_size_t,
                )
                if backdrop_rep
                else None
            )
            img = _render_forecast_image(
                past_boxes[0].cpu(),
                pred_future[0].cpu(),
                future[0].cpu(),
                frame_size_t,
                backdrop=backdrop,
            )
            frames.append(img)
            count += 1
            if count >= args.max_frames:
                break

    if not frames:
        try:
            available = sorted({s.track_id for s in dataset})
        except Exception:
            available = []
        if not available:
            # fallback: list raw track IDs from tracks file
            tracks_path = (
                Path(data_cfg["labels_root"])
                / (folders[0] if folders else "")
                / data_cfg.get("tracks_file", "tracks.txt")
            )
            available = _read_track_ids(tracks_path)
        if available:
            preview = ", ".join(str(t) for t in available[:50])
            more = f" (and {len(available) - 50} more)" if len(available) > 50 else ""
            raise RuntimeError(
                f"No samples found for track_id={args.track_id}. "
                f"Available track IDs: {preview}{more}"
            )
        raise RuntimeError(
            f"No samples found for track_id={args.track_id}. "
            "Dataset yielded zero samples (check that rendered images exist and "
            "max_samples_* isn't too small)."
        )

    _to_gif(frames, args.output, args.duration_ms)
    print(f"Wrote {args.output} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
