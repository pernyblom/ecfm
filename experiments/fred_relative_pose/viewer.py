"""Export an offline RGB and 3D viewer from a pose checkpoint."""

from __future__ import annotations

import argparse
import copy
import json
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from shutil import copyfile

import numpy as np
import torch
from PIL import Image
from torch.utils.data import default_collate

from ecfm.data.fred_dataset import FREDDataset, read_yolo

from .data import PoseTracklets, to_device
from .model import RelativePoseModel


@lru_cache(maxsize=64)
def padded_extent(rgb, folder: str, frame_size: tuple[int, int]):
    index = rgb._rgb_index(folder, "padded_rgb")
    extent = np.array([0, 0, *frame_size], dtype=np.float32)
    if not index:
        return extent
    with Image.open(index[len(index) // 2][1]) as image:
        visible = np.asarray(image.convert("L")) > 8
        size = np.array(image.size)
    columns = np.flatnonzero(visible.mean(axis=0) > 0.10)
    rows = np.flatnonzero(visible.mean(axis=1) > 0.10)
    if len(columns) and len(rows):
        x, y = columns[0], rows[0]
        w, h = columns[-1] + 1 - x, rows[-1] + 1 - y
        if min(w / size[0], h / size[1]) > 0.8:
            extent = np.array([x, y, w, h]) * np.tile(np.asarray(frame_size) / size, 2)
    return extent


def nearest_rgb_box(rgb, folder, time_s, target_px, frame_size):
    time_us = round(time_s * 1e6)
    frame = rgb.rgb_at(folder, time_us, "rgb", load=False) or rgb.rgb_at(folder, time_us, "padded_rgb", load=False)
    if frame is None:
        return None, float("inf")
    label = rgb.root / folder / "RGB_YOLO" / Path(frame["path"]).with_suffix(".txt").name
    boxes = read_yolo(label)["boxes"].numpy()
    if not len(boxes):
        return None, float("inf")
    x, y, w, h = padded_extent(rgb, folder, tuple(frame_size))
    centers = boxes[:, :2] * [w, h] + [x, y]
    distances = np.linalg.norm(centers - target_px[:2], axis=1)
    i = int(distances.argmin())
    return (boxes[i] * np.tile(frame_size, 2)).tolist(), float(distances[i])


def select_examples(dataset, rgb, *, count, time, track_id, frame_size, offset_s):
    samples = dataset.forecast.samples
    eligible = [i for i, sample in enumerate(samples) if track_id is None or sample["track_id"] == track_id]
    if time is not None:
        return sorted(eligible, key=lambda i: abs(samples[i]["anchor_time_s"] - time))[:count]
    sensor = np.asarray(frame_size)
    def visibility(index):
        past = samples[index]["past_boxes"]
        x, y, w, h = past[-1]
        size = min(w * sensor[0], h * sensor[1])
        margin = min(x, y, 1 - x, 1 - y)
        stable = np.abs(np.log(past[:, 2:] / past[-1, 2:])).mean()
        return min(size, 85) * min(margin / 0.20, 1) / (1 + stable)

    ranked = sorted(eligible, key=visibility, reverse=True)
    ranked_by_scene = defaultdict(list)
    for index in ranked:
        ranked_by_scene[dataset.scene_by_folder[samples[index]["folder"]]].append(index)
    # Prefer anchors with a nearby independent RGB annotation, irrespective of model error.
    scene_priority = {}
    for scene, indices in ranked_by_scene.items():
        candidates = []
        for index in indices[:max(16, count)]:
            sample = samples[index]
            target = sample["past_boxes"][-1] * np.tile(sensor, 2)
            _, delta = nearest_rgb_box(rgb, sample["folder"], sample["anchor_time_s"] + offset_s, target, frame_size)
            agreement = delta <= max(19, 0.38 * max(target[2:]))
            candidates.append((agreement, visibility(index), index))
        candidates.sort(reverse=True)
        indices[:len(candidates)] = [index for _, _, index in candidates]
        scene_priority[scene] = candidates[0][:2]
    scene_order = sorted(ranked_by_scene, key=scene_priority.__getitem__, reverse=True)
    picks = []
    for rank in range(count):
        for scene in scene_order:
            indices = ranked_by_scene[scene]
            if rank < len(indices):
                picks.append(indices[rank])
            if len(picks) == count:
                return picks
    return picks


def copy_rgb(rgb, output: Path, folder: str, kind: str, time_s: float):
    frame = rgb.rgb_at(folder, round(time_s * 1e6), kind, load=False)
    if frame is None:
        return None
    source = Path(frame["path"])
    destination = output / "images" / f"{folder}_{kind}_{source.name}"
    if not destination.exists():
        copyfile(source, destination)
    return {"url": destination.relative_to(output).as_posix(), "offset_ms": frame["offset_us"] / 1000}


def save_event_preview(inputs: dict, output: Path, i: int) -> str | None:
    if not inputs:
        return None
    events = next(iter(inputs.values())).abs().sum(0).numpy()
    positive = events[events > 0]
    if len(positive):
        events = (events / np.percentile(positive, 97)).clip(0, 1) ** 0.5
    path = output / "images" / f"events_{i:03d}.png"
    Image.fromarray(np.uint8(255 * events)).save(path)
    return path.relative_to(output).as_posix()


@torch.inference_mode()
def export(args) -> Path:
    torch.set_num_threads(2)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    data = copy.deepcopy(config["data"])
    if args.root:
        data["labels_root"] = str(args.root)
        if data["event_image_source"] == "indexed":
            data["events_root"] = str(args.root)
    scenes, splits = checkpoint["scenes"], checkpoint["splits"]
    split = args.split or ("train" if args.sequence in splits["train"] else "val")
    folders = splits[split]
    if args.sequence:
        if args.sequence not in folders:
            raise ValueError(f"Sequence {args.sequence} is outside the checkpoint's {split} sequences")
        folders = [args.sequence]
        data.setdefault("max_samples_per_sequence", {})[split] = None
    dataset = PoseTracklets(data, folders, scenes, split=split,
                            seed=config["train"]["seed"] + (split == "val"), use_events=config["model"]["use_events"])
    rgb_folders = sorted(dataset.count_by_folder, key=int)
    rgb = FREDDataset(data["labels_root"], sequences=rgb_folders, modalities=(), rgb_match="nearest",
                      rgb_tolerance_us=args.rgb_tolerance_ms * 1000)
    offset_s = args.rgb_offset_ms / 1000
    picks = select_examples(dataset, rgb, count=args.count, time=args.time, track_id=args.track_id,
                            frame_size=data["frame_size"], offset_s=offset_s)
    if not picks:
        raise ValueError("No matching pose tracklets")
    device = torch.device(args.device)
    model = RelativePoseModel(config, scenes["up_priors"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    output = args.output
    (output / "images").mkdir(parents=True, exist_ok=True)
    examples = []
    for i, index in enumerate(picks):
        sample = dataset[index]
        batch = to_device(default_collate([sample]), device)
        result = model(batch)
        rollout = result["rollout"]
        trajectory = result["trajectory"]
        times = [0.0, *sample["future_time"].tolist()]
        frame_wh = torch.tensor(data["frame_size"]).repeat(2)
        boxes = torch.cat((sample["past"][-1:], sample["future"])) * frame_wh
        frames = []
        frame_indices = sorted(set(range(0, len(times), args.frame_step)) | {len(times) - 1})
        for j in frame_indices:
            time_s = sample["anchor_time"] + times[j]
            rgb_time = time_s + offset_s
            native = copy_rgb(rgb, output, sample["folder"], "rgb", rgb_time)
            if native:
                raw_box, delta = nearest_rgb_box(rgb, sample["folder"], rgb_time, boxes[j].numpy(), data["frame_size"])
                native["box_px"] = raw_box if delta <= max(32, 0.75 * float(boxes[j, 2:].max())) else None
            frames.append({"index": j, "time_s": time_s,
                           "padded_rgb": copy_rgb(rgb, output, sample["folder"], "padded_rgb", rgb_time), "rgb": native})
        predicted = result["boxes_px"][0].cpu()
        error = torch.linalg.vector_norm(predicted[1:, :2] - boxes[1:, :2], dim=-1)
        examples.append({
            "sequence": sample["folder"], "track_id": sample["track_id"], "anchor_time_s": sample["anchor_time"],
            "annotation_source": splits.get("annotation_sources", {}).get(sample["folder"], "cleaned tracks"),
            "scene": scenes["names"][sample["scene"]], "times": times, "frames": frames,
            "ade_px": float(error.mean()), "fde_px": float(error[-1]),
            "past_boxes_px": (sample["past"] * frame_wh).tolist(),
            "target_boxes_px": boxes.tolist(), "projected_boxes_px": predicted.tolist(),
            "positions": rollout[0].cpu().tolist(), "up": result["up"][0].cpu().tolist(),
            "up_over_time": trajectory.up[0].cpu().tolist(),
            "velocity_over_time": trajectory.velocity[0].cpu().tolist(),
            "acceleration_over_time": trajectory.acceleration[0].cpu().tolist(),
            "angular_velocity": result["angular_velocity"][0].cpu().tolist(), "specific_thrust": float(result["thrust"][0, 0]),
            "gravity_up": result["gravity_up"][0].cpu().tolist(), "focal": result["focal"][0].cpu().tolist(),
            "principal_point": result["center"][0].cpu().tolist(), "event_preview": save_event_preview(sample["events"], output, i),
        })
    payload = {"split": split, "epoch": checkpoint["epoch"], "frame_size": data["frame_size"],
               "rgb_offset_ms": args.rgb_offset_ms,
               "history_ms": data["history_ms"], "forecast_ms": data["forecast_ms"],
               "event_window_ms": data.get("event_window_ms", data["history_ms"]),
               "body_half_height": model.half_height, "examples": examples}
    (output / "data.js").write_text("window.POSE_DATA = " + json.dumps(payload, allow_nan=False) + ";\n", encoding="utf-8")
    state_export = {
        "axes": ["right", "down", "forward"],
        "length_unit": "assumed body diameter",
        "angular_velocity_unit": "rad/s",
        "reference_diameter_m": config["model"]["reference_diameter_m"],
        "gravity_diameters_s2": model.gravity,
        "linear_drag_per_s": model.linear_drag,
        "epoch": checkpoint["epoch"],
        "states": [
            {"sequence": item["sequence"], "track_id": item["track_id"], "scene": item["scene"],
             "time_s": item["anchor_time_s"], "position": item["positions"][0], "up": item["up"],
             "velocity": item["velocity_over_time"][0], "acceleration": item["acceleration_over_time"][0],
             "angular_velocity": item["angular_velocity"], "specific_thrust": item["specific_thrust"],
             "camera_up": item["gravity_up"], "focal_px": item["focal"], "principal_point_px": item["principal_point"]}
            for item in examples
        ],
    }
    (output / "flight_states.json").write_text(json.dumps(state_export, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    assets = Path(__file__).with_name("viewer_assets")
    for name in ("index.html", "viewer.js"):
        copyfile(assets / name, output / name)
    return output / "index.html"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--root", type=Path, help="Override the checkpoint's FRED data path")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--split", choices=("train", "val"))
    parser.add_argument("--sequence")
    parser.add_argument("--track-id", type=int)
    parser.add_argument("--time", type=float, help="Find the nearest anchor to this elapsed event time in seconds")
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--rgb-tolerance-ms", type=int, default=75)
    parser.add_argument("--rgb-offset-ms", type=int, default=0, help="Shift the RGB lookup time relative to the event clock")
    args = parser.parse_args()
    if args.count < 1 or args.frame_step < 1 or args.rgb_tolerance_ms < 0:
        parser.error("Expected positive count and frame step and nonnegative tolerance")
    if (args.time is not None or args.track_id is not None) and not args.sequence:
        parser.error("--time and --track-id require --sequence")
    print(export(args).resolve())


if __name__ == "__main__":
    main()
