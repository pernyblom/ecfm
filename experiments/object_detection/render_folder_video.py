from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.object_detection.data.dataset import _parse_frame_time, _read_yolo_boxes
from experiments.object_detection.metrics import detection_scores
from experiments.object_detection.models.factory import build_model
from experiments.object_detection.render_sequence_video import (
    _draw_pred_overlay,
    _gt_heatmap_for_rep,
    _infer_fps,
    _is_dataset_rgb_rep,
    _load_background_image,
    _load_input_tensor,
    _overlay_heatmap,
    _parse_mode_list,
    _parse_rep_list,
    _pred_heatmap_for_rep,
    _write_video_cv2,
)
from experiments.object_detection.utils.config import load_config, resolve_representation_image_sizes


def _find_image_stems(*, images_dir: Path, required_reps: List[str]) -> List[str]:
    required_rendered_reps = [rep for rep in required_reps if not _is_dataset_rgb_rep(rep)]
    if not required_rendered_reps:
        raise ValueError("At least one rendered representation is required to discover frame stems.")
    first_rep = required_rendered_reps[0]
    suffix = f"_{first_rep}.png"
    stems: List[str] = []
    for path in images_dir.glob(f"*{suffix}"):
        stem = path.name[: -len(suffix)]
        if all((images_dir / f"{stem}_{rep}.png").exists() for rep in required_rendered_reps):
            stems.append(stem)
    stems.sort(key=lambda item: (_parse_frame_time(item) is None, _parse_frame_time(item) or 0, item))
    return stems


def render_detection_folder(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    reps = _parse_rep_list(args.reps)
    heatmap_modes = _parse_mode_list(args.heatmaps)
    invalid_heatmap_modes = sorted(set(heatmap_modes) - {"pred", "gt"})
    if invalid_heatmap_modes:
        raise ValueError(f"Unknown heatmap modes: {invalid_heatmap_modes}. Use pred, gt, or pred;gt.")
    if not reps:
        raise ValueError("At least one representation must be requested.")
    if "gt" in heatmap_modes and args.labels_dir is None:
        raise ValueError("--heatmaps includes gt, so --labels-dir is required.")

    images_dir = args.input_dir
    if not images_dir.exists():
        raise FileNotFoundError(f"Missing input representation folder: {images_dir}")

    required_reps = sorted(dict.fromkeys(list(data_cfg["representations"]) + reps))
    stems = _find_image_stems(images_dir=images_dir, required_reps=required_reps)
    if args.max_frames is not None:
        stems = stems[: max(0, int(args.max_frames))]
    if not stems:
        raise RuntimeError(f"No frames found in {images_dir} with required reps {required_reps}.")

    device = torch.device(cfg["train"].get("device", "cpu") if torch.cuda.is_available() else "cpu")
    model = build_model(cfg, device)
    state = torch.load(args.checkpoint, map_location=device)
    model_state = state["model"] if isinstance(state, dict) and "model" in state else state
    model.load_state_dict(model_state)
    model.eval()

    image_sizes = resolve_representation_image_sizes(data_cfg)
    run_name = args.name or images_dir.name
    output_dir = args.output_dir / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    fps = float(args.fps) if args.fps is not None else _infer_fps(stems)
    print(f"Rendering detections for {images_dir} with {len(stems)} frames at {fps:.3f} fps")

    labels_dir = args.labels_dir
    dataset_folder_dir = args.dataset_folder_dir
    if dataset_folder_dir is None:
        dataset_folder_dir = labels_dir.parent if labels_dir is not None else images_dir

    rep_frame_dirs: Dict[str, Path] = {}
    heatmap_frame_dirs: Dict[tuple[str, str], Path] = {}
    for rep in reps:
        rep_frames_dir = output_dir / f"frames_{rep}"
        rep_frames_dir.mkdir(parents=True, exist_ok=True)
        rep_frame_dirs[rep] = rep_frames_dir
        for mode_name in heatmap_modes:
            heatmap_dir = output_dir / f"frames_{rep}_{mode_name}_heatmap"
            heatmap_dir.mkdir(parents=True, exist_ok=True)
            heatmap_frame_dirs[(rep, mode_name)] = heatmap_dir

    for idx, stem in enumerate(stems):
        model_inputs: Dict[str, torch.Tensor] = {}
        for model_rep in data_cfg["representations"]:
            model_inputs[model_rep] = _load_input_tensor(
                images_dir / f"{stem}_{model_rep}.png",
                image_sizes[model_rep],
            ).unsqueeze(0).to(device)

        with torch.no_grad():
            preds = model(model_inputs)
            pred_boxes = preds["boxes"][0].detach().cpu()
            pred_scores = detection_scores(preds)[0].detach().cpu()

        label_path = labels_dir / f"{stem}.txt" if labels_dir is not None else None
        gt_boxes = _read_yolo_boxes(label_path) if label_path is not None and label_path.exists() else []
        frame_time_raw = _parse_frame_time(stem)
        frame_time_s = (
            float(frame_time_raw) * float(data_cfg.get("label_time_unit", 1e-6))
            if frame_time_raw is not None
            else 0.0
        )

        for rep in reps:
            bg_img = _load_background_image(
                rep=rep,
                stem=stem,
                label_time_s=frame_time_s,
                images_dir=images_dir,
                dataset_folder_dir=dataset_folder_dir,
                label_time_unit=float(data_cfg.get("label_time_unit", 1e-6)),
                rgb_indices={},
                rgb_source=str(args.rgb_source),
            )
            vis = _draw_pred_overlay(
                bg_img,
                rep=rep,
                pred_boxes=pred_boxes,
                pred_scores=pred_scores,
                score_threshold=float(args.score_threshold),
                gt_boxes=gt_boxes,
                draw_gt=bool(args.draw_ground_truth),
            )
            vis.save(rep_frame_dirs[rep] / f"{idx:06d}.png")

            if "gt" in heatmap_modes:
                gt_heat = _gt_heatmap_for_rep(
                    rep=rep,
                    gt_boxes=gt_boxes,
                    size=bg_img.size,
                    gaussian_radius=int(args.heatmap_radius),
                )
                _overlay_heatmap(
                    bg_img,
                    gt_heat,
                    color=(0, 255, 0),
                    alpha=float(args.heatmap_alpha),
                ).save(heatmap_frame_dirs[(rep, "gt")] / f"{idx:06d}.png")
            if "pred" in heatmap_modes:
                pred_heat = _pred_heatmap_for_rep(preds, rep)
                if pred_heat is not None:
                    _overlay_heatmap(
                        bg_img,
                        pred_heat,
                        color=(255, 196, 0),
                        alpha=float(args.heatmap_alpha),
                    ).save(heatmap_frame_dirs[(rep, "pred")] / f"{idx:06d}.png")
                else:
                    bg_img.save(heatmap_frame_dirs[(rep, "pred")] / f"{idx:06d}.png")

        if (idx + 1) % 100 == 0:
            print(f"rendered {idx + 1}/{len(stems)} frames")

    for rep in reps:
        video_path = output_dir / f"{run_name}_{rep}.mp4"
        _write_video_cv2(rep_frame_dirs[rep], video_path, fps)
        print(f"Wrote {video_path}")
        if not args.keep_frames:
            for frame_path in rep_frame_dirs[rep].glob("*.png"):
                frame_path.unlink()
            rep_frame_dirs[rep].rmdir()
        for mode_name in heatmap_modes:
            heatmap_dir = heatmap_frame_dirs[(rep, mode_name)]
            video_path = output_dir / f"{run_name}_{rep}_{mode_name}_heatmap.mp4"
            _write_video_cv2(heatmap_dir, video_path, fps)
            print(f"Wrote {video_path}")
            if not args.keep_frames:
                for frame_path in heatmap_dir.glob("*.png"):
                    frame_path.unlink()
                heatmap_dir.rmdir()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run an object detection checkpoint on any folder of rendered representation images."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Folder containing images named <stem>_<rep>.png.",
    )
    parser.add_argument(
        "--reps",
        type=str,
        default="cstr3",
        help="Representations to render as video backgrounds, separated by ';'.",
    )
    parser.add_argument(
        "--labels-dir",
        type=Path,
        default=None,
        help="Optional YOLO label folder for drawing ground truth and GT heatmaps.",
    )
    parser.add_argument(
        "--dataset-folder-dir",
        type=Path,
        default=None,
        help="Optional original sequence folder for RGB/PADDED_RGB fallback.",
    )
    parser.add_argument(
        "--rgb-source",
        type=str,
        default="auto",
        choices=["auto", "rgb", "padded_rgb"],
    )
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--draw-ground-truth", action="store_true", default=False)
    parser.add_argument("--heatmaps", type=str, default="")
    parser.add_argument("--heatmap-alpha", type=float, default=0.55)
    parser.add_argument("--heatmap-radius", type=int, default=8)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/object_detection_folder_videos"))
    parser.add_argument("--name", type=str, default=None, help="Optional name for the output subfolder and videos.")
    parser.add_argument("--keep-frames", action="store_true", default=True)
    parser.add_argument("--no-keep-frames", action="store_false", dest="keep_frames")
    args = parser.parse_args()
    render_detection_folder(args)


if __name__ == "__main__":
    main()
