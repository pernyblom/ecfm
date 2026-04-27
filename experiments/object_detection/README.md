Object Detection

This experiment trains a simple single-frame multi-object detector on FRED
using rendered event representations and YOLO labels. The default setup uses
`xt_my`, `yt_mx`, and `cstr3` as inputs, predicts optional `xt` and `yt`
heatmaps, and predicts a fixed set of object queries for the anchor frame.

Input sizing
- The detector now supports per-representation input sizes.
- The default config uses:
  - `data.frame_size: [1280, 720]`
  - `data.temporal_bins: 224`
  - `data.retain_spatial_dimensions: true`
- With that convention:
  - `cstr3`, `xy`, and `rgb` stay at `1280x720`
  - `xt_*` becomes `1280x224`
  - `yt_*` becomes `224x720`
- If you need manual control instead, set `data.image_sizes` per representation.

DETR-lite head
- The fused representation is broadcast to a fixed set of learned object
  queries.
- Each query gets its own learned embedding and is passed through a small MLP
  head that predicts one box and one objectness logit.
- Training uses matching between the `K` predicted queries and the GT boxes, so
  the model can handle `0..K` objects per frame without anchors or NMS in the
  training loss.

Current loss
- box regression uses `L1 + CIoU` on matched queries
- objectness uses binary cross-entropy over all queries
- heatmap heads, when enabled, use binary cross-entropy
- matching between predicted queries and GT boxes uses an internal exact matcher
  suited for the small number of objects per FRED frame
- if heatmaps are disabled, heatmap-only losses and metrics are omitted from
  the summaries instead of being reported as dummy zeros

Reported metrics
- `matched_center_l1_px`
- `matched_box_iou`
- `mAP_50`
- `mAP_50:95`
- `objectness_acc`
- per-plane heatmap IoU for `xt_my` and `yt_mx`

The mAP values follow the FRED paper protocol:
- `mAP_50`: AP at IoU `0.50`
- `mAP_50:95`: mean AP over IoU thresholds `0.50, 0.55, ..., 0.95`

The detector predicts an explicit objectness score per query. Those scores are
used for AP ranking, which is much closer to the detector-style evaluation used
in the paper than the earlier heatmap-based confidence proxy.

What it does
- Loads rendered anchor-frame images and YOLO labels from FRED.
- Uses one encoder per representation and fuses the representation features.
- Predicts:
  - an `xt_my` heatmap
  - a `yt_mx` heatmap
  - `K` query boxes in normalized YOLO format `(cx, cy, w, h)`
  - `K` objectness scores
- Saves visualizations for GT and predicted heatmaps directly in `xt_my` and
  `yt_mx`, plus GT and predicted XY boxes over a selectable backdrop such as
  `cstr3`.

Heatmaps are optional
- Set `data.heatmap_representations: []` to disable heatmap heads and heatmap
  supervision entirely.
- In that mode, the model still trains the XY box head and objectness head, and
  mAP uses the explicit objectness score as before.
- If heatmaps are disabled, heatmap IoU and heatmap visualizations are simply
  omitted.

Negative frames
- Set `data.require_boxes: false` to keep empty frames.
- This is useful for learning the objectness head against true negatives.
- Frames with multiple labeled objects are kept by default.
- Set `data.exclude_multiple_objects: true` if you want to restrict training or
  evaluation to empty frames plus single-object frames for a simpler head.

Heatmap target geometry
- `xt_my` is an XT image where X is horizontal and time is vertical.
- `yt_mx` is a YT image where Y is vertical and time is horizontal.
- Because the YOLO label only gives the object box at the anchor frame, not a
  time-resolved XT/YT tube through the whole render window, this experiment uses
  the least-assumption target:
  - in `xt_my`, the box X-range is marked across the full time axis
  - in `yt_mx`, the box Y-range is marked across the full time axis
- That means the heatmaps supervise the correct spatial support in each plane
  while treating the full render window as relevant for the current label.

Why `33.333 ms` is the right render window
- FRED label stems advance by about `33333` microseconds.
- For this experiment the event image should align exactly with the current box,
  so the event window should match one label period:
  - `--window 33333`
  - `--window-mode trailing`
- `trailing` avoids future leakage.
- The dataset validates the rendered duration against `render_manifest.json`.

Render

```bash
python scripts/render_fred_splits.py --split-file datasets/FRED/dataset_splits/canonical/train_split.txt --output-root outputs/fred_reps --representation "xt_my;yt_mx;cstr3" --window 33333 --window-mode trailing --temporal-bins 224 --retain-spatial-dimensions --event-source raw --num-workers 6 --include-empty
python scripts/render_fred_splits.py --split-file datasets/FRED/dataset_splits/canonical/test_split.txt --output-root outputs/fred_reps --representation "xt_my;yt_mx;cstr3" --window 33333 --window-mode trailing --temporal-bins 224 --retain-spatial-dimensions --event-source raw --num-workers 4 --include-empty
```

The dataset loader checks `render_manifest.json` and will fail fast if the
rendered per-representation sizes do not match the configured convention.

`--event-source` defaults to `raw`. Use `auto` only if you want the renderer to
fall back to `output_events.npz` when streamed raw decode fails.

Train

```bash
python experiments/object_detection/train.py --config experiments/object_detection/configs/base.yaml
```

Training throughput notes
- The dataset sample index is cached under `data.cache_dir`, so the first run
  after changing dataset-related config is the slow one.
- The train and validation loaders now default to persistent workers on
  multi-worker runs, which avoids a full worker respawn at each epoch and when
  switching from training to validation.
- Full epoch mAP is relatively expensive for query-based detection. By default
  it is computed for validation but not for training. Control this with:
  - `train.compute_train_epoch_map`
  - `train.compute_val_epoch_map`
- Useful loader knobs in `train`:
  - `num_workers`
  - `persistent_workers`
  - `pin_memory`
  - `prefetch_factor`

Evaluate

```bash
python experiments/object_detection/eval.py --config experiments/object_detection/configs/base.yaml --checkpoint outputs/object_detection_ckpt/best.pt
```

Render sequence videos

```bash
python experiments/object_detection/render_sequence_video.py --config experiments/object_detection/configs/base.yaml --checkpoint outputs/object_detection_ckpt/best.pt --folder 8 --reps "cstr3;xt_my;yt_mx" --score-threshold 0.5 --draw-ground-truth
```

Omit `--checkpoint` to render backgrounds without model predictions. Combine
that with `--draw-ground-truth` for ground-truth-only videos:

```bash
python experiments/object_detection/render_sequence_video.py --config experiments/object_detection/configs/base.yaml --folder 8 --reps "cstr3;xt_my;yt_mx" --draw-ground-truth
```

This writes per-frame overlays and one MP4 per requested representation using
OpenCV's `cv2.VideoWriter`. On `xt_my` and `yt_mx`, the box is drawn as a stripe
that spans the full time axis.

For `rgb`, the script first uses a rendered `*_rgb.png` if it exists under the
representation output folder. If not, it falls back to the original dataset
`RGB` or `PADDED_RGB` frames for visualization only. This does not change the
representations used by the model itself.

Compose a 2x2 grid video

```bash
python experiments/object_detection/compose_grid_video.py --sequence-dir outputs/object_detection_sequence_videos/8 --output outputs/object_detection_sequence_videos/8/8_grid.mp4
```

This places:
- `cstr3` top-left
- `yt_mx` top-right
- `xt_my` bottom-left
- `rgb` bottom-right

The output size is exactly `2 * tile_width` by `2 * tile_height`, where the
tile size defaults to the `cstr3` video size. The RGB panel is scaled to fit
inside its quadrant while preserving aspect ratio.

Extension points
- Add more inputs via `data.representations`.
- Add more plane-supervision targets via `data.heatmap_representations`.
- Swap the encoder via `model.backbone.type`, for example `resnet18`.
- Increase or decrease `model.num_queries` based on scene complexity.

Current scope
- The detector predicts a fixed number of query boxes per frame and learns which
  ones correspond to real objects through matching and objectness supervision.
- If a frame contains more GT boxes than `model.num_queries`, the largest
  `num_queries` boxes are used for training.
