Object Detection

This experiment trains a simple single-frame detector on FRED using rendered
event representations and YOLO labels. The first version uses `xt_my`, `yt_mx`,
and `cstr3` as inputs, predicts one `xt` heatmap, one `yt` heatmap, and one XY
box for the anchor frame.

Reported metrics
- `center_l1_px`
- `box_iou`
- `mAP_50`
- `mAP_50:95`
- `objectness_acc`
- per-plane heatmap IoU for `xt_my` and `yt_mx`

The mAP values follow the FRED paper protocol:
- `mAP_50`: AP at IoU `0.50`
- `mAP_50:95`: mean AP over IoU thresholds `0.50, 0.55, ..., 0.95`

In the current single-box setup, the detector predicts an explicit objectness
score for the box. That score is used for AP ranking, which is much closer to
the detector-style evaluation used in the paper than the earlier heatmap-based
confidence proxy.

What it does
- Loads rendered anchor-frame images and YOLO labels from FRED.
- Uses one encoder per representation and fuses the representation features.
- Predicts:
  - an `xt_my` heatmap
  - a `yt_mx` heatmap
  - an XY box in normalized YOLO format `(cx, cy, w, h)`
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
python scripts/render_fred_splits.py --split-file datasets/FRED/dataset_splits/canonical/train_split.txt --output-root outputs/fred_reps --representation "xt_my;yt_mx;cstr3" --window 33333 --window-mode trailing --num-workers 6 --include-empty
python scripts/render_fred_splits.py --split-file datasets/FRED/dataset_splits/canonical/test_split.txt --output-root outputs/fred_reps --representation "xt_my;yt_mx;cstr3" --window 33333 --window-mode trailing --num-workers 4 --include-empty
```

Train

```bash
python experiments/object_detection/train.py --config experiments/object_detection/configs/base.yaml
```

Evaluate

```bash
python experiments/object_detection/eval.py --config experiments/object_detection/configs/base.yaml --checkpoint outputs/object_detection_ckpt/best.pt
```

Extension points
- Add more inputs via `data.representations`.
- Add more plane-supervision targets via `data.heatmap_representations`.
- Swap the encoder via `model.backbone.type`, for example `resnet18`.
- Replace the single-box head with a multi-object head later if needed.

Current scope
- The current box head predicts one box per frame.
- The dataset selects one supervision box with `data.select_box` and defaults to
  `largest`.
- This is a pragmatic first step for the current FRED setup while keeping the
  implementation easy to extend.
