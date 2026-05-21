Kalman ML Forecasting

This experiment forecasts UAV boxes with a constant-velocity Kalman-style
baseline plus learned residual dynamics from rendered FRED images.

Model
- State is `[cx, cy, w, h, vx, vy, vw, vh]` in normalized box coordinates.
- The fixed transition is constant velocity:
  `p_{t+1} = p_t + v_t dt`.
- The learned branch predicts acceleration residuals from `event_images` and
  optional RGB:
  `x_{t+1} = F x_t + delta_x_t`.
- In code, `delta_x_t` comes from a multi-branch CNN/ResNet encoder over
  `xt_my`, `yt_mx`, `cstr3`, `rgb`/`padded_rgb`, or any other rendered
  representation listed in `data.representations`.
- The branch follows the object detection experiment pattern: one encoder per
  representation, concatenated pooled features, then an MLP head.

Data
- Samples are track-aligned from `cleaned_tracks.txt` and anchor label times in
  `Event_YOLO`.
- Each sample uses one anchor image at time `t`, past boxes up to `t`, and
  future boxes after `t`.
- Event windows should be rendered with `--window-mode trailing` to avoid future
  leakage.
- `rgb` and `padded_rgb` can be read directly from the FRED sequence folders if
  a rendered aligned PNG does not exist.

Render

```bash
python scripts/render_fred_splits.py --split-file datasets/FRED/dataset_splits/canonical/train_split.txt --output-root outputs/fred_reps --representation "xt_my;yt_mx;cstr3" --window 400000 --window-mode trailing --temporal-bins 224 --retain-spatial-dimensions --event-source raw --num-workers 6 --include-empty
python scripts/render_fred_splits.py --split-file datasets/FRED/dataset_splits/canonical/test_split.txt --output-root outputs/fred_reps --representation "xt_my;yt_mx;cstr3" --window 400000 --window-mode trailing --temporal-bins 224 --retain-spatial-dimensions --event-source raw --num-workers 4 --include-empty
```

Train

```bash
python experiments/kalman_ml_forecasting/train.py --config experiments/kalman_ml_forecasting/configs/base.yaml
```

Metrics
- The trainer reports the learned model metrics and the constant-velocity
  baseline metrics in the same validation pass.
- `cv_ade_center_px` and related `cv_*` metrics are the baseline.
- Non-prefixed metrics are the image-conditioned residual model.

Extension points
- Add or remove branches with `data.representations`.
- Use grid-split render aliases such as `xt_my_10x10` as in object detection.
- Set `model.predict_size_residuals: false` to learn only center acceleration
  residuals while leaving size dynamics at constant velocity.
- Tune `model.residual_scale` if residual accelerations are too aggressive early
  in training.

