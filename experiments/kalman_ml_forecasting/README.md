Kalman ML Forecasting

This experiment forecasts UAV boxes with a real constant-velocity Kalman
baseline plus learned residual dynamics from rendered FRED images.

Model
- State is `[cx, cy, w, h, vx, vy, vw, vh]` in normalized box coordinates.
- The learned residual model uses a fixed constant-velocity transition:
  `p_{t+1} = p_t + v_t dt`.
- The learned branch predicts acceleration residuals from `event_images` and
  optional RGB:
  `x_{t+1} = F x_t + delta_x_t`.
- In code, `delta_x_t` comes from a multi-branch CNN/ResNet encoder over
  `xt_my`, `yt_mx`, `cstr3`, `rgb`/`padded_rgb`, or any other rendered
  representation listed in `data.representations`.
- The branch follows the object detection experiment pattern: one encoder per
  representation, concatenated pooled features, then an MLP head.
- The Kalman baseline is separate from the learned model. It is initialized
  fresh for each sample and only filters the configured history window, for
  example `0.4 s`, before rolling forward without future measurements.
- The old last-two-point constant-velocity extrapolator is still reported as
  `last2_*` metrics for comparison.

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

Optimize Kalman Parameters

Tune the Kalman filter trust parameters on the training split only:

```bash
python experiments/kalman_ml_forecasting/optimize_kalman.py --config experiments/kalman_ml_forecasting/configs/base.yaml --trials 200 --objective fde_center_px
```

Maximize a metric such as mIoU:

```bash
python experiments/kalman_ml_forecasting/optimize_kalman.py --config experiments/kalman_ml_forecasting/configs/base.yaml --trials 200 --objective miou --maximize-objective
```

Optimize a weighted objective by minimizing a score. Negative weights maximize
that metric:

```bash
python experiments/kalman_ml_forecasting/optimize_kalman.py --config experiments/kalman_ml_forecasting/configs/base.yaml --trials 200 --objective-weights "fde_center_px=1,ade_center_px=0.25,miou=-100"
```

The optimizer:
- uses only `split_files.train`
- creates an internal tune-train/tune-val split by `(folder, track_id)` so
  overlapping tracklets from the same track do not cross the split
- ignores rendered representation availability because Kalman tuning only uses
  track boxes and timestamps
- does not apply `data.max_samples_train` unless `--max-samples` is passed
- evaluates every candidate by starting a fresh filter for each sample history
  window
- minimizes either `--objective`, the negative of `--objective` with
  `--maximize-objective`, or the weighted sum from `--objective-weights`
- uses the configured `kalman:` block as the initial incumbent, so random
  trials must beat the current config to become the best result
- prints the best parameters in YAML-ready `kalman:` format
- reports last-two CV metrics beside the tuned Kalman metrics

By default it uses all available train tracklets. You can cap runtime with:

```bash
python experiments/kalman_ml_forecasting/optimize_kalman.py --config experiments/kalman_ml_forecasting/configs/base.yaml --trials 50 --max-tune-train-samples 2000 --max-tune-val-samples 2000
```

Visualize Tracks

Render one GIF per track for a specific FRED folder:

```bash
python experiments/kalman_ml_forecasting/visualize_tracks.py --config experiments/kalman_ml_forecasting/configs/base.yaml --checkpoint outputs/kalman_ml_forecasting_ckpt/best.pt --folder 8 --backdrop-rep cstr3
```

Useful options:
- `--backdrop-rep cstr3`, `xt_my`, `yt_mx`, `rgb`, or `padded_rgb`
- `--track-id 12 --track-id 25` to render only selected tracks
- `--max-tracks 10` to cap a batch render
- `--max-frames-per-track 200` to cap GIF length
- `--include-cv` to draw the configured Kalman CV baseline in cyan alongside the learned prediction
- `--include-last2` to draw the old last-two CV baseline in cyan
- `--baseline-only` to render configured Kalman predictions without loading a checkpoint

The overlay colors are:
- blue: history boxes
- yellow: learned predicted boxes
- green: future ground truth boxes
- cyan: optional Kalman or last-two CV baseline

Metrics
- The trainer reports the learned model metrics, configured Kalman baseline
  metrics, and last-two CV metrics in the same validation pass.
- `kalman_ade_center_px` and related `kalman_*` metrics are the real Kalman
  baseline.
- `last2_ade_center_px` and related `last2_*` metrics are the old last-two
  extrapolator.
- Non-prefixed metrics are the image-conditioned residual model.

Extension points
- Add or remove branches with `data.representations`.
- Use grid-split render aliases such as `xt_my_10x10` as in object detection.
- Set `model.predict_size_residuals: false` to learn only center acceleration
  residuals while leaving size dynamics at constant velocity.
- Tune `model.residual_scale` if residual accelerations are too aggressive early
  in training.
