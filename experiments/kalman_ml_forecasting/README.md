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
- The linear extrapolation baseline follows the paper-style setup: estimate
  velocity from the last four observed poses, then roll forward with constant
  velocity. We apply the same last-four linear fit to all box channels
  `(cx, cy, w, h)`.

Data
- Samples are track-aligned from `cleaned_tracks.txt` and anchor label times in
  `Event_YOLO`.
- Each sample uses one anchor image at time `t`, past boxes up to `t`, and
  future boxes after `t`.
- Event windows should be rendered with `--window-mode trailing` to avoid future
  leakage.
- `rgb` and `padded_rgb` can be read directly from the FRED sequence folders if
  a rendered aligned PNG does not exist.
- `event_frames` can read pre-rendered dataset frames directly from
  `datasets/FRED/<folder>/Event/Frames`. This representation is only usable for
  folders where that directory exists; with `filter_missing_representations:
  true`, folders/samples without it are skipped.

Render

```bash
python scripts/render_fred_splits.py --split-file datasets/FRED/dataset_splits/canonical/train_split.txt --output-root outputs/fred_reps --representation "xt_my;yt_mx;cstr3" --window 400000 --window-mode trailing --temporal-bins 224 --retain-spatial-dimensions --event-source raw --num-workers 6 --include-empty
python scripts/render_fred_splits.py --split-file datasets/FRED/dataset_splits/canonical/test_split.txt --output-root outputs/fred_reps --representation "xt_my;yt_mx;cstr3" --window 400000 --window-mode trailing --temporal-bins 224 --retain-spatial-dimensions --event-source raw --num-workers 4 --include-empty
```

Train

```bash
python experiments/kalman_ml_forecasting/train.py --config experiments/kalman_ml_forecasting/configs/base.yaml
```

Evaluation protocol:
- `train.eval_splits_each_epoch` controls non-training evaluation after each
  epoch. The default evaluates `train_eval`, a capped subset of
  `data.split_files.train`, and `val`.
- Best checkpoints are selected with `train.best_metric_split`,
  `train.best_metric`, and `train.best_metric_mode`.
- To keep the test split untouched during training, set `data.split_files.val`
  to a validation split and `data.split_files.test` to the held-out test split,
  then enable `train.run_test_on_best: true`. The script reloads `best.pt` after
  training and writes `train.test_metrics_json`.

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
- reports last-four linear extrapolation metrics beside the tuned Kalman metrics

By default it uses all available train tracklets. You can cap runtime with:

```bash
python experiments/kalman_ml_forecasting/optimize_kalman.py --config experiments/kalman_ml_forecasting/configs/base.yaml --trials 50 --max-tune-train-samples 2000 --max-tune-val-samples 2000
```

Backprop Optimization

There is also an experimental optimizer that backpropagates through the Kalman
filter applications. It initializes the log-standard-deviation parameters from
the configured `kalman:` block and optimizes the same weighted objective:

```bash
python experiments/kalman_ml_forecasting/optimize_kalman_backprop.py --config experiments/kalman_ml_forecasting/configs/base.yaml --epochs 50 --lr 1.0e-2 --objective fde_center_px
```

Weighted objectives work the same way:

```bash
python experiments/kalman_ml_forecasting/optimize_kalman_backprop.py --config experiments/kalman_ml_forecasting/configs/base.yaml --epochs 50 --lr 1.0e-2 --objective-weights "fde_center_px=1,ade_center_px=0.25,miou=-100"
```

Notes:
- parameters are optimized in log space and clamped by `--min-std`/`--max-std`
- the configured `kalman:` block is the initial incumbent
- mIoU is piecewise differentiable and can have weak gradients when boxes do
  not overlap, so distance-based terms are often useful in the weighted loss

Visualize Tracks

Render one GIF per track for a specific FRED folder:

```bash
python experiments/kalman_ml_forecasting/visualize_tracks.py --config experiments/kalman_ml_forecasting/configs/base.yaml --checkpoint outputs/kalman_ml_forecasting_ckpt/best.pt --folder 8 --backdrop-rep cstr3
```

Useful options:
- `--backdrop-rep cstr3`, `xt_my`, `yt_mx`, `rgb`, `padded_rgb`, or `event_frames`
- `--track-id 12 --track-id 25` to render only selected tracks
- `--max-tracks 10` to cap a batch render
- `--max-frames-per-track 200` to cap GIF length
- `--include-cv` to draw the configured Kalman CV baseline in cyan alongside the learned prediction
- `--include-last4` to draw the last-four linear extrapolation baseline in cyan
- `--baseline-only` to render configured Kalman predictions without loading a checkpoint

The overlay colors are:
- blue: history boxes
- yellow: learned predicted boxes
- green: future ground truth boxes
- cyan: optional Kalman or last-four extrapolation baseline

Metrics
- The trainer reports the learned model metrics, configured Kalman baseline
  metrics, and last-four extrapolation metrics in the same validation pass.
- `kalman_ade_center_px` and related `kalman_*` metrics are the real Kalman
  baseline.
- `last4_ade_center_px` and related `last4_*` metrics are the last-four
  linear extrapolator.
- Non-prefixed metrics are the image-conditioned residual model.

Extension points
- Add or remove branches with `data.representations`.
- Use `data.representations: []` to train a residual model without image/CNN
  inputs. This is only valid when `model.use_filter_state_features: true` or
  `model.filter_covariance_features` is `diag`/`full`.
- Use grid-split render aliases such as `xt_my_10x10` as in object detection.
- Set `model.predict_size_residuals: false` to learn only center acceleration
  residuals while leaving size dynamics at constant velocity.
- Set `model.initial_state_source: kalman_filter` to start the learned residual
  rollout from the configured Kalman filter's final history state instead of
  the default last-four linear-fit state.
- Set `model.use_filter_state_features: true` to append the configured Kalman
  filter's final history state to the CNN encoder features before image fusion.
  The residual rollout still starts from the last-four linear-fit state; the
  filter state is additional conditioning information unless
  `model.initial_state_source: kalman_filter` is also set.
- Set `model.filter_covariance_features: diag` or `full` to append the Kalman
  history covariance diagonal or flattened covariance matrix to the fusion
  features. The default is `none`.
- Tune `model.residual_scale` if residual accelerations are too aggressive early
  in training.
