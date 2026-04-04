Pose + Dynamics Self-Supervised Experiment (FRED)

This experiment is isolated from the main training code and from
`experiments/forecasting`.

Goal
- Learn latent relative pose and dynamics of a UAV from event-history images
  (`xt_my`, `yt_mx`) in a self-supervised way.
- Use approximate camera intrinsics/pose as inputs.
- Train by projecting predicted 3D dynamics into image-plane centers and
  minimizing center reprojection error against track-derived centers.
- Constrain the latent 3D solution with optional target-size and motion-bound
  priors so depth and dynamics stay more physically plausible.

How samples are built
- Uses `cleaned_tracks.txt` (or `tracks_file` in config) + frame timestamps from
  `Event_YOLO/*.txt`.
- Aligns track time to frame time (`time_align`).
- Interpolates tracks to frame times.
- For each sample:
  - Input image at `t0`: one image per representation (`xt_my`, `yt_mx`).
  - Supervision: center trajectory and normalized box sizes for next `future_steps`.

Quick start
1) Render required reps:
   python scripts/render_fred_splits.py --split-file datasets/FRED/dataset_splits/canonical/train_split.txt --output-root outputs/fred_reps --representation "xt_my;yt_mx" --window 33333 --window-mode trailing --num-workers 4 --include-empty

   python scripts/render_fred_splits.py --split-file datasets/FRED/dataset_splits/canonical/test_split.txt --output-root outputs/fred_reps --representation "xt_my;yt_mx" --window 33333 --window-mode trailing --num-workers 4 --include-empty

2) Train:
   python experiments/pose_dynamics/train.py --config experiments/pose_dynamics/configs/base.yaml

3) Visualize a checkpoint:
   python experiments/pose_dynamics/visualization.py --config experiments/pose_dynamics/configs/base.yaml --checkpoint outputs/pose_dynamics_ckpt/best.pt --split val --output-dir outputs/pose_dynamics_vis_best

Notes
- Use `--window-mode trailing` for training data so the event image contains only
  history up to the label time. `center` and `leading` are available for
  analysis, but they leak future information for forecasting-style training.
- `data.history_steps` controls how many previous labeled positions are carried
  into the dataset and encoded by the model.
- `train.vis_every` writes validation trajectory overlays during training.
- `data.camera.intrinsics` and `data.camera.pose` are approximate priors.
- The network learns corrections (`intrinsics_delta`, `pose_delta`) together
  with latent dynamics.
- Intrinsics in `base.yaml` are normalized for normalized center targets.
- `train.size_prior` defines a physical size range for the UAV. The size loss
  does not force one exact size; it penalizes predicted depth/scale combinations
  that make the tracked box too small or too large for that range.
- `train.max_speed` and `train.max_acc` are soft bounds. They are applied as
  hinge penalties, so staying within the bounds incurs no extra loss.
- Training writes structured per-epoch metrics to `train.metrics_jsonl`
  (default `outputs/pose_dynamics_ckpt/metrics.jsonl`) in addition to stdout.
- The most useful metrics to watch are:
  - `center`: mean reprojection L1 error
  - `size`: violation of the configured UAV size range
  - `speed_b`, `acc_b`: motion-bound violations
  - `end_l2`: endpoint forecast error
  - `oob_step`, `oob_traj`: how often predictions leave the normalized image
