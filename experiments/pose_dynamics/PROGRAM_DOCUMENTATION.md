# Pose Dynamics Experiment Documentation

This document explains how the program in `experiments/pose_dynamics` works end to end, what each step does, and which bugs or improvement opportunities are visible from the current implementation.

## Purpose

The experiment trains a self-supervised model that:

1. takes one event-image snapshot per representation at an anchor time,
2. predicts latent camera corrections and latent 3D motion,
3. projects that motion into normalized 2D image centers,
4. compares the projected future centers against centers derived from object tracks.

The implementation is isolated from the main training code and is driven by `experiments/pose_dynamics/train.py`.

## Main Files

- `configs/base.yaml`: experiment configuration
- `train.py`: dataset construction, training loop, validation loop, checkpointing
- `data/track_projection_dataset.py`: sample generation from tracks and frame timestamps
- `models/cnn.py`: small per-representation encoder
- `models/pose_dynamics.py`: latent pose+dynamics projector model
- `models/factory.py`: model builder
- `losses.py`: training objective
- `visualization.py`: trajectory overlay export for manual inspection

## Configuration

`configs/base.yaml` defines the full experiment.

### Data settings

- `images_root`: root containing rendered event representations such as `xt_my` and `yt_mx`
- `labels_root`: root containing FRED labels and track files
- `labels_subdir`: per-sequence label folder, usually `Event_YOLO`
- `tracks_file`: track source, usually `cleaned_tracks.txt`
- `label_time_unit`: converts frame timestamp integers into seconds
- `track_time_unit`: converts track timestamps into seconds
- `time_align`: how tracks are aligned to frame times
- `frame_size`: original width and height used to normalize centers
- `representations`: image modalities loaded for each anchor frame
- `image_size`: resize target for model input
- `history_steps`: number of past labeled positions carried in each sample
- `future_steps`: number of future positions predicted and supervised
- `stride`: spacing between sampled positions in the temporal window
- `split_files`: train/val sequence lists
- `max_tracks`, `max_samples_train`, `max_samples_val`: optional subsampling controls
- `cache_dir`: location for cached sample lists
- `camera.intrinsics`: normalized `[fx, fy, cx, cy]`
- `camera.pose`: approximate `[tx, ty, tz, roll, pitch, yaw]`

### Model settings

- `cnn_channels`: channels for the small CNN backbone
- `feature_dim`: output feature size per representation encoder
- `hidden_dim`: fused latent size
- `min_depth`: lower bound for predicted depth and projected `z`

### Training settings

- `batch_size`, `epochs`, `lr`, `weight_decay`
- `device`, `num_workers`, `log_every`
- regularization weights for center, pose, intrinsics, and acceleration terms
- checkpoint and visualization output settings

## End-to-End Pipeline

### 1. Train script startup

`train.py`:

1. loads YAML config,
2. reads train and validation split files,
3. builds one `TrackProjectionDataset` for train and one for validation,
4. wraps them in `DataLoader`s,
5. builds the model and optimizer,
6. optionally resumes from a checkpoint,
7. runs the epoch loop,
8. saves checkpoints and periodic visualizations.

## Dataset Construction

The dataset is where most of the experiment-specific logic lives.

### 2. Frame discovery

`TrackProjectionDataset._discover_frames()` scans each folder's label files and extracts frame timestamps from filenames using `_parse_frame_time()`.

- It expects filenames containing `_frame_<digits>`.
- The parsed integer timestamp is multiplied by `label_time_unit`.
- Frames are sorted by time.

Output of this step:

- a mapping from sequence folder to ordered `(stem, time_s)` entries.

### 3. Track loading

`_read_tracks()` parses `cleaned_tracks.txt` into:

- `track_id -> [(t, x, y, w, h), ...]`

Each track is sorted by time. The code assumes:

- `x, y, w, h` are bounding-box values in original frame pixels,
- timestamps become seconds after multiplying by `track_time_unit`.

### 4. Optional track subsampling

If `max_tracks` is set, `_select_track_subset()` randomly selects a subset of `(folder, track_id)` pairs using the configured seed. This is a coarse way to reduce dataset size before sample generation.

### 5. Time alignment

For each track, `_build_samples()` aligns track timestamps to frame timestamps.

- `start`: shift the whole track so its first timestamp matches the first label timestamp
- `auto`: compare overlap counts with and without the same shift, then choose the better one
- `none`: do not shift

After alignment, only frame times within the track's time span are kept.

### 6. Track interpolation to frame times

For every valid frame timestamp inside the track coverage:

- `x`, `y`, `w`, and `h` are linearly interpolated with `np.interp`
- center coordinates are computed as:
  - `cx = (x + w / 2) / frame_width`
  - `cy = (y + h / 2) / frame_height`

This converts track boxes into normalized center targets in approximately `0..1`.

### 7. Sliding-window sample creation

The dataset then builds temporal windows of size:

- `history_steps + future_steps + 1`

Meaning:

- `history_steps` past positions,
- `1` anchor position,
- `future_steps` future targets.

For each window:

1. the anchor index is `history_steps`,
2. the anchor frame must have all configured representation images,
3. only the anchor frame image is loaded as model input,
4. past centers are stored for context and visualization,
5. future centers are stored as supervision,
6. `dt` is computed for each future step from consecutive timestamps.

Each stored sample contains:

- image paths for all representations at the anchor time,
- `past_centers`,
- `future_centers`,
- `dt`,
- constant camera intrinsics and camera pose priors,
- frame key, frame time, and track id.

### 8. Sample limiting and caching

If `max_samples` is set, `_apply_sample_limit()` randomly keeps only that many samples.

The resulting sample list can be cached as a pickle file under `cache_dir`. The cache key depends on config-like parameters, not on the underlying dataset file contents.

## Batch Preparation

`_collate_samples()` in both `train.py` and `visualization.py` stacks the per-sample tensors into batched tensors:

- `inputs[rep]`: `[B, 3, H, W]`
- `past_centers`: `[B, history_steps + 1, 2]`
- `future_centers`: `[B, future_steps, 2]`
- `dt`: `[B, future_steps]`
- `intrinsics`: `[B, 4]`
- `camera_pose`: `[B, 6]`

`past_centers` is batched but is not consumed by the model during training.

## Model Forward Pass

`PoseDynamicsProjector` performs the actual pose+dynamics prediction.

### 9. Per-representation encoding

For every representation listed in config:

- a separate `SmallCNN` encoder processes the image,
- each encoder uses stride-2 convolution blocks followed by global average pooling,
- the pooled feature is projected to `feature_dim`.

These features are concatenated and passed through a small MLP fusion block.

### 10. Camera correction heads

From the fused feature, the model predicts:

- `pose_delta` with 6 values,
- `intrinsics_delta` with 4 values.

Both are bounded with `tanh`:

- pose correction scaled by `0.1`
- intrinsics correction scaled by `0.05`

These deltas are applied to the configured priors:

- intrinsics become corrected `fx, fy, cx, cy`
- camera pose becomes corrected translation and roll/pitch/yaw

### 11. Latent motion heads

The same fused feature also predicts:

- initial 3D position `pos`
- initial 3D velocity `vel`
- a future acceleration vector for every prediction step

`z` in the initial position is constrained to be positive with `softplus(...) + min_depth`.

### 12. Temporal integration

For each future step:

1. read that step's `dt`,
2. update velocity using predicted acceleration,
3. update position using the updated velocity,
4. subtract camera translation to get relative world position,
5. rotate into camera coordinates using the roll/pitch/yaw rotation matrix,
6. store the camera-space 3D point.

This is a learned latent dynamics rollout, not a physically calibrated one.

### 13. Projection to 2D centers

For each predicted 3D point in camera coordinates:

- clamp `z` to at least `min_depth`,
- project with:
  - `u = fx * (x / z) + cx`
  - `v = fy * (y / z) + cy`

The output `pred_centers` has shape `[B, future_steps, 2]` and is directly compared to the normalized future centers from the dataset.

## Loss Function

`compute_losses()` uses four terms:

1. `center_loss`: L1 loss between `pred_centers` and target future centers
2. `pose_reg`: mean squared magnitude of `pose_delta`
3. `intr_reg`: mean squared magnitude of `intrinsics_delta`
4. `acc_reg`: mean squared magnitude of predicted accelerations

Total loss:

`center_weight * center_loss + pose_reg_weight * pose_reg + intr_reg_weight * intr_reg + acc_reg_weight * acc_reg`

Interpretation:

- the center term drives supervision,
- the other terms discourage large corrections or unstable dynamics.

## Training Loop

For each epoch:

1. move batch tensors to the chosen device,
2. run the model,
3. compute losses,
4. backpropagate,
5. update parameters with AdamW,
6. accumulate average metrics,
7. evaluate on the validation loader.

Checkpoint behavior:

- save `best.pt` whenever validation loss improves,
- save `epoch_XXX.pt` every `checkpoint_every` epochs.

Optional visualization behavior:

- every `vis_every` epochs, run `export_batch_visualizations()` on the validation loader.

## Visualization Flow

`visualization.py` supports manual inspection of predictions.

It:

1. loads config and checkpoint,
2. builds a dataset for either train or val split,
3. runs the model on the first batch only,
4. overlays three trajectories on the selected representation image:
   - past centers in blue,
   - ground-truth future centers in green,
   - predicted future centers in yellow.

This is useful for debugging whether the projection geometry is at least qualitatively plausible.

## What The Model Actually Learns

Despite the experiment name, the current model only sees the anchor-frame representation images. It does not ingest `past_centers` or a sequence of past images. So the model is learning:

- a mapping from one event-history image snapshot to latent camera correction,
- a latent initial 3D state,
- a latent acceleration rollout that best reproduces the future 2D center path.

That means the learned "pose" and "dynamics" are only weakly identifiable. Many different 3D explanations can produce similar 2D trajectories.

## Known Bugs, Risks, and Improvement Suggestions

### 1. Cache invalidation bug

The dataset cache key only depends on config-like values and not on source file contents, modification times, or split-file contents. If tracks, labels, or rendered images change without a config change, stale cached samples can be silently reused.

Relevant code:

- `data/track_projection_dataset.py`, `_cache_key()`

Suggested fix:

- include split file paths and mtimes, track file mtimes, and possibly label directory mtimes or a manual cache version in the cache key.

### 2. `history_steps` is not used for modeling

The dataset constructs `past_centers`, and the config exposes `history_steps`, but the model forward pass only consumes anchor images, intrinsics, pose priors, and `dt`. This makes the experiment description sound more temporal than the actual model is.

Relevant code:

- sample generation stores `past_centers`
- `train.py` batches `past_centers`
- `models/pose_dynamics.py` never reads them

Suggested fix:

- either feed past centers or past image features into the model, or rename/document the setting more explicitly as visualization/context only.

### 3. Visualization silently ignores samples beyond the first batch

`export_batch_visualizations()` calls `next(iter(loader), None)` and never iterates further. If `max_samples` is larger than `batch_size`, the extra requested samples are not exported.

Relevant code:

- `visualization.py`, `export_batch_visualizations()`

Suggested fix:

- iterate over the loader until `max_samples` images have been written.

### 4. Device selection is narrower than the config suggests

The code uses:

- configured device if CUDA is available,
- otherwise CPU.

That means a non-CUDA device string in config is effectively ignored whenever CUDA is unavailable.

Relevant code:

- `train.py`
- `visualization.py`

Suggested fix:

- honor the configured device directly, or fall back only after explicit validation.

### 5. Weak identifiability of latent 3D state

The only supervised signal is 2D future center reprojection error plus small regularizers. There is no direct supervision for depth, velocity, pose correction, or consistency across views/time. This can let the model fit trajectories with unrealistic latent geometry.

Suggested improvements:

- add multi-step consistency losses,
- add smoothness or bounded-velocity priors,
- supervise anchor center directly,
- compare predicted motion direction against observed past motion,
- add uncertainty estimation or calibration checks.

### 6. Constant camera priors per sample

Every sample receives the same configured intrinsics and pose prior. If the dataset has sequence-dependent or camera-dependent variation, the model must absorb all of it through learned deltas from image appearance alone.

Suggested improvements:

- support per-sequence or per-frame camera metadata,
- log learned delta statistics to see whether the priors are doing meaningful work.

### 7. Data-loading efficiency can become a bottleneck

All images are opened on demand in `__getitem__`, and `num_workers` defaults to `0` in the base config. This is simple but likely slow for larger runs.

Suggested improvements:

- increase `num_workers`,
- add pinned memory when using CUDA,
- consider image caching or precomputed tensor serialization if I/O becomes dominant.

### 8. Evaluation is fairly thin

Only averaged loss components are logged. That makes it hard to tell whether failures come from drift, one bad horizon step, out-of-frame projections, or dataset alignment problems.

Suggested improvements:

- log per-horizon error,
- log out-of-bounds prediction rate,
- log per-sequence metrics,
- save a small fixed validation subset for stable qualitative comparison across epochs.

## Practical Summary

At a high level, the program does this:

1. align track timestamps to frame timestamps,
2. interpolate tracks onto frame times,
3. turn interpolated boxes into normalized center trajectories,
4. cut those trajectories into anchor-plus-future windows,
5. use anchor event images to predict latent pose corrections and 3D motion,
6. project the predicted motion back to future 2D centers,
7. train by minimizing future center reprojection error.

The implementation is coherent for an exploratory experiment, but the current version is best understood as a 2D trajectory reprojection model with latent 3D variables, not as a strongly grounded 3D pose-and-dynamics estimator.
