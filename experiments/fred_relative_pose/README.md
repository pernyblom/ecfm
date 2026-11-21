# FRED relative UAV pose

This experiment estimates a UAV's camera-relative flight state from a causal event cutout and a box tracklet. It learns through reprojection to FRED's 2D tracks. RGB is used in scene review and visualization.

## Train

Run from the repository root with the existing PyTorch environment. The base configuration points to `D:/Datasets/FRED` and uses the canonical training list, cleaned tracks, and a scene-stratified validation split by sequence.

```powershell
python -m experiments.fred_relative_pose.train --config experiments/fred_relative_pose/configs/base.yaml
```

The all-sequence run fits camera parameters for every reviewed scene with usable tracklets. It makes its own sequence-level holdout across the whole dataset. Its numbers are for this holdout and are independent of the canonical split benchmark protocol.

```powershell
python -m experiments.fred_relative_pose.train --all-sequences --forecast-ms 400 --samples-per-sequence 40 --workers 4 --epochs 18 --output-dir outputs/fred_relative_pose/physical_ablations/crop64_events400
```

Use `--forecast-ms 400` or `--forecast-ms 800` to set the horizon. `--event-window-ms` sets the trailing event duration independently of `data.history_ms`, which controls the box history. `--cutout-px` sets the square cutout size at representation resolution. `--history-only` trains the box-conditioned control. `--anchor-only` gives the event network the anchor box while withholding past box motion. The output directory stores `best.pt` (lowest validation center ADE), `last.pt`, the resolved config, sequence and tracklet provenance, per-epoch metrics, and the best checkpoint's per-scene camera estimates. Use a fresh output directory for each run. The indexed SQLite cutout cache is shared across compatible runs. Populating it from raw events dominates the first epoch.

The default sample uses a trailing 400 ms event stack with six bins for each polarity, rendered from indexed EVT3 using the Kalman forecasting loader. It crops 64 × 64 from a 640 × 360 representation, centered on the observed anchor box. The tracklet has 12 observed boxes including the anchor and 12 or 24 future boxes. The default velocity initialization fits the last six observed box centers and log extents. The state head also sees all 12 observed boxes relative to the anchor and the encoded event cutout. `--all-sequences` derives additional tracks from `coordinates.txt` where cleaned tracks are unavailable. It splits spans at time gaps, ambiguous detections, and large box jumps. For sequences without `Event_YOLO` timestamps it uses native event frame names or the coordinate timestamps. Derived tracks live under `outputs/`.

## Geometry and loss

- Camera axes are right, down, and forward. A camera is treated as stationary over each tracklet. One learned pinhole camera per scene provides `fx`, `fy`, `cx`, and `cy`. A learned camera-relative gravity direction starts from a framing-specific prior in `scenes.yaml`.
- The anchor state contains 3D position, linear velocity, a unit up vector, angular velocity about the two tilt axes, and positive thrust per unit mass. The model uses a yaw-symmetric ellipsoid with a fixed diameter and thickness. Rotation is represented by the component of angular velocity perpendicular to up. The default bound is 120°/s. Set `model.angular_speed_limit_deg_s` to zero for fixed up.
- The short rollout assumes constant angular velocity and specific thrust. Body up rotates under that rate, thrust acts along the evolving up vector, gravity acts against scene up, and linear drag acts against velocity. Acceleration is the sum of those contributions. The position, velocity, and up trajectories follow from the anchor state and these dynamics. Every forecast box comes from the perspective bounds of the propagated ellipsoid.
- Smooth L1 loss compares projected 2D centers and log extents against the anchor and future boxes. Extent supervision excludes clipped future boxes. The loss also regularizes camera parameters, the pose head, and depth near the camera. Validation logs center ADE and FDE in sensor pixels, a last-four linear extrapolator, the same 3D state under constant velocity, and inference with event features masked.

Absolute range, physical aircraft diameter, camera intrinsics, and attitude are underconstrained by monocular 2D boxes. Depth is reported in assumed body diameters. The reference diameter sets the gravity scale in the rollout. Box reprojection evaluates forecast geometry in the image plane. The RGB inspection is qualitative because FRED supplies no attitude or 3D position labels for these runs.

## Use the flight state

For inference with measured intrinsics and camera orientation, see [Relative pose inference with a calibrated static camera](static_camera_inference.md).

Load a checkpoint with `RelativePoseModel(checkpoint["config"], checkpoint["scenes"]["up_priors"])`, call `load_state_dict(checkpoint["model"])`, and switch to evaluation mode. A causal batch contains `past` boxes in normalized `(cx, cy, w, h)`, `past_time` in seconds relative to the anchor, the representation tensors in `events`, and the integer `scene` IDs from the checkpoint's scene list.

```python
import torch

with torch.inference_mode():
    state = model.estimate_state(batch)
    camera = model.cameras(batch["scene"])
    times_s = state.position.new_tensor([0.0, 0.1, 0.2, 0.4]).expand(len(state.position), -1)
    trajectory = model.propagate(state, times_s, camera["gravity_up"])
```

`state.position`, `state.velocity`, and `state.thrust` use assumed body diameters, diameters/s, and diameters/s², respectively. `state.angular_velocity` is in rad/s and `state.up` is a unit vector. `trajectory.acceleration` includes the anchor when the first query is zero. You can use the anchor state on its own or pass it to the flight integrator, which receives no image or box features. Older fixed-up checkpoints load with zero angular velocity.

## Train with a pose history

Use `--pose-history` to render a trailing event crop for each observed box. Each crop ends at that box's timestamp and is centered on that box. With the 100 ms configuration each of the 12 boxes has a corresponding `[12, 64, 64]` crop. The loader reuses the indexed cutout cache across overlapping tracklets.

The same encoder and pose head estimate position, up, velocity, tilt rate, and specific thrust independently at each observation. A second head fuses the physical states, boxes, and relative timestamps into the anchor flight state. The future projection loss trains the full chain end to end. An additional reprojection loss supervises each local state at its own timestamp and the next three labeled timestamps. Set `loss.history_rollout_steps` and `loss.history_rollout_weight` to change the default three steps and weight of 0.3.

The canonical benchmark starts from the resolved single-crop box-history config with script-generated tracks. It uses the same batch size of 96 and the same 45,046 training and 6,400 validation tracklets.

```powershell
$baseline = 'outputs/fred_relative_pose/canonical_track_sources/script_with_history'
$poseHistory = 'outputs/fred_relative_pose/canonical_pose_history'
python -m experiments.fred_relative_pose.train --config "$baseline/config.yaml" --pose-history --output-dir "$poseHistory"
foreach ($horizon in @(400, 800)) {
    python -m experiments.fred_relative_pose.eval --checkpoint "$baseline/best.pt" "$poseHistory/best.pt" --forecast-ms $horizon --output "$poseHistory/test_${horizon}ms.json"
}
```

### Canonical test comparison

The 18-epoch pose-history run selected epoch 12 by 400 ms validation center ADE. Validation ADE / FDE was 11.14 / 20.57 px, compared with 11.78 / 22.13 px for the single-crop model. Both test horizons have the same tracklet hashes as the earlier annotation-source comparison. The 800 ms test uses the frozen 400 ms checkpoint.

| Model | 400 ms center ADE / FDE (px) | 400 ms mIoU | 800 ms center ADE / FDE (px) | 800 ms mIoU |
| --- | ---: | ---: | ---: | ---: |
| Pose history + box history | 12.38 / 22.76 | 0.553 | 26.67 / 62.90 | 0.383 |
| Single crop + box history | 12.96 / 24.26 | 0.541 | 28.42 / 66.36 | 0.364 |
| Single crop + anchor box | 13.85 / 25.90 | 0.518 | 37.73 / 104.60 | 0.339 |
| Last-four linear extrapolation | 20.97 / 40.29 | 0.406 | 44.76 / 97.38 | 0.264 |

Against the matched single-crop box-history model, pose history lowered center ADE by 4.5% at 400 ms and 6.1% at 800 ms. Original-track ADE fell from 14.94 to 14.26 px at 400 ms and from 31.64 to 29.82 px at 800 ms. On coordinate-generated tracks it fell from 7.50 to 7.23 px and from 19.16 to 17.60 px, respectively. The two JSON reports include box errors, annotation-source metrics, and camera-prior metrics.

### Buffered inference

For deployment, call `model.estimate_observation(box, crops, camera)` once per new box. Keep the resulting `FlightState` with the box and its timestamp in a per-target buffer. Stack the last 12 states along the history dimension and call `model.fuse_observations(past, past_time, camera, observations)` once the buffer is full. State tensors and box history use batch and time as the first two dimensions. The camera dictionary can come from `model.cameras(scene)` or measured calibration as in the [static-camera guide](static_camera_inference.md#run-the-checkpoint). `model.propagate()` uses the fused anchor state for the forecast.

## Kalman forecasting with box and pose measurements

The [pose Kalman optimizer](optimize_pose_kalman.py) fits a classical 800 ms forecast using the frozen pose-history checkpoint above. Like the [box Kalman variants](../kalman_ml_forecasting/KALMAN_VARIANTS.md), it can tune on training sequences and select on a sequence-level holdout. It fits the same dynamics with box-only and box-plus-pose observations on identical train and validation windows.

- Each normalized center coordinate and scaled log box extent has a position, velocity, and acceleration state. The transition integrates exponentially decaying acceleration over the actual timestamp intervals.
- At every history timestamp the box observes position. The 3D pose is projected through the checkpoint camera at the observation time and 25 ms either side. These local projections supply box position, velocity, and acceleration measurements. The filter learns a same-axis error correlation between box position and projected pose position because both use the same box.
- The optimizer learns initial, process, and measurement noise scales and acceleration decay by minimizing box ADE divided by 30 px, box FDE divided by 65 px, and the IoU error divided by 0.65. The lowest validation objective selects each filter independently.
- Optimization uses the checkpoint's canonical training and validation anchors that have 800 ms labels. The test scores the frozen filters on the full canonical 800 ms test split. The learned pose-history forecast and the last-four extrapolator are scored on those same windows.

```powershell
python -m experiments.fred_relative_pose.optimize_pose_kalman `
    --checkpoint outputs/fred_relative_pose/canonical_pose_history/best.pt `
    --output-dir outputs/fred_relative_pose/canonical_pose_kalman `
    --run-test-on-best
```

The run saves the two fitted filters in `best.pt`, the optimization trace in `validation.json`, and the test metrics and fitted parameters in `report.json`. It caches the frozen per-box pose measurements for reuse during optimization.

The run used 39,681 training and 5,595 validation windows. Both filters selected epoch 24. The test scored 101,118 windows across all 47 canonical test sequences.

| 800 ms forecast | Box ADE / FDE (px) | mIoU | Center ADE / FDE (px) |
| --- | ---: | ---: | ---: |
| Optimized box-only Kalman | 47.34 / 103.30 | 0.316 | 44.53 / 101.13 |
| Optimized box-plus-pose Kalman | 34.96 / 74.61 | 0.353 | 31.62 / 71.97 |
| Frozen learned pose-history forecast | 30.31 / 66.06 | 0.383 | 26.67 / 62.89 |
| Last-four linear extrapolation | 58.69 / 116.89 | 0.264 | 44.76 / 97.38 |

The pose measurements lowered box ADE by 26.2% and box FDE by 27.8% relative to the matched box-only fit. On original tracks, box ADE dropped from 53.66 to 39.05 px. On coordinate-generated tracks, it dropped from 29.12 to 23.16 px.

### Three dimensional extended Kalman filter

Use `--space 3d` to filter the flight state in the camera's 3D frame. The 13-state filter tracks 3D position, velocity, and acceleration, plus two slopes for target up and their rates. It uses log range to keep depth positive and retains a full state covariance. Acceleration and tilt rate decay during prediction over the actual timestamp intervals.

At each observed timestamp the frozen pose supplies position, linear velocity, up, and tilt rate. Specific thrust, up, gravity, and drag give a 3D acceleration measurement. The box-update variant also uses the observed image center, log width, and log height. Its observation model projects a 3D ellipsoid and linearizes the resulting pixel bounds about the current 3D state. It learns how much of the box error is already shared with the pose projection. The pose-only update variant still obtains its poses using the anchor boxes.

The ellipsoid uses the checkpoint's assumed body diameter and half-height. Projecting the ellipsoid gives all four box coordinates after rollout. Its apparent size changes with range and tilt. Absolute size and distance share a monocular scale ambiguity, so the benchmark holds body dimensions fixed. A known target size or a range sensor would give an independent constraint for metric distance.

```powershell
python -m experiments.fred_relative_pose.optimize_pose_kalman `
    --checkpoint outputs/fred_relative_pose/canonical_pose_history/best.pt `
    --space 3d `
    --output-dir outputs/fred_relative_pose/canonical_pose_ekf_3d `
    --run-test-on-best
```

The same 800 ms image loss tunes both filters. Model and data outputs have the same layout as the image-space optimization. `PoseEKF3D.filter()` returns the posterior and covariance in filter coordinates. `PoseEKF3D.rollout()` returns Cartesian position, velocity, acceleration, and the unit up vector in body-diameter units. The existing `project_body()` maps that trajectory to the image. For a moving-camera extension, express incoming poses in a common tracking frame and use the camera extrinsics at each timestamp in the box observation model and future projection. The FRED runs use a static camera per tracklet.

Both 3D variants selected epoch 24. The canonical test used the same 101,118 windows and all 47 sequences as the image-space benchmark above.

| 800 ms forecast | Box ADE / FDE (px) | mIoU | Center ADE / FDE (px) |
| --- | ---: | ---: | ---: |
| 3D Kalman, pose updates | 37.52 / 77.21 | 0.323 | 33.07 / 72.52 |
| 3D EKF, pose and direct box updates | 37.70 / 77.61 | 0.322 | 33.38 / 73.16 |
| Image-space Kalman, boxes and pose | 34.96 / 74.61 | 0.353 | 31.62 / 71.97 |
| Learned pose-history rollout | 30.31 / 66.06 | 0.383 | 26.67 / 62.89 |

Direct box updates slightly worsened the combined projection scores. Box ADE went from 41.70 to 41.99 px on original tracks and from 25.49 to 25.35 px on coordinate-generated tracks. The two models use the same frozen per-box pose estimates.

## Review scenes

`scenes.yaml` covers sequences 0–230. Camera groups follow the view and framing, so the courtyard has several camera groups. Sequence 21 has an obscured view and gets its own entry. Sequence 111 is dark enough to defeat local feature matching, but contrast enhancement shows the skyline landmarks.

```powershell
python -m experiments.fred_relative_pose.review_scenes --root D:/Datasets/FRED --output outputs/fred_relative_pose/scene_review_all
python -m experiments.fred_relative_pose.review_scenes --root D:/Datasets/FRED --manifest experiments/fred_relative_pose/scenes.yaml --neighbors 0 --output outputs/fred_relative_pose/review_grouped
```

The review script uses original middle RGB frames where available and the middle padded RGB frame otherwise. It shortlists pairs by thumbnail correlation and sequence proximity, then measures SIFT inliers and a robust similarity transform's mean corner shift. Use OpenCV for the matching pass. The scene manifest reflects visual inspection of the contact sheets alongside these matches. A camera group is a framing hypothesis based on the images.

For example, sequences 82–83 have almost identical framing. Sequence 84 shares landmarks with 82 but shifts the view by about 52 pixels at 640 × 360. The manifest gives these framings separate camera models.

## Inspect in 3D and RGB

```powershell
python -m experiments.fred_relative_pose.viewer --checkpoint outputs/fred_relative_pose/forecast_800ms/best.pt --output outputs/fred_relative_pose/viewer_800ms
python -m experiments.fred_relative_pose.viewer --checkpoint outputs/fred_relative_pose/forecast_800ms_anchor_only/best.pt --output outputs/fred_relative_pose/viewer_800ms_anchor_only
python -m experiments.fred_relative_pose.viewer --checkpoint outputs/fred_relative_pose/forecast_400ms_full/best.pt --count 31 --output outputs/fred_relative_pose/viewer_400ms_full
```

Open `index.html` inside the export folder. The viewer runs offline and keeps the event-camera projection, RGB image, and 3D state at the same selected forecast step. Select **Zoom UAV** for a close-up. The camera-projection mode shows how the estimated shape and up axis appear in the event-camera view. The orbit views show up relative to scene up and the camera origin. Toggle the overlay to inspect the UAV silhouette. The aligned image comes from FRED's `PADDED_RGB` stream. Registration can drift for fast or nearby UAVs because the dataset image warp and timestamp pairing are approximate.

The viewer displays velocity, acceleration, up, tilt angular velocity, and specific thrust. Its `flight_states.json` exports the anchor states with units and camera parameters for the same examples.

The default examples favor visible UAVs with nearby RGB annotations and cover multiple scenes. Use `--sequence 215 --time 24 --count 1` to inspect close to a specified elapsed event time. Add `--track-id` to select a specific track, or `--rgb-offset-ms -50` to adjust the RGB lookup for a sequence. If moving the whole export, keep `index.html`, `viewer.js`, `data.js`, and `images/` together.

## Cutout and event-window ablations

Each run uses a 400 ms box history, a 400 ms forecast, six event bins per polarity, and the same 7,120 training and 2,080 validation tracklets. The `tracklet_sha256` entries in `splits.json` identify the selected anchors. The crop sizes below refer to the 640 × 360 representation, so 128 × 128 covers twice the width and height of 64 × 64 at the same spatial sampling. Shorter windows use finer temporal bins while keeping the same number of channels.

```powershell
$poseAblationArgs = @('--all-sequences', '--forecast-ms', '400', '--samples-per-sequence', '40', '--workers', '4', '--epochs', '18')
python -m experiments.fred_relative_pose.train @poseAblationArgs --event-window-ms 400 --cutout-px 64 --output-dir outputs/fred_relative_pose/physical_ablations/crop64_events400
python -m experiments.fred_relative_pose.train @poseAblationArgs --event-window-ms 400 --cutout-px 128 --output-dir outputs/fred_relative_pose/physical_ablations/crop128_events400
python -m experiments.fred_relative_pose.train @poseAblationArgs --event-window-ms 200 --cutout-px 64 --output-dir outputs/fred_relative_pose/physical_ablations/crop64_events200
python -m experiments.fred_relative_pose.train @poseAblationArgs --event-window-ms 100 --cutout-px 64 --output-dir outputs/fred_relative_pose/physical_ablations/crop64_events100
```

As a measure of visibility, 60.7% of held-out tracklets had all 12 annotated history centers inside the 64 × 64 cutout. With 128 × 128 this rose to 86.6%. This measures the crop's coverage of the annotation history.

| Cutout | Event window | Best epoch | ADE (px) | FDE (px) |
| --- | ---: | ---: | ---: | ---: |
| 64 × 64 | 400 ms | 6 | 14.64 | 28.52 |
| 128 × 128 | 400 ms | 11 | 14.46 | 27.90 |
| 64 × 64 | 200 ms | 11 | 14.52 | 28.11 |
| 64 × 64 | 100 ms | 6 | 14.42 | 27.96 |

The last-four linear baseline gave 19.01 px ADE on these tracklets. The same 64 × 64 and 400 ms model with the tilt rate set to zero gave 14.61 px ADE, compared with 14.64 px when it learned tilt rate. The 100 ms input reached the lowest ADE in this single-seed comparison. Setting its event features to zero at evaluation increased ADE to 17.20 px.

For the 817 tracklets with at least one history center outside the 64 × 64 crop, the 128 × 128 crop reduced ADE from 19.72 to 19.12 px. The 100 ms and 200 ms windows gave 19.29 and 19.41 px on that subset. Paired resampling of the validation sequences gave intervals spanning zero for each overall change relative to the reference. The run directory's `comparison.json` records the paired results.

```powershell
python -m experiments.fred_relative_pose.viewer --checkpoint outputs/fred_relative_pose/physical_ablations/crop64_events100/best.pt --output outputs/fred_relative_pose/physical_ablations/viewer_100ms
python -m experiments.fred_relative_pose.viewer --checkpoint outputs/fred_relative_pose/physical_ablations/crop128_events400/best.pt --output outputs/fred_relative_pose/physical_ablations/viewer_wide
```

The same flight-state model also trained at 800 ms on the canonical-training subset. With 3,320 training and 1,000 held-out tracklets it reached 42.92 px ADE, compared with 56.48 px for last-four extrapolation. That checkpoint is in `outputs/fred_relative_pose/physical_800ms`.

## Larger 100 ms run

The larger run used the 64 × 64 cutout, 400 ms box history and 400 ms forecast from the ablation. Raising the per-sequence caps from 40 to 320 for training and 160 for validation gave 55,796 training tracklets and 8,320 validation tracklets on the same sequence split. Training ran for 18 epochs and the best checkpoint was at epoch 16.

| Validation anchors | Ablation ADE / FDE (px) | Larger run ADE / FDE (px) |
| --- | ---: | ---: |
| Original 2,080 | 14.42 / 27.96 | 11.12 / 21.26 |
| Expanded 8,320 | 14.67 / 28.15 | 11.32 / 21.50 |

The resolved config, checkpoints, paired `comparison.json`, and viewer are in `outputs/fred_relative_pose/physical_crop64_events100_large`. The viewer covers one example per validation scene and exports the corresponding flight states.

### Larger run without box history

The matched ablation used the same 55,796 training and 8,320 validation tracklets and ran for 18 epochs. It disabled `model.use_box_motion` and `model.encode_box_history` while keeping the anchor box and the 230 px/s learned image-axis velocity limit. The anchor box defines the crop and initializes position and depth. With the box-motion seed removed, that limit applies to the entire learned initial image-axis velocity. On 23.0% of validation tracklets, at least one component reached more than 95% of the limit.

| Validation anchors | With box history ADE / FDE (px) | Anchor box only ADE / FDE (px) |
| --- | ---: | ---: |
| Original 2,080 | 11.12 / 21.26 | 12.60 / 23.88 |
| Expanded 8,320 | 11.32 / 21.50 | 12.89 / 24.25 |

Epoch 7 gave the best ADE for the anchor-only run. The paired evaluation verified that passing only the anchor box produced the same inferred state as passing the whole box history to this checkpoint. The config, checkpoint, comparison, and viewer are in `outputs/fred_relative_pose/physical_crop64_events100_large_anchor_only`.

## Canonical FRED test evaluation

The two canonical retrains use the 100 ms event window, 64 × 64 cutout, 400 ms forecast, and training caps from the larger matched run. Their configs use `canonical/train_split.txt` as the source for both optimization and sequence-level validation. The best checkpoint is selected by validation center ADE.

```powershell
$canonicalPose = 'outputs/fred_relative_pose/canonical_crop64_events100'
python -m experiments.fred_relative_pose.train --config "$canonicalPose/with_history/run_config.yaml"
python -m experiments.fred_relative_pose.train --config "$canonicalPose/anchor_only/run_config.yaml"
python -m experiments.fred_relative_pose.eval --checkpoint "$canonicalPose/with_history/best.pt" "$canonicalPose/anchor_only/best.pt" --output "$canonicalPose/test_metrics.json"
```

The evaluation takes the test list from the checkpoint config and checks sequence disjointness before constructing test tracklets. It scores all eligible windows on both frozen checkpoints and the last-four baseline, with center and full-box ADE/FDE in original 1280 × 720 sensor pixels and mean IoU. The report includes fitted-camera and camera-prior groups and identifies annotation sources. Test-only scenes use the configured camera priors. The test cohort uses the pose tracklet filters above, including coordinate-derived tracks where cleaned tracks are unavailable.

Both runs used 45,038 training and 6,400 validation tracklets across 143 and 40 disjoint sequences. Sequence 119 was excluded because no eligible annotation source was available. The test scored 109,562 tracklets across all 47 canonical test sequences.

| 400 ms forecast | Best epoch | Validation center ADE (px) | Test center ADE / FDE (px) | Test box ADE / FDE (px) | Test mIoU |
| --- | ---: | ---: | ---: | ---: | ---: |
| Events and box history | 14 | 11.25 | 12.94 / 24.26 | 16.94 / 27.79 | 0.542 |
| Events and anchor box | 8 | 12.64 | 13.97 / 26.17 | 18.49 / 30.64 | 0.516 |
| Last-four linear extrapolation | | 19.32 | 20.94 / 40.27 | 31.39 / 55.09 | 0.406 |

The four test-only scenes contribute 4,580 tracklets. With their camera priors, center ADE / FDE are 8.86 / 17.87 px with box history and 10.43 / 20.92 px with the anchor box. The evaluation report is `outputs/fred_relative_pose/canonical_crop64_events100/test_metrics.json`.

The published residual CSTR model reports 13.22 / 24.72 px center ADE / FDE and 0.531 mIoU on the 400 ms canonical test. The last-four baseline gives 20.43 / 39.38 px in the paper and 20.94 / 40.27 px here. Tracklet eligibility and annotation source affect comparisons across these results. [Residual Kalman Dynamics, Table 1](https://arxiv.org/html/2609.00839v1#S4.T1)

### Annotation source and rollout horizon

The original-track runs use the pre-existing `cleaned_tracks.txt` files in sequences that have `tracks.txt`. The expanded runs use those files and fall back to `coordinates.txt` where they are unavailable. The new expanded runs use the same segmentation and rows as [the coordinate conversion script](../../scripts/build_cleaned_tracks_from_coordinates.py). The experiment writes these tracks to its cache. I verified byte-for-byte parity with the original script on all 97 canonical sequences eligible for fallback. The previous expanded runs used a more conservative generator that also split tracks on abrupt motion and box-area changes and discarded timestamps with multiple boxes.

The four new runs keep the previous canonical sequence assignments and use the same original anchors across training-source variants. Both pairs select checkpoints by validation center ADE at 400 ms. The original-track pair uses 26,560 training and 4,000 validation windows. The expanded pair uses 45,046 training and 6,400 validation windows. The 800 ms test rolls each frozen 400 ms checkpoint farther using its original dynamics and calibration.

```powershell
$runs = 'outputs/fred_relative_pose/canonical_track_sources'
$previous = 'outputs/fred_relative_pose/canonical_crop64_events100'
$variants = @('script_with_history', 'script_anchor_only', 'original_with_history', 'original_anchor_only')
foreach ($variant in $variants) {
    python -m experiments.fred_relative_pose.train --config "$runs/$variant/run_config.yaml"
}
$checkpoints = @($variants | ForEach-Object { "$runs/$_/best.pt" })
$checkpoints += @("$previous/with_history/best.pt", "$previous/anchor_only/best.pt")
foreach ($horizon in @(400, 800)) {
    python -m experiments.fred_relative_pose.eval --checkpoint @checkpoints --compare-training-sources --forecast-ms $horizon --output "$runs/test_${horizon}ms.json"
}
```

The first checkpoint supplies the test annotation configuration. Within each horizon, all models are scored on the same canonical test windows. Each JSON report includes center and box ADE/FDE and mean IoU for the original-track, coordinate-generated, and combined groups. The 400 ms test has 110,050 windows and the 800 ms test has 101,118. Both cover all 47 canonical test sequences.

| Training tracks | Box input | 400 ms original | 400 ms generated | 400 ms all | 800 ms original | 800 ms generated | 800 ms all |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Original only | History | 15.56 / 29.00 | 9.90 / 19.28 | 14.05 / 26.41 | 32.81 / 74.99 | 22.91 / 52.72 | 30.26 / 69.25 |
| Original only | Anchor | 16.45 / 29.41 | 10.74 / 20.74 | 14.93 / 27.10 | 41.88 / 115.30 | 26.49 / 65.60 | 37.92 / 102.49 |
| Original + conversion script | History | 14.94 / 27.72 | 7.50 / 14.74 | 12.96 / 24.26 | 31.64 / 72.86 | 19.16 / 47.66 | 28.42 / 66.36 |
| Original + conversion script | Anchor | 15.93 / 29.60 | 8.10 / 15.75 | 13.85 / 25.90 | 43.14 / 120.43 | 22.17 / 59.02 | 37.73 / 104.60 |
| Original + conservative fallback (previous) | History | 14.98 / 27.84 | 7.38 / 14.38 | 12.96 / 24.25 | 31.71 / 73.28 | 18.60 / 46.04 | 28.33 / 66.26 |
| Original + conservative fallback (previous) | Anchor | 16.05 / 29.79 | 8.24 / 16.12 | 13.97 / 26.15 | 43.99 / 125.46 | 22.79 / 61.15 | 38.52 / 108.88 |
| Last-four linear baseline | Four boxes | 24.22 / 46.36 | 12.03 / 23.56 | 20.97 / 40.29 | 50.67 / 109.44 | 27.75 / 62.66 | 44.76 / 97.38 |

Values are center ADE / FDE in native sensor pixels. The original and generated test groups have 80,711 and 29,339 windows at 400 ms and 75,055 and 26,063 at 800 ms. The best epochs are 8 and 7 for the original-only pair and 12 and 9 for the conversion-script pair, with history first in both pairs. The generator audit is in `generator_audit.json` alongside the test reports.

The published event-frame and CSTR models report 28.62 / 61.96 px and 28.84 / 62.50 px at 800 ms. The script-expanded model with box history gives 28.42 / 66.36 px across the combined test cohort and 31.64 / 72.86 px on original tracks. The publication uses horizon-specific training and a different set of eligible test windows. [Residual Kalman Dynamics, Table 1](https://arxiv.org/html/2609.00839v1#S4.T1)

## Earlier fixed-up results

ADE averages center distance over the forecast. FDE measures it at the last forecast step. Both use original 1280 × 720 sensor pixels.

The 400 ms all-sequence run used 14,232 training and 4,160 validation tracklets across 178 and 52 sequences, respectively. All 37 scene cameras were fitted and 31 scenes supplied held-out sequences. The run excluded sequence 119 because it has only interpolated coordinates. The 31-example viewer includes one tracklet per validation scene.

| 400 ms run | Best epoch | ADE (px) | FDE (px) |
| --- | ---: | ---: | ---: |
| Last-four linear extrapolation | | 19.03 | 37.34 |
| Box history only | 20 | 16.21 | 31.48 |
| Events and anchor box | 6 | 13.87 | 26.69 |

Validation ADE fell from 17.30 px to 13.87 px and reached 14.23 px at the final epoch. Masking events at the best checkpoint raised ADE to 40.51 px. Using constant velocity with the same 3D starting state raised ADE to 15.03 px.

The 800 ms runs used 13,280 training and 3,125 validation tracklets across 83 and 25 sequences, respectively. Their camera tables fitted 20 scene groups.

| 800 ms run | Best epoch | ADE (px) | FDE (px) |
| --- | ---: | ---: | ---: |
| Last-four linear extrapolation | | 55.01 | 121.39 |
| Box history only | 22 | 41.08 | 87.47 |
| Events and box motion | 10 | 35.90 | 76.87 |
| Events and anchor box | 6 | 34.50 | 74.91 |
| Events and box motion, with drag and an upright prior | 12 | 36.51 | 77.85 |
| Events and anchor box, with drag and an upright prior | 12 | 34.61 | 74.84 |

For the events-and-anchor run, validation ADE fell from 40.72 px to 34.50 px, then reached 36.71 px at the final epoch. Masking event features at the best checkpoint raised ADE to 88.23 px. Using constant velocity with the same 3D starting state raised ADE to 41.68 px. The run directory's `config.yaml` records the previous model and training settings.

With events and the anchor box, adding linear drag of 0.65/s, a 45° tilt limit, and a 0.2 tilt prior weight changed mean tilt from 41.9° to 15.5° while ADE differed by 0.11 px. In an RGB spot check of the base run, the anchor at 18.77 s in sequence 91 showed a projected up axis leaning to the left, consistent with the visible banking direction. The anchor at 11.47 s in sequence 222 showed a nearly upright axis. Other cases had substantial overlay offsets or different tilt magnitudes under the two priors.
