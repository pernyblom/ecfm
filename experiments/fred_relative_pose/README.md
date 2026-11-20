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
