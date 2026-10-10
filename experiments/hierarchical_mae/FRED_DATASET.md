# FRED dataset and task backbones

The shared loader is [FREDDataset](../../src/ecfm/data/fred_dataset.py). It reads
indexed RAW windows directly and renders representations in memory. Hierarchical
MAE pretraining, CenterNet detection, and residual forecasting now use this loader.
PNG exports and decoded event files are optional rather than prerequisites.

## Frame access

```python
from ecfm.data.fred_dataset import FREDDataset

dataset = FREDDataset(
    'datasets/FRED', split='train',
    frame_source='event_labels',
    modalities=('events', 'event_boxes', 'rgb', 'padded_rgb', 'rgb_boxes'),
    event_window_s=0.033333,
    tracks_file='cleaned_tracks.txt',  # explicit generated annotation choice
    image_sizes={'cstr3': (640, 360), 'xt': (640, 224), 'yt': (224, 360)},
)
frame = dataset.get_frame('13', 25_166_415)
events = frame['events']                 # int64 [N,4]: x,y,shifted t_us,polarity
event_boxes = frame['event_boxes']       # boxes [N,4], classes, available, path
rgb = frame['rgb']                       # tensor, path, size, matched time/offset
padded_rgb = frame['padded_rgb']         # separate image and matching metadata
rgb_boxes = frame['rgb_boxes']           # normalized boxes in the RGB coordinate frame
```

`dataset[i]` returns the configured modalities for `dataset.frame_ref(i)`.
`frame_source='event_labels'` uses the actual annotation timestamps; empty YOLO
files remain valid negative detection examples. Missing labels return
`available=False`, which is different from a labeled frame with no objects.
`frame_source='grid'` creates a timestamp grid without requiring annotations,
including unlabeled recordings. `frame_stride_us` sets its spacing. Grid lengths
use the final indexed timestamp conservatively; arbitrary windows can also be read
with `dataset.events(sequence, start_us, end_us)`.

Boxes are normalized `[cx, cy, w, h]`. Event boxes belong to the event sensor;
RGB boxes belong to the original RGB image. RGB labels are looked up using the
matched RGB filename. The loader does not silently transform RGB boxes into padded
or event coordinates. Use returned image sizes and the dataset's known padding
transform if such a conversion is needed.

The RGB timeline subtracts the first timestamped image's wall-clock time, following
the existing FRED renderer's alignment convention. Matching defaults to the last
image **at or before** the event anchor, with a 50 ms tolerance. Missing or stale
RGB is returned as `None` by frame access. `rgb_match='nearest'` is available for
offline inspection; the RAW task adapters use causal matching. Alignment offsets
are returned for inspection, and `rgb_tolerance_us` is configurable. This convention
does not independently establish physical camera synchronization.

Frame records have variable event/box counts and optional images. Use
`collate_fred_frames` for a DataLoader returning a list of frame records; task
adapters below supply their own fixed-input collation.

## Representations and hierarchical patches

```python
images = dataset.representations_at(
    '13', 25_166_415, ['cstr3', 'xt', 'yt', 'xt_my', 'yt_mx'], lookback_s=0.4,
)
# Each image is a float32 CHW tensor in [0,1]. No file is written.

from experiments.hierarchical_mae.config import load_config
mae_cfg = load_config('experiments/hierarchical_mae/configs/fred.yaml')
view = dataset.hierarchical_patches('13', 25_166_415, mae_cfg, lookback_s=0.4)
# Existing MAE contract: patches, metadata, log_counts, valid_mask.
```

Supported event images reuse the frame renderer's implementation: XY, XT, YT,
rotated projections, CSTR2/3/fixed, XT_MY/YT_MX, event images, and supported grid
aliases. `rgb` and `padded_rgb` are also supported by `representations_at` and history
requests. Set `image_sizes`, `temporal_bins`, and `cstr_max_count` as needed.
Requesting a missing RGB representation raises an explicit error rather than
substituting future pixels or zeros.

Event image tensors retain the existing uint8 rendering semantics before division
by 255. MAE hierarchy patches use the existing floating point local-voxel renderer.
These are distinct contracts: `xt_my` is not an alias for MAE `xt`.

Every context is `[anchor-T, anchor)`. Integer timestamps are subtracted before
normalization; the requested duration is preserved even with empty time regions.
Contexts extending before recording time zero retain zero padding over that prefix.
Optional hierarchy spatial crops are `(x,y,width,height)` in sensor pixels. Events
are translated without resizing their coordinates; geometry is normalized to the
crop, matching the existing MAE renderer.

## Tracklets

```python
track_id, rows = next((tid, rows) for tid, rows in dataset.tracks('13').items()
                      if len(rows) >= 50)
anchor_us = int(rows[12, 0])
tracklet = dataset.get_tracklet(
    '13', anchor_us, track_id,
    history_steps=12, future_steps=24,
    representations=['cstr3', 'xt'], representation_window_s=0.033333,
    hierarchy_cfg=mae_cfg, hierarchy_lookback_s=0.4,
)
```

`history_steps=12` means **12 preceding observations plus the anchor**, giving 13
past boxes. `future_steps=24` gives 24 future boxes. `times_s` and the separate
`past_times_s/future_times_s` are relative to the anchor, whose time is exactly zero.
`observed_times_us` records the actual annotation timestamps. `step_us` defaults to
33,333; arbitrary positive step spacing can be requested when observations exist.

The loader matches observed rows within `match_tolerance_us` (default 32 us), never
interpolates through a gap, and rejects incomplete tracklets. This tolerance handles
minor rounding differences between annotation timestamps and the frame grid.
History image tensors are `[history_steps+1,C,H,W]` under `history_inputs`. The
hierarchy view under `inputs['hierarchy']` ends at the final observed anchor.
Future boxes are targets; no future event/RGB representation is constructed.

Track files use `time,track_id,x,y,width,height`, with time in seconds by default.
`track_time_unit` and `track_frame_size` explicitly configure other units or pixel
coordinate frames. No overlap-maximizing clock shift or interpolation is applied.

The default source is original `tracks.txt`; `cleaned_tracks.txt` must be selected
explicitly. By default, original `tracks.txt` must also exist before generated
track annotations are accepted. `tracks_source`, `sequence_info`, and task dataset
`excluded` records expose that provenance. The RAW forecasting adapter enforces
the original-track requirement.

## Pretraining

```powershell
.venv312/Scripts/python.exe -m experiments.hierarchical_mae.train --config experiments/hierarchical_mae/configs/fred.yaml
```

The FRED preset uses 33--400 ms random contexts, one nominal anchor per second,
and the existing MAE architecture/loss. Training anchors and spatial crops are
seeded by sample and epoch, including persistent workers. Validation is fixed.
The normal train loop, checkpoints, resume command and inspection outputs work.

Sequences are split before windows are generated. With the restored corpus and
the default seed, the canonical training partition has **156 training sequences /
28 validation sequences**, corresponding to **18,730 / 3,331 nominal windows**.
Canonical test's 47 sequences are reserved. Training-window durations/anchors are
resampled each epoch; these counts describe samples per epoch, not independent
recordings or a precomputed crop bank.

Folders 181--189 now have restored RAW files. Their lack of Event_YOLO and original
tracks does not prevent unlabeled pretraining. Folder 187 belongs to canonical
test and is therefore excluded from the preset's pretraining train/validation
partitions. The other eight can contribute.

`data.patch_cache` remains optional. Positive `train_views` enables a reusable crop
bank; zero retains fresh training windows and caches fixed validation only when
enabled. FRED cache keys include exact window bounds, crop, RAW and index metadata,
and relevant decoder/renderer digests. The default FRED preset disables disk patch
caching. Index readers are bounded by count and event caches by bytes and entry
count per dataset worker; no full recording is decoded into RAM or a temporary file.

## CenterNet and residual forecasting

```powershell
.venv312/Scripts/python.exe experiments/object_detection/train.py --config experiments/object_detection/configs/fred_hierarchical.yaml
.venv312/Scripts/python.exe experiments/kalman_ml_forecasting/train.py --config experiments/kalman_ml_forecasting/configs/fred_hierarchical.yaml
```

In each task preset, set `model.backbone.checkpoint` to
`outputs/hierarchical_mae_fred/best.pt` to use pretrained weights. The initial
`null` value builds a random backbone for controlled comparisons. `mae_config`
must describe the checkpoint's model and hierarchy; use the checkpoint's saved
`config.yaml` when the architecture differs from the preset. `freeze: true` freezes
the encoder while training the task head and pooled-feature projection; `false`
finetunes encoder parameters. Reconstruction modules are excluded from downstream
optimization.

`data.backend: raw` selects the new adapters. `representations: [hierarchy]` supplies
the nested hierarchy view to a single MAE branch, rather than separate CNNs per
projection. Input collation and transfers support the nested view. The new
backbone also works with another compatible hierarchy config/checkpoint.

**CenterNet:** encoder token IDs restore the spatial layout. The adapter averages
time/projection features at `fmap_level` (the finest level by default), producing a
spatial feature map for the existing heatmap/size/offset heads, losses, decoding,
and metrics. `image_sizes.hierarchy` sets the detector output canvas together with
`output_stride`; it does not resize MAE patches. The initial finest spatial grid
is only 4x4, so interpolation gives a first localization baseline. Finer hierarchy
levels or a richer spatial neck should be evaluated for small UAVs. The RAW adapter
currently uses single-class XY supervision and disables velocity prediction.

**Forecasting:** the pooled encoder feature replaces the image feature branch.
Observed boxes/history/filter features, residual acceleration rollout, configured
Kalman initializer, losses and metrics remain available. The preset uses 12 history
steps plus anchor and 24 future steps; `model.history_steps: 13` consumes all past
observations. A single shared scene encoding is combined with each track's history.

Both task presets hold out the same training sequences as the default pretraining
split using the same seed/fraction. Their `train_eval` split is disjoint validation,
and best checkpoints are selected there. Test evaluation is opt-in after selection.
Detection evaluation accepts `--split train_eval` (or `val` as a holdout alias) with
the RAW preset. Missing explicit task split definitions are rejected instead of
falling back to the entire corpus.

Supervised detection requires actual event annotation files, retaining empty
negative frames. Forecasting additionally requires original and selected track
files and complete observed windows. Unlabeled folders are excluded and recorded.
CNN task backbones can also use the RAW adapter with ordinary event/RGB
representations. The new RAW adapters currently require trailing windows, no
spatial cutout, and no box augmentation/decorrelation. Existing image visualization
tools expect image inputs; the hierarchy presets set `vis_every: 0`. RGB and
representation access above can provide inspection backdrops separately.

## Validation and smoke run

```powershell
.venv312/Scripts/python.exe -m experiments.hierarchical_mae.fred_smoke --workers 1
```

This bounded CPU run pretrains a tiny MAE, loads its checkpoint into both task
backbones, and executes one training/validation batch through each existing trainer
on different canonical-training sequences. It also exercises Windows worker spawn.
Outputs and metrics are under `outputs/fred_loader_smoke`; test sequences are not
used. All three paths completed with finite losses on 2026-10-10. This validates
execution and gradients, not representation quality or full-training throughput.

183 targeted/regression tests passed across the new FRED loader, indexed RAW,
shared image rendering, hierarchical MAE, CenterNet, detection data and forecasting.
Tests include provenance gates, missing versus empty annotations, track gaps,
causal RGB/history reads, anchor-zero times, cache behavior, checkpoint transfer,
frozen/finetuned encoder gradients and disjoint sequence splits.
