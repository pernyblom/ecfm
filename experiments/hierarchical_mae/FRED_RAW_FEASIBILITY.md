# FRED directly from RAW: feasibility investigation

Investigated 2026-10-09 against this workspace. This is a design and bounded
performance investigation, not an implemented FRED training backend.

**Implementation update (2026-10-10):** [FRED_DATASET.md](FRED_DATASET.md) documents
the shared FRED loader, pretraining integration, and tested CenterNet/forecasting
backbone adapters. The inventory and implementation gaps below describe the
earlier investigation; RAW files for 181--189 have since been restored.

**Follow-up:** [FRED_INDEX_INVESTIGATION.md](FRED_INDEX_INVESTIGATION.md) validates
direct window extraction using the existing `.raw.tmp_index` bookmarks: 72/72
windows match sequential decoding on three recordings. Prefer evaluating this
existing-index backend before implementing the new full-state index discussed
below. The repository's production loaders still do not perform indexed reads.

**You can keep the original event recordings and train both pretraining and
downstream tasks without exporting event chunks or frame images.** The MAE still
needs its histogram/CSTR patch tensors, but these can be constructed in RAM.
The substantial work is efficient window access and task adapters, not a mandatory
conversion to an image dataset. Current training commands do not support this yet.

## What the existing code actually does

| Component | Existing behavior | Required change |
| --- | --- | --- |
| [MAE dataset](data.py) | Loads a complete NumPy recording through `region_worldmodel.data.load_events`, crops it, renders patches in RAM | Accept explicit event windows from a FRED source; separate rendering from loading |
| [MAE split reader](../region_worldmodel/data.py) | THU-style file/class entries, recording holdout | FRED folder splits, window manifest, no classification-label requirement for pretraining |
| [RAW decoder](../../src/ecfm/utils/evt3.py) | Whole-file decode, or stateful chunk decode written to a complete array file | Bounded iterator/ring buffer; robust indexed access if random windows are needed |
| [Frame renderer](../../scripts/render_evt3_yolo_frames.py) | RAW becomes a temporary float32 memmap before timestamp slicing and image export | Reuse representation/window semantics, not its whole-recording loading path |
| [Detection dataset](../object_detection/data/dataset.py) | Discovers timestamped YOLO labels, resolves image paths, loads PIL images | Build samples from labels plus available event intervals and render on demand |
| [Forecast dataset](../kalman_ml_forecasting/data/track_dataset.py) | Builds history/future boxes from tracks and label timestamps; learned inputs come from images | Preserve target construction; replace image-path resolution/loading and render-manifest checks |
| [MAE model](model.py) | `encode()` returns tokens, token IDs, padding; `features()` mean-pools tokens | Use tokens for localization, pooled/object-conditioned features for forecasting |
| [MAE downstream runner](downstream.py) | Classification probe/finetuning | New detection/forecast runners or adapters to the existing task trainers |

The MAE loss and transformer do not require PNGs. However, FRED pretraining is
not simply a configuration change: loading, sample identity, splits, cache keys,
and checkpoint split metadata currently assume a recording/class pair.

## Local data and measurements

Excluding synthetic folders 900+, the workspace has 231 numbered FRED folders.
222 contain `Event/events.raw`, totaling **98.81 GiB**. RAW file sizes range from
53.3 to 1,667.2 MiB, with median 244.2 MiB. Twelve folders have
`Event/output_events.npz`. Folders 181--189 lack RAW, have NPZ and cleaned tracks,
and have no `Event_YOLO/*.txt` labels in the expected directory. The canonical
train split includes eight of these folders; test includes 187. A RAW-only backend
must report these exclusions explicitly, or support the existing NPZ source and
separately resolve annotation availability. It cannot silently promise coverage of
the complete canonical splits.

Inspected recordings 0, 8, 66 and 180 declare EVT3, 1280 x 720. Their sidecars
declare index version 2.0, 2 ms bookmarks and per-recording `ts_shift_us` values.
For recordings 0, 8 and 180, label spacing is 33,333 microseconds and indexed
recording endpoints are approximately 112, 115 and 126 seconds. These are inspected
examples, not a duration survey of every recording.

`metavision_core` is not installed in `.venv312`; no SDK throughput was measured.

Bounded CPU experiments used `.venv312/Scripts/python.exe`, the current Python
decoder, and the **actual `configs/thu.yaml` hierarchy**: 411 tokens, three levels,
32 x 32 patches, `cstr2/xt/yt`, 192-dimensional embeddings. The older README and
PERFORMANCE document describe a different 420-token configuration; their timing
and cache-size figures should not be applied to this config or FRED.

* Sequential decoding of the first 16 MiB of recordings 0, 8 and 180, in 1 MiB
  chunks while retaining decoder state, took 3.65, 5.23 and 6.21 seconds respectively:
  **2.58--4.38 MiB/s**, roughly 1.0--1.1 million decoded events/s. These timings
  isolate decoding, excluding file-read time, final concatenation and rendering.
* For recording 0, complete shifted-time windows ending at 500,000 us were passed
  in memory through `HierarchyDataset.__getitem__`, with caching disabled,
  full-sensor geometry and no spatial crop. Each row below is the median of three
  calls. Loading was replaced in-process with the already normalized event array;
  no event file or image was exported.

| Window | Events | CPU patch preparation | Patch tensor payload |
| --- | ---: | ---: | ---: |
| `[466667, 500000)` us | 173,415 | 202 ms | 3.746 MiB |
| `[100000, 500000)` us | 2,114,076 | 1,421 ms | 3.746 MiB |

These are early-recording windows from one sequence, not representative training
throughput. The 16 MiB cap did not reach 500,000 us in recordings 8 and 180, so
their incomplete-window rendering results were excluded. No GPU training,
random-seek latency or multiworker throughput was measured. Initial microbenchmarks
also used 256 KiB prefixes; the larger measurements above are the useful results.

The evidence supports treating both decoding and rendering as bottlenecks.
Avoiding PNG writes saves storage and permits flexible windows; it does not remove
the cost of repeatedly preparing those windows. At the measured patch payload,
100,000 cached views would require about **366 GiB for patches alone**. A blanket
patch cache is not automatically a storage-efficient replacement for frames.

## Recommended event access design

Use one shared event source with an explicit contract, conceptually:

```text
read_window(sequence_id, start_us, end_us)
  -> events(x, y, integer t_us, polarity), sensor geometry, clock metadata

render_hierarchy(events, start_us, end_us, spatial_crop, hierarchy)
  -> the existing patches / metadata / log_counts / valid_mask structure
```

Each supervised sample references a sequence and timestamp, plus boxes or a track
ID. It does not reference a generated image. Pretraining samples reference
unlabeled time windows. The manifest and seek index contain metadata, not event
copies. A strictly file-free derivative path can keep these in memory too, at the
cost of rebuilding them.

**Start with sequential/block sampling and bounded RAM reuse.** Each worker owns
a reader and a rolling decoded-event buffer. Decode new events once, retain the
history needed by subsequent windows, and evict older events. Shuffle sequence or
block order and use a bounded sample shuffle buffer; split workers/ranks explicitly
so they do not duplicate complete streams. Standard globally shuffled map-style
indices plus persistent workers do not guarantee useful locality. Open reader
handles inside workers, which also accommodates Windows spawn.

For example, adjacent 400 ms contexts at 33.333 ms label spacing overlap by about
92%. Independent reads can decode the same events roughly twelve times. A rolling
buffer removes that repeated decode; it does **not** automatically remove repeated
histogram construction. Several tracks at the same anchor can share decoding and,
for a full-scene encoder, the same encoder result. Object-specific crops need their
own rendering/encoding or an object-query mechanism.

For arbitrary shuffled windows, add **real indexed seeking**. EVT3 stores changes
relative to decoder state, so arbitrary byte offsets are insufficient. A custom
index can record byte offsets plus complete decoder state at chunk boundaries,
including time wrap, row/vector state and pending continuation state. Seek to a
preceding checkpoint, decode forward and filter the requested half-open interval.
Index creation requires an initial sequential scan but writes no decoded events.
Version and fingerprint the index against the source and decoder implementation.

The existing `.raw.tmp_index` files are useful, but our renderer only reads their
timestamp shift and approximate time bounds. It does not use them to restore
decoder state. Do not treat their bookmark offsets as checkpoints compatible with
the custom Python decoder without validating SDK format and restart semantics.

Two backend options are reasonable:

* **Native SDK decoding:** benchmark it first if its installation fits the target
  training machines. `EventsIterator`/`RawReader` provide bounded event batches.
  But the inspected OpenEB Python `RawReader.seek_time()` calls `_advance()` and
  repeatedly runs the decoder while dropping events. Its name does not guarantee
  indexed random access. Validate a lower-level indexed-seek path separately.
* **Repository-owned decoding:** the existing stateful decoder is a useful
  correctness reference and can support a streaming prototype. For sustained
  training, investigate a compiled decoder loop and checkpoint index. Pure Python
  per-word decoding at the measured rate is a significant limitation.

Official references: [EVT3 stateful format](https://docs.prophesee.ai/stable/data/encoding_formats/evt3.html),
[event I/O API](https://docs.prophesee.ai/stable/api/python/core/event_io.html),
[OpenEB reader source](https://github.com/prophesee-ai/openeb/blob/master/sdk/modules/core/python/pypkg/metavision_core/event_io/raw_reader.py),
[timestamp shifting](https://docs.prophesee.ai/stable/data/streaming_decoding/timestamp_shifting.html).
The upstream source reference follows `master`; recheck the installed version.

## Pretraining changes

1. Extract a reusable renderer from `HierarchyDataset.__getitem__`. Keep the
   existing group layout, masking and reconstruction loss.
2. Add a FRED sequence/window dataset and sequence-aware loader. Choose physical
   durations (for example 33--400 ms as an initial experiment), rather than cropping
   30--100% of a two-minute recording. Sample times beyond annotated frames too.
3. Select train/validation/test sequences before constructing windows. Reserve
   test recordings from pretraining for an ordinary inductive evaluation; using
   unlabeled test recordings would be a different, explicitly transductive protocol.
4. Decouple pretraining splits from `downstream.num_classes`. Extend cache and
   sample IDs to include exact integer window bounds, source/time-shift provenance,
   spatial crop and rendering settings. Existing cache APIs are recording-based.
5. Profile decoder, rendering, loader wait and GPU step independently. The current
   renderer generates all candidate patches before token selection. A small encoder
   token budget does not currently reduce rendering cost. MAE also needs target
   patches; lazy downstream rendering has a different opportunity from pretraining.

The existing CSTR renderer accumulates arrays at sensor/voxel resolution before
resizing. Larger FRED sensors and event counts can therefore cost substantially
more even when output patches have the same dimensions. Binning directly at output
resolution or using GPU histograms may help, but changes in normalization/resizing
semantics need deliberate validation rather than being called equivalent.

## Downstream detection

At each `Event_YOLO` timestamp t, load a causal event interval `[t-W, t)`, render
the hierarchy, and pair it with boxes at t. Empty annotation files are valid
negative examples. Jointly transform boxes and events for spatial augmentation;
retain crop origin/scale for mapping predictions to full-sensor coordinates.

There are two distinct levels of integration:

* Rendering existing full-frame representations directly into tensors can preserve
  the existing CNN detector. This removes image exports but does not evaluate the
  MAE representation.
* To evaluate the pretrained MAE, use `encode()` tokens plus their IDs/geometry and
  padding mask as memory for object queries predicting boxes/objectness. Reuse
  existing matching, box losses and evaluation where compatible. Mean-pooled
  `features()` is available for a simple baseline, but discards the explicit token
  layout needed for a stronger localization head. A CenterNet-style head requires
  an additional mapping/fusion of multilevel tokens onto a spatial grid.

The finest current 4 x 4 spatial cells span roughly 320 x 180 sensor pixels in
FRED, represented by 32 x 32 patches. Small UAV localization is a modeling risk.
Evaluate finer spatial levels, larger patches or an explicitly designed refinement
stage. Changing representation groups/patch sizes also changes checkpoint
compatibility. No RAW access design by itself establishes detection quality.

## Downstream forecasting

Reuse `cleaned_tracks.txt`, history/future boxes, time deltas, Kalman baselines,
residual rollout and metrics. Replace the image feature branch with MAE features
from the causal context. Condition on observed box/history/filter state; for
multiple objects, object-conditioned pooling or queries should be compared with
one pooled scene vector. A frozen encoder plus a small residual head is the
simplest first representation test; follow with joint finetuning.

The current forecasting dataset already supports target construction without
requiring representations (`require_representations`), which helps reuse the
label side. It still needs a new event-input provider and a model adapter. Disabling
image checks alone does not supply MAE inputs.

A longer MAE time volume and a sequence of short encoded windows are both possible.
Compare them while keeping observed time coverage and forecast anchors equal.
For frozen experiments, optionally cache features rather than images: one
192-dimensional float32 pooled vector is 768 bytes per view. Detection needs token
features/IDs/masks instead (411 x 192 float32 values are about 308 KiB before
metadata). These are optional derived artifacts, invalid whenever encoder weights,
input windows, selection or relevant preprocessing change. Finetuning cannot reuse
stale encoder features.

## Correctness gates before drawing model conclusions

* Preserve integer microseconds until subtracting the window origin. The current
  `_decode_evt3_words` returns float32 event matrices, so timestamp precision is
  lost before later shifting; float32 spacing at 100 million us is 8 us. A new
  source should expose integer timestamps, then normalize locally for the MAE.
* Apply RAW timestamp shifts exactly once. Do not assume an existing NPZ uses the
  same origin; the renderer already has special handling for shifted NPZ data.
* Normalize time using requested window bounds, including quiet edges and fully
  empty windows. The current NumPy loader normalizes to first/last observed event;
  directly reusing it on each FRED slice would change the physical time geometry.
* Keep every encoder input at or before the forecasting anchor, including coarse
  tokens and any optional RGB. A centered window contains future events. Masking
  fine future tokens does not make a coarse token spanning that future causal.
* Audit track alignment/interpolation separately: the existing loader uses
  `np.interp` and an optional overlap-maximizing clock alignment. For strict online
  claims, history observations must not derive from future track measurements.
  Label interpolation can remain a documented offline evaluation convention.
* Compare streamed and indexed windows against a trusted sequential decoder:
  boundaries, timestamp wraps, chunk/vector splits, empty windows and repeated
  worker reads. Compare rendering with the existing implementation with explicit
  allowance for intended float-tensor versus uint8-image differences. `xt_my` and
  `yt_mx` are not automatically equivalent to the MAE's `xt` and `yt` projections.
* Track usable/missing sequences and annotations in a saved manifest. Benchmark
  complete windows at later timestamps and across varied activity, not only starts.

## Suggested implementation order and effort

These are rough engineering estimates for someone familiar with the repository,
not measured development times or promises of model convergence.

| Deliverable | Difficulty / indicative effort |
| --- | --- |
| Sequential RAW window source, in-memory rendering, FRED MAE smoke run | Moderate; approximately 2--4 working days |
| Forecast adapter with frozen MAE and existing targets/baselines | Moderate; approximately 2--4 additional days |
| MAE token-based detector with existing task evaluation | Moderate to substantial; approximately 3--7 additional days |
| Robust fast indexed reader, precision fixes and throughput tuning | Substantial uncertainty; approximately 3--10+ days, depending on native backend support |

First prove complete-window/time alignment, then benchmark a locality-preserving
loader against the GPU's consumption rate. Only then decide whether native decode,
renderer optimization, a bounded RAM patch cache or an optional frozen-feature
cache is necessary. A fully disk-cache-free path is feasible, but efficient
end-to-end training has not yet been demonstrated by these measurements.

Recommended first downstream experiment: frozen MAE forecasting using the
existing track targets, comparing pretrained features, randomly initialized
features and the Kalman/history-only baselines under the same sequence split.
Then add token-based detection. This tests representation usefulness with limited
new task code before investing in a full foundation-model training pipeline.
