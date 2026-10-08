# Hierarchical event MAE

Pretrain on local event volumes at multiple spatial and temporal resolutions, then
run a linear probe on saved features or finetune the encoder. This experiment uses
ordinary transformer attention and additive learned encodings; it has no relative
position encoding or attention bias. It reuses the project's `Region`/`build_patch`
rendering and `region_worldmodel` THU loading, split and training utilities.

## Start with THU-EACT-50-CHL

Run from the repository root with the project environment (`.venv312/Scripts/python.exe`
on this Windows workspace, or `python` with `ecfm` installed):

```powershell
python -m experiments.hierarchical_mae.train --config experiments/hierarchical_mae/configs/thu.yaml
python -m experiments.hierarchical_mae.downstream --config experiments/hierarchical_mae/configs/thu_linear_probe.yaml --checkpoint outputs/hierarchical_mae/best.pt --mode linear_probe
python -m experiments.hierarchical_mae.downstream --config experiments/hierarchical_mae/configs/thu_finetune.yaml --checkpoint outputs/hierarchical_mae/best.pt --mode finetune
```

The starting configuration has 420 candidate tokens, a 128-dimensional encoder,
a 64-dimensional decoder, and random crops spanning 30–100% of the recording's
duration. The spatial base volume defaults to the full sensor. THU timestamps are converted
from microseconds to seconds. Empty voxels remain valid tokens with zero event count.
No event subsampling is applied, so counts describe the actual cropped volume.

For a CPU pipeline check, use `configs/smoke.yaml` in all three commands and the
checkpoint `outputs/hierarchical_mae_smoke/best.pt`. It uses eight training/validation
recordings, 18 tokens, one training batch and four test recordings. These limits
check execution, not representation quality. `--resume .../last.pt` resumes MAE
training with optimizer state and deterministic epoch seeds; increase `train.epochs`
to the desired total.

Pretraining and downstream selection use a seeded validation holdout from train.
Test recordings are evaluated only after choosing the best downstream checkpoint
on validation loss. The downstream runner preserves the pretraining holdout.
Pretraining saves `config.yaml`, `splits.json`, `metrics.jsonl`, `best.pt`, and
`last.pt`. Downstream runs save `best.pt`, an epoch-by-epoch `last.pt`, metrics and
`results.json`.

## Resume downstream training

Set `downstream.epochs` to the desired **total** number of epochs, then run:

```powershell
python -m experiments.hierarchical_mae.downstream --config experiments/hierarchical_mae/configs/thu_linear_probe.yaml --resume outputs/hierarchical_mae/linear_probe/last.pt
python -m experiments.hierarchical_mae.downstream --config experiments/hierarchical_mae/configs/thu_finetune.yaml --resume outputs/hierarchical_mae/finetune/last.pt
```

Resume restores the classifier/encoder, optimizer, completed epoch, training-loader
random state and best validation checkpoint. Training continues at the next epoch;
an interrupted partial epoch is repeated from the most recent completed checkpoint.
The best model is still used for final testing, even if no resumed epoch improves
validation. `last.pt` includes that best state and can be copied to a new directory.
Checkpoint writes are atomic. On Windows, transient file locks can prevent the
final replacement (`WinError 5`, access denied). The saver retries replacement up
to eight times over 4.5 seconds without rewriting the tensors or deleting the
previous checkpoint. If replacement still fails, it preserves the fully written
checkpoint as `last.recovery-<unique-id>.pt` (or `best.recovery-...`) and reports
its path. That file can be passed directly to `--resume`. A serialization failure
leaves the previous checkpoint intact and removes the partial file when possible.

The mode is inferred; `--mode` is optional and must match if supplied. New-format
downstream checkpoints contain everything needed to resume, so the original MAE
`--checkpoint` file is not required. Its saved digest lets linear probing reuse
the existing frozen-feature cache. Resume writes to the checkpoint's directory by
default; `--output-dir` selects a different directory.

**Existing runs made before resume support only have `best.pt`.** Pass that file
to `--resume` instead. Its saved weights and epoch are restored, but it has no
optimizer or loader-random state: the optimizer starts fresh and an explicit
warning is printed. This continues from the best saved epoch, which may be earlier
than the last epoch the old process completed. The next saved checkpoints include
full resume state. New-format `best.pt` is also resumable with its optimizer state.

Extend `downstream.epochs` freely. Device, worker count, output/cache locations and
cache rebuild options can change. Resume checks the model, data splits, crop
schedule, token selection, seed and downstream training settings for compatibility;
learning rates and batch size must match the saved configuration. If the configured
epoch total has already been reached, the restored best model is evaluated without
additional training.

## Spatial training crops

`data.crop_fraction` crops time only. To also crop space during pretraining and
finetuning, set:

```yaml
data:
  crop_fraction: [0.3, 1.0]
  spatial_crop_fraction: [0.7, 1.0]
```

Width and height fractions are sampled independently from this range (fractions
of sensor dimensions, not area), rounded down to whole pixels, and placed uniformly
at random within the sensor. The hierarchy tiles the resulting crop, with spatial
coordinates and geometry normalized relative to that crop. Events outside it are
discarded; event coordinates are translated without rescaling. Token counts and
output patch sizes stay unchanged, while event counts describe only the crop.
The minimum crop must leave at least `max_splits[0]` pixels in width and
`max_splits[1]` in height so every spatial cell has at least one pixel.

Omitting this option or using `[1.0, 1.0]` keeps the full sensor. Validation, test,
and frozen linear-probe feature extraction always use the full sensor spatially.
Spatial crops follow the existing per-recording/epoch seed and reusable training
view bank; patch-cache keys include the actual spatial crop bounds. Existing
patch and feature caches are automatically invalidated by the implementation change.

## Define a hierarchy

```yaml
hierarchy:
  max_splits: [4, 4, 8]
  levels:
    - {splits: [1, 1, 1], patch_size: 24, representations: [xy, xt, yt, cstr3]}
    - {splits: [2, 2, 2], patch_size: 16, representations: [xy, xt, yt, cstr3]}
    - {splits: [4, 4, 8], patch_size: 8, representations: [xy, xt, yt]}
```

`max_splits` means the maximum **number of cells** along `[x,y,t]`, not a binary
split depth. Each level must divide that maximum and refine the preceding level
by integer factors. The maximum defines the smallest addressable voxel, even if
the finest configured level is coarser. Pixel boundaries use a shared integer
lattice, so uneven sensor dimensions still tile exactly. Temporal cells partition
the sampled crop, and the recording's final event is included exactly once per
tiling/representation.

Each token owns its voxel and uses coordinates local to that voxel when rendering.
The patch size is an independent square output resolution, unrelated to its true
pixel/time extent. Levels may choose any positive patch size, in either size order,
and different representation lists. Patches are grouped by level/representation,
so a three-channel 24×24 CSTR token and a two-channel 8×8 `xt` token need no pixel
padding. Each group has its own patch projection and reconstruction head.

Supported representations:

| Representations | Channels and meaning |
| --- | --- |
| `xy`, `xt`, `yt` | Negative and positive event histograms, using existing patch normalization |
| `xy_p45`, `xy_m45`, `yt_p45`, `yt_m45` | Existing rotated local histogram projections |
| `cstr2` | Three channels: mean positive local time, zero, mean negative local time |
| `cstr3` | Same time channels, with voxel-max-normalized per-pixel event count in green |
| `cstr3_fixed` | Same, with green divided by `data.cstr_max_count` and clipped to one |

CSTR follows the channel semantics in `scripts/render_evt3_yolo_frames.py`, but
keeps floating point values instead of quantizing to an RGB file. CSTR time channels
are relative to the entire voxel duration, including empty leading/trailing time.

## Token encoding and reconstruction

The encoder sums patch content, a projection of `log1p(event_count)`, and geometry:

- Learned per-axis origin and extent tables, bounded by `max_splits`.
- Learned level and representation embeddings.
- An MLP over normalized position/extent `(x,y,t,dx,dy,dt)` and
  `log1p(start_seconds)`, `log1p(duration_seconds)`, `log1p(base_duration_seconds)`.

Time origin is relative to the sampled base volume. Absolute recording offset is
not used; the duration in seconds distinguishes a short crop from a long crop with
the same normalized layout. Encoder and decoder have separate geometry encoders.

Only selected visible patches and counts enter the encoder. The encoder physically
gathers them before attention, padding only to the largest visible count in the
batch. The decoder restores the complete token order with learned mask tokens and
known geometry. Masked event counts are targets, never part of decoder queries.

Patch MSE is averaged over pixels/channels per token, then averaged equally over
active level/representation groups. This prevents larger patches or the large
number of fine tokens from automatically dominating the patch objective. Count MSE
is measured in `log1p(count)` space over all target tokens, weighted by
`loss.count_weight`. Duration is known conditioning information, not a prediction
target. Selection exclusions and padding never become targets implicitly.

## Learn overlaps while also learning missing context

The default `mixed` strategy draws a policy independently for each example:

- With `overlap_probability: 0.5`, ordinary token masking permits parent/child and
  alternate-projection overlaps between visible and target tokens. This explicitly
  trains cross-level and cross-representation reconstruction.
- Otherwise, `subtree` masking selects cells at `masking.level`, targets their
  tokens and descendants, and excludes every visible token intersecting those
  target volumes. Visible subtrees still contain overlapping levels and planes.

Thus overlapping training examples are intentional. Their loss can be easier than
strict reconstruction, so compare policies as separate runs when assessing how
much is learned from missing spatial/temporal context. The logged mixed validation
loss uses fixed random choices each epoch.

Other strategies are `token`, `subtree`, and `voxel`. `voxel` has the same strict
exclusion rule but reconstructs only the chosen level. `ratio` is a token fraction
for token masking and a voxel fraction at the selected level for spatial masking.
At least one cell/token is held out and one remains. A spatial masking level needs
at least two cells. Strict masking generally excludes the full-volume root; this
is why overlap-permitting examples are part of the default mixture.

## Token selection and a future learned policy

`selection` controls MAE encoder inputs after reconstruction targets are chosen.
`downstream.selection` independently controls probing/finetuning inputs:

```yaml
downstream:
  selection: {strategy: activity, budget: 64}
```

Strategies are `all`, `random`, `activity` (highest event counts),
`activity_random` (activity-weighted random sampling), and `coarse`
(lowest level first). Budget zero means all eligible tokens; otherwise it is an
upper bound. Ties use canonical order. Random feature-cache selection is seeded
per recording, independent of cache extraction batch size. Activity selection uses
candidate counts as side information; report that sensing cost when comparing with
a policy that only observes coarse tokens.

Structure-based strategies `information` and `activity_information` rank raw
voxel neighbor support, entropy deficit, or shuffled-null-corrected
autocorrelation, alone or combined with event count. See
[INFORMATION_SELECTION.md](INFORMATION_SELECTION.md) for score definitions,
54-token presets, and a multi-seed linear-probe ablation suite.

The extensible contract is `TokenPlan(visible, target, gates=None)`, with boolean
`[B,N]` masks and optional differentiable nonnegative `[B,N]` weights. Visible and
target must be disjoint, with at least one visible token per example. MAE also
requires at least one target. Tokens can belong to neither set.

```python
from experiments.hierarchical_mae.masking import TokenPlan, make_plan, select

# The policy may first call model.encode/features with a coarse-only TokenPlan.
# layout.descendants(coarse_ids) identifies finer candidate tokens across planes.
# Produce [B,N] scores from those coarse features and candidate geometry.
base = make_plan(view, model.layout, cfg['masking'])
visible, gates = select(base.visible, model.layout, {'budget': 32},
                        view['log_counts'], scores=policy_scores,
                        straight_through=True, temperature=1.0)
plan = TokenPlan(visible, base.target, gates)
result = model(view, plan=plan)       # pretraining
features = model.features(view, plan=plan)  # downstream
```

The straight-through sigmoid gate provides a surrogate gradient through selected
scores, not through hard top-k indices. For dense soft training, keep eligible
tokens visible and supply continuous gates; this retains gradients but does not
save encoder attention compute. A policy must preserve the base plan's eligibility
to maintain strict masking. A staged policy that observes the root overlaps all
targets by construction, so train it with overlap-permitting masking.

The downstream [learned selection experiment](LEARNED_SELECTION.md) now trains an
activity-initialized policy conditioned on coarse features, using straight-through
relaxed top-k. It includes selector training, frozen-selector feature probing and
joint finetuning, with optimizer/checkpoint/cache support. MAE pretraining continues
to use its existing selection configuration. Rendering currently creates
all candidate patches before selection; encoder attention is sparse, but rendering
and the full MAE decoder are not. Lazy rendering can be added at the same interface
once the policy decides which candidate IDs to request.

## Patch preparation, workers and reusable crop caches

Patch generation partitions the events once per hierarchy level, shares each voxel
across its representations, and uses bincount histograms for `xy`/`xt`/`yt`.
This preserves the local representation and normalization semantics while avoiding
420 separate scans of the entire event crop. The initial THU configuration now
uses four persistent CPU workers and pinned transfers. Workers receive the epoch
with each sample index, so fresh crop seeds advance even with persistent workers.
Set `train.num_workers: 0` for serial loading, or adjust for your machine's CPU/RAM.

`data.patch_cache` is separate from the frozen-encoder feature cache:

```yaml
data:
  patch_cache:
    enabled: true
    dir: outputs/hierarchical_mae_patches
    train_views: 0
    rebuild: false
```

With `train_views: 0`, training retains fresh random crops every epoch and does
not write one-use training patches. Fixed validation/test/probe crops are cached.
For reusable training patches, choose a positive `train_views`, such as the eight
views in `configs/thu_cached.yaml`. Every recording then uses a deterministic bank
of random crops, selecting `epoch % train_views`. The first visits render and save
patches; later visits load them directly, skipping raw event loading and rendering.
Masking and token selection still run afresh on each pass. The crop bank remains
bounded even if disk caching is disabled, so cached and uncached results can be
compared using the same crop schedule.

An eight-view bank trades unlimited crop diversity for reuse. It is opt-in: the
base `thu.yaml` retains fresh crops. The full 420-token THU bank takes approximately
**3.9 GB** for eight crops × 1,518 train recordings plus 268 validation recordings.
Changing preprocessing or the crop schedule can create additional entries; old
cache namespaces are not automatically removed. No masks, model weights or encoder
features are stored in the patch cache.

To switch an existing pretraining run to the eight-view bank, restart with:

```powershell
python -m experiments.hierarchical_mae.train --config experiments/hierarchical_mae/configs/thu_cached.yaml --resume outputs/hierarchical_mae/last.pt
```

Resume accepts changes to cache settings and worker settings, including opting
into the crop bank. Model, hierarchy and other data semantics must still match.
To retain fresh crops while benefiting from the renderer/worker improvements,
use `configs/thu.yaml` instead. Changes take effect in a newly launched process.

The cache fills lazily; optionally prepare the entire bank before training:

```powershell
python -m experiments.hierarchical_mae.cache --config experiments/hierarchical_mae/configs/thu_cached.yaml --workers 4
```

Use `--split validation` to prepare only fixed validation crops, also supported by
the base config. Interrupted preparation can be rerun: complete entries are reused.
Cache keys cover the source path/statistics, exact crop interval, geometry,
representation settings, relevant implementation digests, and NumPy/PyTorch versions.
Writes use atomic replacement. Reads verify metadata, shapes, dtype and finiteness.
`rebuild: true` forces regeneration on every access; set it back to false to reuse
the rebuilt entries.

`train.log_every` prints progress every N batches (`0` disables intermediate logs).
Epoch metrics include `data_wait_seconds_per_batch`, `step_seconds_per_batch` and
`samples_per_second`. Data wait includes worker startup and cache misses; with
prefetching it measures time the main process actually waits, not total CPU work.
The step timing includes transfers, masking, model work and optimizer updates.

See [PERFORMANCE.md](PERFORMANCE.md) for measurements and the bounded profiling
command. These measurements isolate rendering, warm cache reads, data workers and
GPU training rather than extrapolating from model parameter count.

## Compare selection strategies with linear probing

The supplied `thu_linear_probe_random27.yaml`, `thu_linear_probe_activity27.yaml`,
`thu_linear_probe_activity_random27.yaml` and `thu_linear_probe_coarse27.yaml` inherit the same probe hyperparameters from
`thu_linear_probe.yaml` and change only `downstream.selection`. The base probe
config is the all-token baseline. Use `--checkpoint` to start a new head for each
strategy; `--resume` continues an existing probe and rejects a changed selection.

The current three-plane hierarchy (`cstr2`, `xt`, `yt`) has **411 tokens**:
3 at `[1,1,1]`, 24 at `[2,2,2]`, and 384 at `[4,4,8]`. A representation of a voxel
counts as one token, so the three representations of one voxel cost three tokens.

| Strategy | Behavior |
| --- | --- |
| `all`, budget `0` | All 411 tokens; baseline |
| `random`, budget `27` | Uniform random subset of 27 tokens across all levels/representations; fixed per recording in the feature cache |
| `activity`, budget `27` | The 27 largest `log1p(event_count)` values, selected separately for each recording |
| `activity_random`, budget `27` | Sample 27 distinct tokens with higher probability for higher activity |
| `coarse`, budget `27` | Exactly the 3 full-volume and 24 middle-level tokens; identical layout for every recording |

Activity scores use total voxel counts, not counts divided by area or duration, so
large/active volumes are favored. Different representations of the same voxel
have equal scores. Ties are resolved in canonical order: level, representation,
then t/y/x. Coarse selection uses the same tie-breaking within a level; for example,
budget 64 takes the first 37 fine tokens after the 27 coarse tokens, rather than a
balanced sample of fine representations. Budget 3 with coarse selects only the
full-volume tokens. Other useful budgets are 12, 64 and 128. Budget **0 means no
limit**, even with random/activity/activity_random/coarse; it does not mean select no tokens.

`activity_random` samples **without replacement**, using successive-draw weights
`w = (1 + log1p(event_count)) ** activity_power`. The default power is `1.0`.
Power `0` gives uniform random sampling; values between `0` and `1` weaken the
activity preference, and larger values strengthen it. The additive one gives empty
voxels nonzero weight; an all-empty recording is sampled uniformly. Probabilities
are normalized among the remaining eligible tokens at each draw. These are draw
weights, not exact marginal inclusion probabilities when choosing several tokens.
Gumbel top-k implements the sampling while respecting masking eligibility and the
budget. As with the existing random strategy, probe selections are reproducible
per recording and fixed in the feature cache; finetuning/pretraining resample.

```yaml
downstream:
  selection: {strategy: activity_random, budget: 27, activity_power: 1.0}
```

The strategy also works under top-level `selection` for pretraining. It uses the
existing log-count activity measure, so the preference is gentler than weighting
directly by raw event counts. It does not normalize for voxel size or duration.

All these linear probes use a fixed full recording with the default
`data.eval_fraction: 1.0`; training-only temporal/spatial crops and MAE masks are
disabled. Selection happens before encoder attention, and the selected encoder
outputs are mean-pooled into one cached feature per recording. Each head learns
from the features of its own selection policy. Random subsets are drawn once per
recording when extracting features, not every head-training epoch. Selection
settings and seed are included in feature-cache keys, so the presets can share
the feature-cache directory without mixing features.

If pretraining is still running, use **one stable snapshot** of its best checkpoint
for all comparisons. Pointing each probe at the changing `best.pt` can give the
strategies different encoders; do not modify or replace a snapshot while probes
use it. Take a copy between checkpoint saves and verify that it loads successfully.
Set `$checkpoint` below to that snapshot and choose a new output root for each
comparison checkpoint:

```powershell
$checkpoint = 'outputs/hierarchical_mae/selection_probes/pretrained_snapshot.pt'
$probeRoot = 'outputs/hierarchical_mae/selection_probes/comparison'

python -m experiments.hierarchical_mae.downstream --config experiments/hierarchical_mae/configs/thu_linear_probe.yaml --checkpoint $checkpoint --mode linear_probe --output-dir "$probeRoot/all"
foreach ($strategy in @('random', 'activity', 'activity_random', 'coarse')) {
    python -m experiments.hierarchical_mae.downstream --config "experiments/hierarchical_mae/configs/thu_linear_probe_${strategy}27.yaml" --checkpoint $checkpoint --mode linear_probe --output-dir "$probeRoot/${strategy}27"
    if ($LASTEXITCODE -ne 0) { throw "Probe failed: $strategy" }
}
```

Results are written to `<probeRoot>/<strategy>/results.json`, with training curves
in `metrics.jsonl`. Separate output directories prevent overwriting another probe's
head checkpoints. The presets currently inherit 500 epochs and learning rate 0.01;
change those in the base probe config to change them consistently for all strategies.
To try another budget, copy a preset, change its `budget`, and use a new output
directory. Compare validation results to choose a strategy/budget. Each invocation
also reports test results for its validation-selected head.

Feature extraction still prepares all candidate patches, so selection reduces
encoder work, not rendering on cache misses. Linear-head training itself uses saved
features. Extracting features while pretraining runs also competes for the same GPU.

## Cached probes and patch inspection

Linear probing always saves pooled frozen-encoder features to CPU `.pt` files.
Training the linear head subsequently reads those features without rerunning the
encoder. Cache keys include checkpoint contents, input file identity/statistics,
data/hierarchy/model/selection settings, seed, PyTorch version, and relevant source
digests. Writes are atomic; cache loads check shape, finiteness and labels.
`downstream.feature_cache.rebuild: true` forces regeneration.

Set `train.inspect_every: N` to save a PNG contact sheet and full precision `.pt`
artifact every N epochs (`0` disables). `inspect_per_group` bounds the number shown
per group; targets are shown first. The artifact retains every first-example patch,
prediction, count, geometry and mask. The PNG labels visible/target/excluded tokens,
patch size and duration; display values are clipped to `[0,1]` while saved tensors
retain their full range. Files are under `<output_dir>/patches/`.

Tests: `python -m pytest tests/test_hierarchical_mae.py -q`.

## Interactive reconstruction UI

Run the local browser inspector from the repository root:

```powershell
python -m experiments.hierarchical_mae.visualize --checkpoint outputs/hierarchical_mae/best.pt
```

The script loads the architecture, hierarchy, representations, patch sizes and data
settings from the pretraining checkpoint, then opens `http://127.0.0.1:8765`.
It uses the existing Python dependencies; no frontend build or external service is
required. Stop it with Ctrl+C. `--device auto` uses CUDA when available and CPU
otherwise; specify `--device cpu` or `--device cuda:0` to override. Other options:

```powershell
python -m experiments.hierarchical_mae.visualize --checkpoint path/to/best.pt --data-root datasets/THU-EACT-50-CHL --port 8766 --no-browser
python -m experiments.hierarchical_mae.visualize --checkpoint path/to/best.pt --config experiments/hierarchical_mae/configs/thu.yaml --split validation
```

`--config` is optional and supports inherited YAML configurations. Its model and
hierarchy must match the checkpoint. Use MAE pretraining checkpoints containing
the decoder (`best.pt` or `last.pt`), rather than downstream classifier checkpoints.

Choose a training recording from the searchable list. Both panes show every patch
in canonical level/representation/time/y/x order, grouped into spatial grids per
time slice. Click patches on the left or use **Toggle group** to select targets.
The right pane replaces only masked patches with model predictions; all visible
patches retain their original pixels. Scrolling in either pane moves the other to
the same position. The group filter affects display only; masks remain selected
across hidden groups. **Clear masks**, **Random 65%**, manual **Reconstruct**, and
**Auto update** support repeated comparisons without reloading the recording.

The default view uses the fixed evaluation crop (`data.eval_fraction`, normally
the full recording). **Training crop** uses the actual training temporal/spatial
crop sampler; its epoch field reproduces the deterministic per-recording seed,
including `train_views` crop banks. Loading a new recording or crop clears masks.
The inspector retains one rendered view in memory and disables disk patch caching.
The default list is the training subset after the seeded validation holdout;
`--split validation` switches to that holdout.

Manual masks use all remaining tokens as encoder context, independently of the
training selection budget. By default, overlapping parent/child and alternate
representations remain visible, matching overlap-permitting token masking.
**Exclude overlapping context** additionally removes every other token whose
spatiotemporal volume intersects a selected target. Excluded patches are marked
on the right and shown as originals for comparison, but are never encoder inputs
or reconstruction targets. Select at least one visible token; strict exclusion
of a full-volume root leaves none and reports an error.

Pixels use the same fixed RGB mapping as saved inspection sheets and are clipped
to `[0,1]` for display only. Hover over an original patch for event count/duration,
or a reconstructed patch for raw patch MSE and predicted `log1p(count)`. The status
bar reports group-balanced patch MSE and target log-count MSE from unclipped model
outputs. Empty masks show originals without invoking the MAE decoder.

UI inference/API tests: `python -m pytest tests/test_hierarchical_mae_visualize.py -q`.

For supervised, conservative changes to an existing activity-selected set, see
[the one-swap experiment](SWAP_SELECTION.md). It evaluates real swap benefits,
trains a policy with a no-change option, and compares fresh linear probes across
matched classifier seeds.

The [bounded set-selection policy](SET_SELECTION.md) extends this to learned
joint replacement sets with hard limits of 1, 2, or 4 swaps and label-free,
single-transformer-pass inference.
