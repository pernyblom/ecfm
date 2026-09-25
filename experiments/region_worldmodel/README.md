# Region sensing world model on THU

This separate experiment learns how an event representation changes when an
action moves or resizes its sensing regions. It reuses ECFM's region sampler,
histogram projections, masked region-token encoder, and relative attention bias.
It has no dependency on the FRED experiment in `experiments/leworldmodel`.

## Interpretation and research assessment

For an event volume `E` and ordered region set `R`, construct observations
`o = render(E, R)` and `o' = render(E, action(R))`. The event volume stays fixed;
the target histograms are extracted again from the new windows. Moving metadata
while retaining the original patches would describe a different experiment.
Region identities correspond across the pair, including their projection planes.

This is a promising controlled test of active-sensing representations: there are
unlimited labeled transformations without requiring physical action recordings.
It learns changes in observation under sensing actions, rather than the temporal
evolution of the physical scene. Rotations mixing time and space describe offline
queries into a recording, not necessarily causal operations a live sensor can do.
Regions moved into unseen content introduce uncertainty that a deterministic MSE
predictor can only average over. Keep the first actions small and overlapping.

The most serious shortcut is encoding region geometry and ignoring events.
Another is becoming invariant to every action. Neither a small prediction loss
nor a healthy latent variance alone proves useful learning. The optional masked
patch reconstruction objective gives content supervision; downstream accuracy,
content ablations, and action ablations are needed to judge success.

The design is **LeWorldModel-inspired**, not an exact reproduction. LeWM offers
an economical end-to-end latent prediction objective with Gaussian regularization,
which suits training a custom event encoder from scratch.
See the [official LeWorldModel implementation](https://github.com/lucas-maes/le-wm).
As of September 2026, [V-JEPA 2.1](https://arxiv.org/abs/2603.14482) provides an
updated video representation recipe with dense features, and
[V-JEPA 2-AC](https://github.com/facebookresearch/vjepa2) provides action-conditioned
video prediction. Neither establishes superiority for this event-region task.
My recommendation is to establish this small baseline before importing a large
RGB-video backbone. Dense feature supervision is a useful future comparison.

## Actions and coordinates

The action catalog is an ordinary YAML list. Each entry has a unique `name` and
a `type`; multiple magnitudes and axes can coexist. Training samples uniformly
from the list. Validation evaluates every action for every held-out recording.

```yaml
actions:
  - {name: identity, type: identity}
  - {name: x_ccw10, type: rotate, axis: x, degrees: 10}
  - {name: x_cw10, type: rotate, axis: x, degrees: -10}
  - {name: x_ccw20, type: rotate, axis: x, degrees: 20}
  - {name: custom20, type: rotate, axis: [1, 2, 1], degrees: 20}
  - {name: smaller, type: scale, factor: 0.9}
  - {name: larger, type: scale, factor: 1.1}
  - {name: spatial_only, type: scale, factor: [1.1, 1.1, 1.0]}
  - {name: later, type: translate, offset: [0, 0, 0.05]}
```

- Coordinates are `(x / width, y / height, normalized_time)`. One full temporal
  recording therefore has the same rotation-space extent as the image width or
  height. This choice is explicit, not a conversion between physical units.
- A rotation moves **centers** around `(0.5, 0.5, 0.5)` using Rodrigues' formula.
  Windows remain axis aligned with unchanged sizes and projection planes.
  `x`, `y`, `t`, and nonzero custom three-vectors are supported; vectors are
  normalized. Positive degrees follow the right-hand rule: CCW looking from the
  positive axis toward the origin in this coordinate system. Sensor y increases
  downward, so an image viewer's intuitive CW/CCW may look reversed.
- Scaling multiplies window extents about each window's own center. `0.9` means
  10% smaller; `1.1` means 10% larger. A three-vector scales x/y/time independently.
  It does not scale the distances of region centers from the volume center.
- Translation offsets are fractions of the full x/y/time extent.
- Boundary handling: clamp sizes to the volume, then shift each window inside
  it. Spatial origins/extents are rounded to pixels. Empty windows remain valid
  observations; only padding is invalid. This preserves token correspondence but
  makes boundary actions non-invertible and can make some actions effectively
  identities. Check per-action results, especially for large windows.
- The predictor receives 15 continuous values: flattened `(rotation - I)`,
  log x/y/time scale factors, and translation. Identity is the zero vector.
  Catalog indices are only for reporting, so unseen magnitudes can be evaluated
  without changing model dimensions. Prediction quality there is not guaranteed.

`configs/thu.yaml` includes CW/CCW 10 and 20 degree rotations around all three
axes, a custom-axis rotation, shrink/enlarge, identity, and translation.

## Architecture and loss

The shared EventMAE encoder embeds two-channel patches, nine metadata values,
and plane IDs using relative attention and no token-order position embeddings.
Metadata is the existing `(x, y, dx, dy, t, dt, t_seconds, dt_seconds,
sequence_seconds)` representation; x/y/t are window origins, not centers.
The predictor is an MLP applied to each contextualized region token and action:

```text
z = encoder(source, optional content mask)
z_target = encoder(resensed target)
predicted_delta = MLP(concat(z, action_vector))
z_predicted = z + predicted_delta
```

Training minimizes valid-token MSE between `z_predicted` and `z_target`, plus a
SIGReg-style characteristic-function integral on the valid-mean-pooled source
and target embeddings across recordings. Gradients flow through **both** views;
there is no stop-gradient or EMA. The numerical regularizer is local to this
experiment and not claimed to be identical in normalization to the paper.
`loss.regularizer_weight` will need tuning with batch size and latent dimension.
Use enough independent recordings per batch; the smoke batch is not a sensible
anti-collapse training regime.

`train.mask_ratio` hides source patch content while retaining metadata/planes.
`loss.reconstruction_weight` optionally adds masked source-patch reconstruction
through the existing MAE decoder. Defaults are 0.25 and 0.1. Set reconstruction
weight to zero for the two-term latent objective; also set mask ratio to zero
for an unmasked LeWM-style baseline. Validation uses full, unmasked observations.
No count reconstruction loss is used here.

`model.features(view)` returns valid-mean-pooled encoder latents `[B, D]` for
classification. `model.predict(tokens, action)` returns next token latents;
subtract input tokens to obtain the predicted change. These are one-step
predictions; repeated rollouts are possible algebraically but not trained or
validated by this experiment.

## Running

Run from the repository root with the environment installed (`pip install -e .`).
On Windows the tested interpreter is `.venv312/Scripts/python.exe`; the commands
below use `python` assuming that environment is active. No new dependencies.

```powershell
# Bounded test with the actual THU files, then full training:
python -m experiments.region_worldmodel.train --config experiments/region_worldmodel/configs/smoke.yaml
python -m experiments.region_worldmodel.train --config experiments/region_worldmodel/configs/thu.yaml

# Resume at the next epoch; increase train.epochs in the config if needed:
python -m experiments.region_worldmodel.train --config experiments/region_worldmodel/configs/thu.yaml --resume outputs/region_worldmodel/last.pt

# Frozen backbone, single linear classification layer:
python -m experiments.region_worldmodel.downstream --config experiments/region_worldmodel/configs/thu.yaml --checkpoint outputs/region_worldmodel/best.pt --mode linear_probe

# Train the encoder plus a linear classification layer:
python -m experiments.region_worldmodel.downstream --config experiments/region_worldmodel/configs/thu.yaml --checkpoint outputs/region_worldmodel/best.pt --mode finetune

# Re-evaluate validation, including unseen angles/axis/scales:
python -m experiments.region_worldmodel.eval --checkpoint outputs/region_worldmodel/best.pt --actions-config experiments/region_worldmodel/configs/heldout_actions.yaml --output outputs/region_worldmodel/heldout_actions.json
```

Use `smoke.yaml` and `outputs/region_worldmodel_smoke/best.pt` together for bounded
downstream checks. Configuration files may `extends: thu.yaml`; nested mappings
merge and lists replace. All dataset/output paths are relative to the working
directory, while `extends` resolves relative to its YAML file.

The loader reads THU `train.txt`/`test.txt` entries and resolves file basenames
under `data.root`, matching this repository's THU layout. Missing files and
duplicate recordings fail explicitly. Classification labels must be zero-based
indices matching `downstream.num_classes`. Pretraining ignores labels.
Timestamps are normalized over the complete recording before optional event
subsampling, and relative timestamps are computed before float32 conversion.

A seeded recording-level holdout is carved from the training list before SSL
or supervised training. The test list is used only for final downstream scoring
(or explicitly using `eval --split test`). This is not an additional subject-wise
validation protocol; test separation follows the provided THU lists. Keep the
same `split_seed`, subset, and split settings when comparing SSL and downstream
runs. For a subject-wise validation study, prepare appropriate lists first.

Source regions/actions vary deterministically with recording index and epoch
during SSL. Validation is fixed across epochs and enumerates the entire catalog.
The linear probe freezes **all** backbone parameters and keeps it in eval mode,
with fixed source regions. Finetuning resamples training regions each epoch and
uses `downstream.encoder_lr` separately from the head learning rate. Both use
unmodified observations for classification.

Training writes resolved `config.yaml`, `splits.json`, `metrics.jsonl`, `best.pt`,
and `last.pt`. SSL checkpoints include optimizer state, epoch, configuration,
and validation metrics. Best SSL is selected on prediction + regularization;
best downstream is selected on validation cross-entropy, followed by test once.
Downstream writes under `linear_probe/` or `finetune/`, including `results.json`.
Use separate output directories for independent runs; metric logs append.
`train.max_batches` and `downstream.max_batches` bound training only. Validation
always covers its selected recordings; `max_test_samples` is for smoke runs.

Raw files are loaded and both views rendered on demand; no persistent token
cache is shared with other experiments. This favors correct action pairing over
throughput. Validation costs roughly one rendering pair per recording/action.
Increase `num_workers` if I/O/tokenization dominates, accounting for worker RAM.

## Fixed downstream layouts

Downstream region sampling can be configured independently of pretraining with
`downstream.regions`. Without this block, the existing random sampler remains
in use. Grid and multiscale layouts have the same geometry for every recording
and every epoch, including finetuning. Time coordinates are fractions of each
recording; absolute durations and observed contents still differ.

Two example configs provide equal **27-token** budgets:

| Config | Volumetric windows | Projection tokens |
| --- | --- | --- |
| `configs/thu_downstream_grid.yaml` | `3 x 3 x 1` grid: 9 windows | 9 x 3 = 27 |
| `configs/thu_downstream_multiscale.yaml` | Full volume + `2 x 2 x 2`: 9 windows | 9 x 3 = 27 |

Each window is projected in XY, XT, and YT. This pair compares a spatial grid
with a hierarchy that also splits time; it is not a controlled isolation of
scale alone. Customize grid dimensions/levels for other comparisons.

```yaml
downstream:
  regions:
    mode: grid
    grid: [3, 3, 1]       # x, y, t subdivisions
    plane_mode: all
    num_regions: 27      # optional assertion of the resulting token count
```

For multiscale, replace `grid` with `levels`:

```yaml
downstream:
  regions:
    mode: multiscale
    levels: [[1, 1, 1], [2, 2, 2]]
    plane_mode: all
    num_regions: 27
```

Every level tiles the entire volume. Spatial boundaries use integer pixel
partitions; temporal boundaries divide normalized time uniformly. Different
levels overlap intentionally. The token count is derived from the layout, never
achieved by randomly selecting or duplicating windows. An explicit `num_regions`
must match it. With `plane_mode: all`, count is the sum of grid-cell counts times
`len(data.plane_types)`. With `cycle`, each cell gets one projection, cycling
through the plane list within each level, and the count is just the cell total.
All layout tokens are valid, including windows with no events.

A fixed-count random baseline is also supported:

```yaml
downstream:
  regions: {mode: random, num_regions: 27}
```

This retains fixed per-recording draws for probing/evaluation and per-epoch
resampling for finetuning. Random spatial/time scales still come from `data`.
Grid/multiscale modes ignore those random scale/count settings. None of these
downstream settings changes SSL sampling. If `data.max_events` is nonzero,
finetuning can still resample events each epoch even with fixed window geometry.

```powershell
python -m experiments.region_worldmodel.downstream --config experiments/region_worldmodel/configs/thu_downstream_grid.yaml --checkpoint outputs/region_worldmodel/best.pt --mode linear_probe
python -m experiments.region_worldmodel.downstream --config experiments/region_worldmodel/configs/thu_downstream_multiscale.yaml --checkpoint outputs/region_worldmodel/best.pt --mode linear_probe
```

Use `--mode finetune` for either layout. Configs inherit the current `thu.yaml`
training/data settings; no new pretraining is required. Explicit layouts write
to directories such as `linear_probe_grid_27` and `finetune_multiscale_27` under
`train.output_dir`; use `--output-dir` when comparing multiple layouts of the
same type and count. Results record the region specification and maximum token
count. These configs do not change the current default sampler.

Existing region-world-model checkpoints work with larger or smaller downstream
token counts because the encoder uses relative attention and disables absolute
position embeddings. The unused absolute embedding tensors retain their original
checkpoint shapes. Downstream checkpoints therefore include `backbone_config`
alongside the run `config`: construct `RegionWorldModel(backbone_config)`, wrap
it in `Classifier`, and load the classifier state. Dataset construction uses the
run `config`. Plane IDs, patch dimensions and normalization must still agree.
This compatibility does not guarantee equal accuracy when the region distribution
differs from pretraining. All tokens retain equal weight in mean pooling, so the
multiscale example gives the eight fine windows more combined weight than the
single global window.

## How to decide whether it works

Validation reports prediction MSE, latent standard deviation, per-action MSE,
and these matched-target errors (lower is better):

| Metric | Predictor input / baseline | Purpose |
| --- | --- | --- |
| `persistence` | Copy source tokens | Does prediction beat doing nothing? |
| `zero_action` | Replace action with identity | Does the trained predictor use actions? |
| `wrong_action` | Replace with next catalog entry | Is the action information specific? |
| `blank_source` | Zero source patches, retain geometry | Does event content matter? |

The ablations use the trained model; they are not independently trained
action-free or metadata-only baselines. Latent standard deviation is measured
over validation target views, which include repeated recordings, and can also
reflect geometry. These diagnostics alone do not rule out shortcuts.

Compare frozen probe and finetune accuracy against the existing region MAE,
random initialization, an unmasked two-term run, and an independently trained
action-free baseline with equal compute and the same data split. Evaluate held-out
angles/axes to distinguish interpolation from memorizing catalog entries.
Only then expand to larger moves, uncertainty-aware predictors, action sequences,
or learned action selection. This implementation does not yet learn a sensing
policy or model unseen event content probabilistically.

## Verification

```powershell
python -m pytest tests/test_region_worldmodel.py -q
```

Tests cover canonical/custom-axis geometry, inverse rotations away from edges,
scales/bounds, configuration rejection, timestamp precision, deterministic
resampling, identity pairs, actual target patch changes, target gradients,
padding isolation, frozen probe versus finetuning, and end-to-end checkpoint
training/loading/resume. Bounded local THU runs also completed for all three
training modes. These checks establish execution correctness, not accuracy:
after two smoke training batches, persistence still outperformed prediction and
the tiny eight-example downstream test had zero accuracy. Full training and
comparative representation quality remain to be measured.
