# Learned selection of bounded replacement sets

This is a **deployable, coarse-conditioned selection policy**. At inference it
requires no labels or oracle evaluation. It starts from activity selection and
chooses an entire replacement set, or no change, with a hard maximum swap count.

## Train and compare

First generate the fixed-classifier loss targets with `swap_oracle` as described
in [the oracle instructions](SWAP_SELECTION.md#multi-swap-oracle-diagnostics).
Then run:

```powershell
python -m experiments.hierarchical_mae.set_experiment `
  --baseline outputs/hierarchical_mae/linear_probe_activity54/best.pt `
  --oracle outputs/hierarchical_mae/swap54_multi_oracle/results.json `
  --output-dir outputs/hierarchical_mae/set_selection54 `
  --limits 1 2 4 --epochs 500
```

The default experiment trains three separate policies using all training
recordings. Each policy is selected by validation loss over 500 epochs and a
fixed threshold grid. The zero-initialized no-change policy is an explicit
fallback. Each frozen policy and the activity baseline then receive independent
fresh linear probes for seeds 7, 17, and 27, using the baseline checkpoint's
500-epoch probe configuration. Policies, thresholds, and probe checkpoints are
fixed before test inputs are extracted. Test oracle losses are used only for
final reporting and to verify the actual inference path.

Targets are `activity_classifier_loss - replacement_set_classifier_loss` from
the training split. The pretrained encoder and activity classifier stay frozen.
This version uses the original classifier's **in-sample training targets**;
out-of-fold target generation is not implemented here.

## Policy inputs and hard limits

The input consists of:

- Concatenated patch projections for level-zero/root tokens, before transformer
  encoding.
- Candidate activity, relative activity, rank, nine geometry/time values,
  hierarchy level, and representation.

No fine-patch pixel content, alternative transformer outputs, labels, or oracle
losses enter policy inputs. Event counts are still needed across the hierarchy
to construct activity selection and its candidate pool. The existing dataset
pipeline currently renders all patches; the policy's limited patch access does
not yet avoid that data-generation cost.

The candidate pool is the four lowest-ranked selected tokens and six
highest-ranked excluded tokens. Complete allowed sets are enumerated: 25
including no-change for one swap, 115 for two, and 210 for four. Pool positions
use canonical activity tie handling. A shared candidate encoder projects each
descriptor to 16 dimensions. Summed removal and insertion embeddings, their
difference, the number of replacements, and a 32-dimensional coarse context
feed a shared MLP with hidden dimension 64 and dropout 0.1. This permits
interactions within each replacement set without running the event transformer.

The selected set can **never exceed the configured swap limit** because every
choice in the policy's output space already meets it. All choices still contain
exactly 54 tokens. No-change has known predicted gain zero. A replacement is
accepted only when its predicted gain exceeds the validation-selected threshold.
The event transformer then runs **once** on the chosen 54 tokens.

## Inference with saved checkpoints

```python
from experiments.hierarchical_mae.set_selector import load_policy
from experiments.hierarchical_mae.loading import to_device

model = load_policy(
    'outputs/hierarchical_mae/set_selection54/policy_swaps2.pt',
    probe='outputs/hierarchical_mae/set_selection54/probe_swaps2_7.pt',
    device='cuda',
)
logits = model(to_device(batch['source'], 'cuda'))
```

Omit `probe` to use the original activity classifier. `baseline=...` can override
the saved baseline path when moving machines; the loader verifies its digest.
It also verifies that a supplied probe belongs to the selected policy. The
loader restores evaluation mode and freezes all parameters. The resulting
object is for inference, not joint finetuning.

To inspect selected sets, call
`set_selector.selected_features(model.classifier.backbone, view, model.policy,
model.threshold, model.budget)`. It returns pooled features and a state index per
recording; `model.policy.states[index]` contains removed and inserted candidate
pool positions. `model.policy.depths[index]` is the actual number of swaps.

## Caches and continuation

The experiment validates checkpoint, recording order, preprocessing, and source
implementation provenance against the oracle target bank. Cached root inputs
and candidate descriptors are small, and training never re-runs the encoder.
Selected pooled features for probing are cached separately and fingerprinted
by policy, source, and dataset provenance.

`last_swaps1.pt`, `last_swaps2.pt`, and `last_swaps4.pt` contain optimizer state,
epoch, and best-policy snapshots. Re-running the same command after an
interruption automatically continues those policy runs, with deterministic
epoch seeds and checked training settings. Probe training restarts from its
deterministic initialization; feature extraction reuses compatible caches.
Completed experiment directories are protected from overwriting; use a new
directory for another completed experiment.

Final `policy_swaps*.pt` files contain scorer architecture, weights,
normalization, threshold, baseline path/digest, and selected epoch.
`validation_selection.json` records choices before test evaluation, policy
histories record every epoch, and `results.json` contains fixed-head and
fresh-probe results. Standalone set-selector checkpoints are distinct from the
older `downstream --mode selector_probe` checkpoint format.

See [the completed 500-epoch local experiment](SET_SELECTION_RESULTS_2026-10-07.md)
for per-seed results, behavior under each swap limit, and training curves.

## Spatial and temporal crop augmentation

```powershell
python -m experiments.hierarchical_mae.set_experiment `
  --baseline outputs/hierarchical_mae/linear_probe_activity54/best.pt `
  --oracle outputs/hierarchical_mae/swap54_multi_oracle/results.json `
  --output-dir outputs/hierarchical_mae/set_selection54_crops4 `
  --crop-views 4 --epochs 500
```

`--crop-views` defaults to zero, preserving fixed-view training. With four,
each training recording has its original full view plus four reproducible
random crops. The ranges come from the baseline checkpoint's training data
configuration: THU uses temporal fractions `[0.3, 1.0]` and independent width
and height fractions `[0.6, 1.0]`, with random offsets. Crops use the existing
`HierarchyDataset` sampler and seeded view IDs; they are not resized full-view
targets. Activity ranking, the replacement pool, policy inputs, and all loss
targets are recomputed from the **same crop**.

Training uses one view per recording per epoch, with seeded per-recording phase
offsets and a cycle through all views. Thus 500 epochs have the same recording
count, batch count, and optimizer updates as fixed-view training, rather than
five times more updates. With five total views, every recording is seen in its
full view 100 times and in each cropped view 100 times. Normalization statistics
use all training views. There are no crops of validation/test recordings in the
training bank.

The original full-view oracle supplies validation/test targets and the full
training view. Crop targets are generated once with the same frozen activity
classifier and cached under `--crop-cache-dir` (default
`outputs/hierarchical_mae_selector_crops`). Cache keys include source entries,
checkpoint digest, crop configuration and view ID, and implementation hashes.
Only compact policy inputs and targets are stored; the rendered random patches
are discarded. Completed per-view caches can be reused after interruption or
in another experiment directory. This is a finite crop bank, not newly sampled
crops at every epoch; increase `--crop-views` for more diversity.

The fresh linear probes continue to train on **full recording features**. Their
configuration, and the fixed full-view validation/test protocol, are unchanged.
This isolates augmentation of the selector itself.

After policies are fixed, compare the original and augmented selectors on
identical *unseen* crops:

```powershell
python -m experiments.hierarchical_mae.crop_selection `
  --fixed-dir outputs/hierarchical_mae/set_selection54 `
  --augmented-dir outputs/hierarchical_mae/set_selection54_crops4 `
  --output outputs/hierarchical_mae/set_selection54_crops4/unseen_crops.json
```

This defaults to two crops of every test recording, using view IDs 10000 and
10001, and compares activity plus both policies at each swap limit. All use the
same original frozen classifier so the effect of changing selection is isolated.
It reports cross-entropy, accuracy, and mean swaps, and saves per-view labels and
predictions. These views are evaluation-only and never determine policy epochs
or thresholds. Multiple crops from one recording are correlated observations,
not additional independent test recordings. Use `--split validation`, `--views`,
and `--first-view` to define other explicit diagnostic sets.

See [the completed crop-augmentation comparison](CROP_SELECTION_RESULTS_2026-10-07.md)
for full-view and unseen-crop results and the limitations of the frozen teacher.

### Milder crops and YAML configuration

The preset [thu_selector54_crop80.yaml](configs/thu_selector54_crop80.yaml)
keeps 80–100% of the temporal length and independently 80–100% of the width and
height. Offsets are random. Its main-pipeline settings are:

```yaml
extends: thu_selector54.yaml
data:
  crop_fraction: [0.8, 1.0]
  spatial_crop_fraction: [0.8, 1.0]
  eval_fraction: 1.0
  patch_cache:
    train_views: 0
downstream:
  epochs: 500
  learned_selector:
    budget: 54
    training_crops: true
```

To train through the main downstream pipeline:

```powershell
python -m experiments.hierarchical_mae.downstream `
  --config experiments/hierarchical_mae/configs/thu_selector54_crop80.yaml `
  --checkpoint outputs/hierarchical_mae/best.pt `
  --mode selector_train `
  --output-dir outputs/hierarchical_mae/selector_train54_crop80
```

Use a **pretrained MAE** checkpoint for this command, not an activity-probe or
bounded-swap checkpoint. `selector_train` uses the original Gumbel selector and
trains the classification head alongside it, with a frozen encoder. Its budget
is 54 selected tokens; it does not enforce a limit on swaps from activity.
With `training_crops: true` and `train_views: 0`, each training epoch gets fresh
spatial/temporal crops. Validation and test use the full view. Positive
`train_views` instead cycles through a reusable patch-cache bank.

To fit a fresh full-view probe afterward, use the existing probe preset:

```powershell
python -m experiments.hierarchical_mae.downstream `
  --config experiments/hierarchical_mae/configs/thu_selector_probe54.yaml `
  --checkpoint outputs/hierarchical_mae/selector_train54_crop80/best.pt `
  --mode selector_probe `
  --output-dir outputs/hierarchical_mae/selector_probe54_crop80
```

`selector_probe` is frozen and uses fixed evaluation views regardless of the
training-crop flag. For continuation of the crop-trained run, use the same
training YAML with `--resume .../selector_train54_crop80/last.pt`. Changing crop
settings when resuming is rejected; start a new run to compare a different range.

The separate **bounded-swap** experiment can import the same YAML crop ranges:

```powershell
python -m experiments.hierarchical_mae.set_experiment `
  --baseline outputs/hierarchical_mae/linear_probe_activity54/best.pt `
  --oracle outputs/hierarchical_mae/swap54_multi_oracle/results.json `
  --output-dir outputs/hierarchical_mae/set_selection54_crops80 `
  --crop-views 4 `
  --crop-config experiments/hierarchical_mae/configs/thu_selector54_crop80.yaml
```

Here `--crop-config` reads **only** `data.crop_fraction` and
`data.spatial_crop_fraction`; the bounded experiment's limits, epochs, and other
settings come from its CLI/checkpoint. It still trains from the frozen activity
classifier's cached targets and uses the finite four-crop-plus-full bank.
This distinction matters: the main pipeline also adapts its classifier to crops.

Alternatively specify `--temporal-crop 0.8 1.0 --spatial-crop 0.8 1.0` without
`--crop-config`. Overrides require `--crop-views > 0`; they affect only training
crop generation. Full-view target provenance, probe views, and validation/test
stay unchanged. Target-cache keys include the effective crop ranges.

The `crop_selection` unseen-crop evaluator now defaults to the augmented
policies' **saved training ranges**, so comparing a crop80 policy evaluates both
sets of policies on the same 80–100% crops. Explicit `--temporal-crop` and
`--spatial-crop` overrides can select a different diagnostic distribution.

See [the completed 80–100% rerun](CROP80_SELECTION_RESULTS_2026-10-07.md) for
full-view and matched mild-crop results.
