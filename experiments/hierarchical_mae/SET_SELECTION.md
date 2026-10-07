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
