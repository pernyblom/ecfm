# Learned bounded set selection — 2026-10-07

Implemented and ran three real, coarse-conditioned policies with hard limits of
1, 2, and 4 swaps. All trained for 500 epochs on all 1,518 training recordings.
The encoder stayed frozen. Policies were selected using 268 validation
recordings; all policy/threshold and fresh-probe checkpoint choices were fixed
before evaluating the 544 test recordings.

**The one-swap policy improved mean fresh-probe test accuracy from 43.93% to
44.73%. Increasing the swap limit did not improve on that result.** This is a
modest exploratory gain, not a recovery of the much larger label-assisted oracle
results.

## Fresh linear probes

Each policy, and activity-54, received a fresh 500-epoch probe for three matched
classifier seeds. Probe hyperparameters and validation-loss checkpoint selection
match the existing downstream experiment. All models use exactly 54 tokens.

| Selection | Seed 7 test | Seed 17 test | Seed 27 test | Mean test | Difference vs activity |
|---|---:|---:|---:|---:|---:|
| Activity-54 | 44.12% | 43.93% | 43.75% | **43.93%** | — |
| Learned, at most 1 swap | 45.04% | 45.77% | 43.38% | **44.73%** | +0.80 pp |
| Learned, at most 2 swaps | 43.38% | 45.59% | 43.75% | **44.24%** | +0.31 pp |
| Learned, at most 4 swaps | 43.38% | 43.93% | 43.20% | **43.50%** | −0.43 pp |

The one-swap policy gains 5 and 10 correct test predictions in seeds 7 and 17,
but loses 2 in seed 27. There is only one training seed for each selector, and
all probes reuse the same test recordings; these are not independent dataset
replications. The test set has also been used in preceding exploratory studies.
Activity remains a strong default, and this gain needs replication before a
general claim of improvement.

## Fixed original classifier and selection behavior

| Policy | Chosen policy epoch (1-based) | Threshold | Validation loss | Validation accuracy | Test loss | Test accuracy | Test swap rate | Mean test swaps |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Activity | — | — | 1.90043 | 45.52% | 2.17844 | 44.12% | 0% | 0 |
| Limit 1 | 11 | 0 | 1.86565 | 45.52% | 2.16734 | 44.12% | 58.46% | 0.585 |
| Limit 2 | 10 | 0 | 1.86624 | 47.01% | 2.15070 | 44.49% | 58.64% | 0.710 |
| Limit 4 | 479 | 0.2 | 1.88226 | 46.27% | 2.20573 | 44.12% | 26.65% | 0.798 |

A limit is an upper bound, not a requirement to perform that many swaps. The
four-swap policy's selected high threshold makes it leave most recordings
unchanged. Its fixed-head test loss is worse than baseline despite slightly
better validation loss. Refitting the classifier changes the comparison, which
is why both evaluations are reported.

## What was trained

Each policy has **24,081 trainable parameters**. Inputs are the three root patch
projections before the transformer, and the activity, rank, geometry, hierarchy,
and representation descriptors of the candidate tokens. Fine patch content,
labels, and alternative transformer outputs are not policy inputs.

The replacement pool stays fixed at four removable activity tokens and six
excluded insertion candidates. The policy scores 25, 115, or 210 complete sets,
depending on its limit. It uses learned candidate embeddings and pooled removal
and insertion summaries alongside coarse context, allowing interactions among
the replacements. No-change is always available and has gain zero. Only a set
whose predicted gain exceeds the selected threshold can replace the baseline.

Training targets are actual cross-entropy improvements measured by the original
activity classifier. These come from the cached exhaustive oracle **training**
bank. The classifier was trained on those same training recordings; this run
does not use out-of-fold supervision. MSE regression uses AdamW with learning
rate 0.001, weight decay 0.01, batch size 128, hidden dimension 64, dropout 0.1,
and gradient clipping at 1. Normalization uses training inputs only.

At inference, the predictor needs no labels and runs the event transformer
**once on the final 54 tokens**. Hard limits follow from the enumerated output
space rather than a penalty that could be violated. Existing data loading still
renders all candidate patches; patch-generation savings are a separate task.

## Did longer training help?

Training MSE continued falling, but the best one- and two-swap policies were
found within the first eleven epochs. The four-swap model improved its best
validation loss slightly late in training, but this did not translate into a
better test result.

Correlation between predicted and actual gains over all replacement sets at
the selected checkpoints:

| Limit | Training correlation | Validation correlation |
|---|---:|---:|
| 1 | 0.566 | 0.058 |
| 2 | 0.578 | −0.008 |
| 4 | 0.921 | −0.007 |

These correlations describe overall gain regression, which differs from the
quality of the final thresholded choice. Nevertheless, the large gap supports
overfitting/generalization as the main concern. Longer training alone did not
solve it. [Training curves](../../outputs/hierarchical_mae/set_selection54/training_curves.png)
show falling training MSE alongside the selected-set validation loss.

## Artifacts, usage, and verification

All outputs are in `outputs/hierarchical_mae/set_selection54`:

- `policy_swaps1.pt`, `policy_swaps2.pt`, `policy_swaps4.pt`: deployable policies.
- `probe_swaps<limit>_<seed>.pt`: matching fresh classifier heads.
- `last_swaps*.pt`: optimizer state and best snapshot for interrupted training.
- `results.json`, `validation_selection.json`, `fit_diagnostics.json`, and
  per-epoch histories: results and analysis.
- Cached inputs and pooled features; `training_curves.png`.

The encoder is restored from the original activity probe checkpoint, whose
SHA-256 is `0a73fb3c395a235e16608c92d27a01508fde16eb84360eb1d88b38d35e2d8a70`.
It is referenced and digest-checked by the loader, rather than copied into every
policy file. The pretraining digest is unchanged from the earlier experiments.

For a fixed-seed example using the strongest mean variant:

```python
from experiments.hierarchical_mae.set_selector import load_policy
from experiments.hierarchical_mae.loading import to_device

model = load_policy(
    'outputs/hierarchical_mae/set_selection54/policy_swaps1.pt',
    probe='outputs/hierarchical_mae/set_selection54/probe_swaps1_7.pt',
    device='cuda',
)
logits = model(to_device(batch['source'], 'cuda'))
```

See [training, continuation, and inference instructions](SET_SELECTION.md).
These standalone checkpoints are separate from the old Gumbel selector modes.

**98 tests passed.** Tests cover hard swap limits, no-change equivalence,
nonzero training gradients, invariance to fine-patch content in policy inputs,
single-transformer-pass inference, and checkpoint/probe identity checking. On
real data, selected-feature inference was checked against cached oracle losses
for every validation and test recording. Reloaded policies with fresh heads also
matched cached validation features, and all three inference paths respected the
54-token budget with one encoder call.

The next useful training change is out-of-fold target generation, which would
measure swap benefits on recordings unseen by the teacher classifier. That is
not implemented here. It would test whether target construction contributes to
the generalization gap before increasing policy capacity or the number of swaps.
