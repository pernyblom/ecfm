# THU activity-54 residual selection experiments — 2026-10-07

Implemented and ran on the local GPU. All experiments use the encoder and
classifier from `outputs/hierarchical_mae/linear_probe_activity54/best.pt`.
That checkpoint embeds the encoder, so a changing pretraining `best.pt` cannot
change these experiments. Baseline checkpoint SHA-256:
`0a73fb3c395a235e16608c92d27a01508fde16eb84360eb1d88b38d35e2d8a70`.
Pretraining digest:
`11959e0628741b5ac77cd22c68fbe5f323046f83c0deab1804c873a25fb4d49d`.

**Outcome: conservative selection avoids the previous learned selector's large
drop, but no convincing improvement over activity was established.**

## Existing runs

| Run | Validation accuracy | Test accuracy |
|---|---:|---:|
| All 411 tokens | 64.18% | 43.57% |
| Activity 54 | 45.52% | 44.12% |
| Learned selection 54, fresh probe | 38.43% | 31.25% |

The all-token, activity, and selector-training checkpoints report the same
pretraining digest. The all-token run's much larger validation/test gap is worth
remembering when interpreting small validation improvements. None of these
experiments changes the existing split: 1,518 training, 268 validation, and 544
test recordings, with fixed evaluation crops and the original 411-token layout.

At its selected checkpoint the old learned selector retained only 38.64% of the
activity set on validation. Its average level counts were 3 root, 3.46 middle,
and 47.54 fine tokens. This describes the drift; it does not prove its cause.

Activity-27 and coarse-27 overlap by **74.96%** on validation and are identical
for **14.18%** of recordings. Training figures are 74.72% and 12.19%. Activity
thus often replaces some low-activity middle-level regions with finer regions
from more active parts of a recording.

## One-swap design

Start with the exact activity-54 set. Consider removing any of its four
lowest-ranked members and inserting any of the six highest-ranked excluded
tokens: 24 alternatives, each changing just one token. Include the unchanged
set as option zero. All alternatives still contain exactly 54 tokens.

For training examples, evaluate the actual decrease in cross-entropy under the
frozen baseline classifier and regress that improvement. No labels or
alternative transformer outputs enter policy inputs. Validation loss selects
the epoch and acceptance threshold; no-change is always a candidate. Test data
is accessed only after those choices and fresh-probe checkpoints are fixed.

Three policy variants were evaluated:

- **Coarse MLP:** root patch projections plus both candidates' activity,
  geometry, hierarchy, representation, and rank. Hidden dimension 64.
- **Local MLP:** adds both candidates' patch projections.
- **Ridge:** standardized linear regression; validation selects between the two
  input sets, five regularization strengths, and seven thresholds. It selected
  coarse inputs, regularization 0.001, threshold 0.025.

The MLPs trained for 100 epochs. Validation selected coarse epoch 61 and local
epoch 73, both with threshold 0.005. The source classifier and encoder stayed
frozen. Each frozen selection policy then received a **fresh** linear probe for
500 epochs with the original AdamW settings, selecting by validation loss.
Three matched classifier seeds were used; the selector itself has only one
training seed. All fresh probes use the same initialization and shuffle method
as the existing downstream implementation.

## Fresh-probe test results

| Selection | Seed 7 | Seed 17 | Seed 27 | Mean | Mean change vs activity |
|---|---:|---:|---:|---:|---:|
| Activity 54 | 44.12% | 43.93% | 43.75% | **43.93%** | — |
| Coarse MLP, up to one swap | 44.12% | 44.49% | 44.12% | **44.24%** | +0.31 pp |
| Local MLP, up to one swap | 42.83% | 42.65% | 43.38% | **42.95%** | −0.98 pp |
| Ridge, up to one swap | 44.30% | 44.49% | 42.65% | **43.81%** | −0.12 pp |

The seed-7 activity reproduction selected epoch 99 and matched the original
validation loss and test accuracy. The coarse MLP gain amounts to zero, three,
and two additional correct test predictions across the three seeds. These
seeds reuse the same test recordings and the same learned policy; they are not
three independent dataset replications. This is a small exploratory result,
not evidence of a reliable accuracy improvement.

## Keeping the original activity classifier

| Policy | Validation loss | Validation accuracy | Test loss | Test accuracy | Test swap rate |
|---|---:|---:|---:|---:|---:|
| Activity | 1.90043 | 45.52% | 2.17844 | 44.12% | 0% |
| Coarse MLP | 1.86010 | 47.76% | 2.17552 | 44.49% | 62.13% |
| Local MLP | 1.85559 | 45.15% | 2.17694 | 44.12% | 87.32% |
| Ridge | 1.88256 | 46.27% | 2.16106 | 44.49% | 46.14% |

The learned predictors capture some average loss improvement, but the larger
validation gains mostly fail to transfer to test. Fixed-head improvement also
does not guarantee improvement after refitting the classifier.

As an overfitting diagnostic, the correlation between predicted and actual swap
improvements across all proposals was approximately 0.746 on training versus
0.036 on validation for the coarse MLP, and 0.876 versus 0.046 for the local MLP.
Only training data contributes normalization and regression targets. Giving the
MLP richer local inputs did not help this experiment; that does not establish
that local patch content is intrinsically unhelpful.

## Do useful swaps exist?

A labelled diagnostic selects the lowest true cross-entropy among all 25
alternatives, using the original classifier:

| Split | Activity-oracle loss | Activity-oracle accuracy | Mean loss decrease |
|---|---:|---:|---:|
| Training | 0.55529 | 88.41% | 0.15037 |
| Validation | 1.59348 | 55.97% | 0.30694 |
| Test | 1.81178 | 52.21% | 0.36665 |

This oracle uses labels and is **not deployable**, not a promised improvement,
and not an accuracy-maximizing oracle. It demonstrates considerable room to
choose better single swaps within this small candidate set. The difficult part
is predicting their effect on unseen recordings.

## Interpretation and next experiment

Keep activity-54 as the default. The constrained coarse policy is a useful
research starting point because it stays near baseline, but the current gain
does not justify replacing the heuristic generally. Increasing the number of
swaps immediately would give this imperfect predictor more opportunities to
make mistakes.

The next experiment I would prioritize is **out-of-fold swap supervision**:
train several activity classifiers on disjoint training folds, then generate
each recording's swap targets with a classifier that did not train on that
recording. Keep the encoder frozen and leave validation/test outside those
folds. This tests whether a policy can learn improvements relevant to unseen
recordings rather than effects specific to a classifier's training examples.
It is a proposal, not implemented or evaluated here. A further untouched split
or remote replication would help assess any eventual small improvement.

## Artifacts and verification

- [Usage and inference](SWAP_SELECTION.md)
- `outputs/hierarchical_mae/swap54_boundary/results.json`
- `outputs/hierarchical_mae/swap54_ridge/results.json`
- Policies and all 12 fresh-probe heads are saved in those two directories.
- `validation_selection.json` records choices made before each experiment's
  test evaluation; policy histories and ridge trials are also saved.
- Logs: `outputs/swap54_boundary.log`, `outputs/swap54_ridge.log`.

91 hierarchical-MAE tests passed, including exact no-op equivalence, tie and
invalid-token handling, fixed budget, one-token difference, and coarse overlap.
All three saved policies were additionally checked on real validation views:
single-pass inference choices matched cached choices, with maximum feature
difference below 4e-7. At inference the transformer runs once on 54 tokens;
alternative transformer evaluations are only a training-data generation cost.
