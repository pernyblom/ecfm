# Learned activity/information blends at 216 tokens — 2026-10-08

Completed both full experiments: 100 selector-training epochs followed by 500
epochs of fresh linear probing. **The global blend had better final test
accuracy: 45.04% versus 41.73% for the coarse-conditioned blend.** The coarse
variant had better validation performance, but that advantage did not carry
over to test in this run.

## Setup

- Local pretrained MAE: `outputs/hierarchical_mae/best.pt`, saved epoch index 849.
- Checkpoint SHA-256: `11959e0628741b5ac77cd22c68fbe5f323046f83c0deab1804c873a25fb4d49d`.
- 216 of 411 candidate tokens, using the `support` information metric.
- Both weights start at 0.9. The global model fits one scalar; the coarse model
  predicts one scalar per recording from root patch embeddings.
- Seed 7; identical initial classifier weights across variants.
- 1,518 training, 268 validation, and 544 test recordings. Full views throughout.
- Frozen pretrained MAE during selector training; both MAE and selector frozen
  during fresh probing. Only the fresh probe head trains in the second stage.
- All configured batches and recordings were used. No smoke-test limits.
- Checkpoints selected by validation loss, with test evaluated afterward.

The stage presets are documented in [LEARNED_BLEND_SELECTION.md](LEARNED_BLEND_SELECTION.md).
Epoch numbers below are 1-based; checkpoint JSON uses 0-based indices.

## Results

| Variant | Selector best epoch | Selector validation accuracy | Selector test accuracy | Probe best epoch | Probe validation accuracy | Probe test accuracy | Probe test loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Global blend | 100 | 44.78% | 44.12% | 121 | 57.09% | **45.04%** | **2.1196** |
| Coarse blend | 99 | 44.78% | 45.96% | 137 | **58.96%** | 41.73% | 2.2301 |

The fresh global probe got **245/544** test recordings correct, versus
**227/544** for the coarse probe: a difference of **18 recordings / 3.31
percentage points**. The coarse variant improved validation accuracy by 1.87
points, but also had higher test cross-entropy.

Fresh probing increased global test accuracy by 0.92 points relative to its
training-stage head. For the coarse variant it decreased test accuracy by 4.23
points. The probe validation-loss curves bottom out well before epoch 500;
the reported results use those validation-selected checkpoints, not the last
epoch's weights.

## Learned weights and selected tokens

The global model selected **alpha = 0.368091**: approximately 36.8% normalized
activity score and 63.2% normalized support score.

| Coarse predictor split | Mean activity weight | Standard deviation across recordings |
| --- | ---: | ---: |
| Train | 0.398283 | 0.464872 |
| Validation | 0.392894 | 0.432331 |
| Test | 0.405483 | 0.442819 |

The dynamic predictor varies substantially across recordings despite having a
mean near the global constant. Its extra flexibility did not improve final
probe test performance here.

| Variant | Train overlap with activity top-k | Validation overlap | Test overlap |
| --- | ---: | ---: | ---: |
| Global blend | 96.11% | 96.25% | 93.18% |
| Coarse blend | 95.12% | 94.78% | 94.88% |

Both still select mostly the same tokens as activity. These are token-overlap
diagnostics, not activity-only classifier performance measurements. No fresh
activity-only 216-token probe was run as part of this request, so the experiments
do not establish whether either blend beats that baseline. This is also one
seed on the local pretrained checkpoint, not the stronger remote checkpoint.

## Artifacts and verification

All artifacts are under
[`outputs/hierarchical_mae/blend216_full_20261008`](../../outputs/hierarchical_mae/blend216_full_20261008):

- `global_train/` and `coarse_train/`: fitted selectors and training heads.
- `global_probe/` and `coarse_probe/`: fresh probes with their frozen selectors.
- Each stage has `best.pt`, `last.pt`, `metrics.jsonl`, and `results.json`.
- [`summary.json`](../../outputs/hierarchical_mae/blend216_full_20261008/summary.json)
  aggregates results and diagnostics; `run_status.json` records commands and timings.
- [`learning_curves.png`](../../outputs/hierarchical_mae/blend216_full_20261008/learning_curves.png)
  plots selector loss, activity weights, selection overlap, and probe performance.

Runtime was approximately 26.4 minutes on the RTX 4070 SUPER: 12.8 minutes for
global training, 9.4 for coarse training, and 2.1 for each probe, including
feature extraction. The global run incurred the initial data/cache work.

Post-run checks verified every expected epoch and full split size, selection of
the minimum-validation-loss checkpoints, unchanged pretrained encoder weights
during selector fitting, and unchanged encoder/selector weights during probing.
