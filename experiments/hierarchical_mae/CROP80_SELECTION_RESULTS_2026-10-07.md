# Mild 80–100% crop experiment — 2026-10-07

Reran the bounded-swap experiment with temporal, width, and height fractions
all in **[0.8, 1.0]**, with random offsets. Four cached crops plus a full view per
training recording, 500 policy epochs, swap limits 1/2/4, and three fresh-probe
seeds match the previous experiment. One view per recording per epoch keeps
optimizer updates identical across the fixed-view, broad-crop, and mild-crop
runs. Full-view validation/test and probe training are unchanged.

The 80% minimum applies independently to each spatial dimension, before pixel
rounding, rather than to retained image area. The encoder and teacher remain
the same frozen activity-54 checkpoint as in previous comparisons.

## Full-recording test accuracy

Means across fresh-probe seeds 7, 17, and 27:

| Swap limit | Fixed-view training | Broad crops (time 30–100%, width/height 60–100%) | Mild crops (all 80–100%) |
|---|---:|---:|---:|
| 1 | 44.73% | 43.75% | **44.12%** |
| 2 | 44.24% | 44.36% | **43.20%** |
| 4 | 43.50% | 43.57% | **43.20%** |

Activity-54 remains **43.93%**. Mild crops improve the one-swap result over broad
crops, but do not establish an overall improvement over activity or fixed-view
selector training. The small one-swap advantage over activity is only one
additional correct test prediction per classifier seed on average.

| Selection | Seed 7 | Seed 17 | Seed 27 |
|---|---:|---:|---:|
| Activity-54 | 44.12% | 43.93% | 43.75% |
| Limit 1 | 44.67% | 45.40% | 42.28% |
| Limit 2 | 43.38% | 43.75% | 42.46% |
| Limit 4 | 43.57% | 43.20% | 42.83% |

These are three classifier seeds with one selector seed and the same test
recordings. This remains an exploratory comparison, not independent dataset
replication or a statistically established improvement.

## Original frozen classifier

| Policy | Selected epoch (1-based) | Threshold | Validation loss | Full-view test loss | Full-view test accuracy | Mean test swaps |
|---|---:|---:|---:|---:|---:|---:|
| Activity | — | — | 1.90043 | 2.17844 | 44.12% | 0 |
| Limit 1 | 118 | 0.005 | 1.86178 | 2.19037 | 43.20% | 0.689 |
| Limit 2 | 75 | 0 | 1.83347 | 2.20212 | 43.20% | 1.246 |
| Limit 4 | 94 | 0 | 1.85065 | 2.18804 | 44.12% | 1.748 |

The two-swap model's favorable validation loss did not transfer to the test set.
The four-swap policy preserves the original classifier's full-view test accuracy,
although refitting a fresh head produces a lower mean accuracy.
[Validation curves](../../outputs/hierarchical_mae/set_selection54_crops80/validation_curves.png)
compare the training-view conditions.

## Unseen 80–100% test crops

All policies see the same two new crops per test recording (view IDs 10000 and
10001): 1,088 observations from 544 recordings. Every row uses the original
frozen classifier. Saved evaluation ranges were verified to be `[0.8, 1.0]` for
time and space. No test crops determine policy epochs or thresholds.

| Selection | Accuracy | Cross-entropy | Mean swaps |
|---|---:|---:|---:|
| Activity-54 | 39.43% | 2.55232 | 0 |
| Fixed-view policy, limit 1 | 39.61% | 2.54629 | 0.545 |
| Mild-crop policy, limit 1 | 38.79% | 2.55075 | 0.762 |
| Fixed-view policy, limit 2 | 39.71% | 2.55633 | 0.697 |
| Mild-crop policy, limit 2 | 38.97% | 2.56128 | 1.335 |
| Fixed-view policy, limit 4 | 39.80% | 2.55713 | 0.570 |
| Mild-crop policy, limit 4 | **40.17%** | **2.54346** | 1.942 |

Limit 4 gives a small positive crop-transfer result: eight additional correct
crop predictions over activity, and four over the fixed-view limit-4 policy.
Limits 1 and 2 do not improve. Crops of the same recording are correlated; these
are not 1,088 independent test recordings. Policies were selected on full-view
validation, not cropped-view validation.

The earlier broad-crop activity result was 21.60%, versus 39.43% here. That
comparison describes a change in crop difficulty: the evaluation crops differ,
so it is not a policy improvement caused by training. The teacher also scores
50.07% on mild training crops versus about 22% on broad training crops. Milder
crops substantially reduce the distribution shift, but do not eliminate it.

## YAML and pipeline usage

Created [thu_selector54_crop80.yaml](configs/thu_selector54_crop80.yaml). It sets
both crop ranges to `[0.8, 1.0]`, enables
`downstream.learned_selector.training_crops`, sets the token budget to 54, and
configures 500 epochs. Evaluation remains full-view.

For the user to run later:

```powershell
python -m experiments.hierarchical_mae.downstream `
  --config experiments/hierarchical_mae/configs/thu_selector54_crop80.yaml `
  --checkpoint outputs/hierarchical_mae/best.pt `
  --mode selector_train `
  --output-dir outputs/hierarchical_mae/selector_train54_crop80
```

This requires a **pretrained MAE checkpoint** and uses the original Gumbel
selector. Its classification head trains on fresh crops alongside the selector,
with a frozen encoder. It does not impose a swap limit. This main-pipeline run
has **not** been launched; a small synthetic pipeline test verified the setup.

The completed experiment above is the separate bounded-swap runner using
`--crop-views 4 --crop-config experiments/hierarchical_mae/configs/thu_selector54_crop80.yaml`.
That flag imports only the YAML crop ranges; other settings come from the
CLI/checkpoint. It retains the original frozen classifier for target generation.
The main pipeline therefore tests a different setup that also lets its
classifier adapt to cropping.

See [training, probing, and resume instructions](SET_SELECTION.md#milder-crops-and-yaml-configuration).
The standalone runner also accepts `--temporal-crop MIN MAX` and
`--spatial-crop MIN MAX` instead of the YAML flag. Training overrides leave
full-view oracle provenance and validation/test preprocessing unchanged.

## Artifacts and checks

Outputs: `outputs/hierarchical_mae/set_selection54_crops80`.
`results.json` holds the full-view results; `unseen_crops.json` and `.pt` hold
matched mild-crop metrics and predictions. Policies, fresh heads, continuation
checkpoints, histories, and `validation_curves.png` are saved there too.
Logs are `outputs/set_selection54_crops80.log` and
`outputs/set_selection54_crop80_transfer.log`.

**103 tests passed**, including the new YAML, range validation and isolation,
and a main-pipeline smoke test proving that a pretrained checkpoint accepts
training-only crop changes while evaluation datasets stay fixed. The actual
bounded run also checked live validation/test features against cached oracle
losses, and the diagnostic's saved evaluation ranges were explicitly verified.
