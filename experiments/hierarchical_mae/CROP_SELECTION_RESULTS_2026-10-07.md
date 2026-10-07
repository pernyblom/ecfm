# Crop-augmented set selection — 2026-10-07

Implemented and ran crop augmentation for the learned bounded-swap selector.
**This controlled experiment did not establish an accuracy improvement.** The
one-swap full-recording result worsened; two- and four-swap results changed only
slightly. On unseen test crops, augmented policies reduced cross-entropy modestly
but did not improve accuracy.

## Controlled setup

- Same pretrained encoder, original activity-54 classifier, candidate pool, hard
  swap limits 1/2/4, optimizer, model seed, and 500 policy epochs as before.
- All 1,518 training recordings, each with four random crops plus its full view:
  **7,590 cached views**. Temporal fractions are 0.3–1.0; width and height
  fractions are independently 0.6–1.0, with random spatial and temporal offsets.
- One view per recording per epoch, cycling through all five with seeded
  per-recording phase offsets. Each view is used 100 times. This matches the
  prior experiment's **759,000 training examples and 6,000 optimizer updates per
  policy**, rather than increasing training compute fivefold.
- Crop-specific activity ranking, candidate pool, coarse inputs, and all 210
  set losses are generated together from that crop. Full-view losses are never
  used as cropped targets. Only compact inputs/targets are cached.
- Full-view validation (268 recordings) chooses policy epoch and threshold.
  Full-view train features are then used for the same three-seed fresh-probe
  comparison; the classifier probes themselves receive no crop augmentation.
- All policies and probe checkpoints are fixed before test evaluation. The
  normal test set remains 544 full recordings. A separate diagnostic evaluates
  two identical new crops of each test recording for old and new policies.

The random crops are a **finite reproducible bank**, not freshly sampled at
every training epoch. More views can be requested with `--crop-views`.

## Full-recording test: fresh linear probes

Each entry below averages classifier seeds 7, 17, and 27. There is only one
training seed per selection policy.

| Selection | Fixed-view policy training | Crop-augmented policy training | Change |
|---|---:|---:|---:|
| Activity-54 | 43.93% | 43.93% | 0.00 pp |
| At most 1 swap | 44.73% | 43.75% | −0.98 pp |
| At most 2 swaps | 44.24% | 44.36% | +0.12 pp |
| At most 4 swaps | 43.50% | 43.57% | +0.06 pp |

Augmented-policy results by classifier seed:

| Selection | Seed 7 | Seed 17 | Seed 27 |
|---|---:|---:|---:|
| Activity | 44.12% | 43.93% | 43.75% |
| Limit 1 | 43.93% | 44.12% | 43.20% |
| Limit 2 | 43.75% | 45.59% | 43.75% |
| Limit 4 | 43.38% | 44.12% | 43.20% |

The small positive mean changes at limits 2 and 4 each come from one or two
additional correct predictions across the three runs in total. They do not
establish a reliable benefit. These runs reuse the same test recordings, and
this dataset has already been used in preceding exploratory experiments.

## Full-recording test: original frozen activity classifier

| Augmented policy | Chosen epoch (1-based) | Threshold | Validation loss | Test loss | Test accuracy | Mean test swaps |
|---|---:|---:|---:|---:|---:|---:|
| Activity | — | — | 1.90043 | 2.17844 | 44.12% | 0 |
| Limit 1 | 152 | 0.025 | 1.85529 | 2.19678 | 44.12% | 0.546 |
| Limit 2 | 26 | 0 | 1.87175 | 2.21303 | 43.57% | 0.792 |
| Limit 4 | 110 | 0.01 | 1.84332 | 2.17791 | 43.20% | 1.551 |

Compared with fixed-view training, the best validation loss improved for limits
1 and 4, but these improvements did not transfer consistently to test accuracy.
[Validation curves](../../outputs/hierarchical_mae/set_selection54_crops4/validation_curves.png)
compare the matched-update runs.

Correlation between predicted and actual gains across all validation proposals
increased from 0.058 to 0.097 for limit 1, −0.008 to 0.018 for limit 2, and −0.007
to 0.056 for limit 4. These are still weak correlations, and improved overall
gain regression does not necessarily improve the final classification decision.

## Generalization to unseen crops

The diagnostic uses crop view IDs 10000 and 10001 on all 544 test recordings:
1,088 crop observations. Training used view IDs 0–3 on disjoint recordings.
All methods see exactly the same evaluation crops and use the same original
frozen activity classifier. No test crops participate in checkpoint selection.

| Selection | Test crop accuracy | Test crop loss | Mean swaps |
|---|---:|---:|---:|
| Activity-54 | 21.60% | 5.78639 | 0 |
| Fixed-view policy, limit 1 | 22.15% | 5.77662 | 0.474 |
| Augmented policy, limit 1 | 21.32% | 5.77248 | 0.675 |
| Fixed-view policy, limit 2 | 21.88% | 5.79415 | 0.657 |
| Augmented policy, limit 2 | 21.42% | 5.75731 | 1.251 |
| Fixed-view policy, limit 4 | 21.69% | 5.78922 | 0.423 |
| Augmented policy, limit 4 | 21.23% | 5.74741 | 1.965 |

Augmented policies make more replacements on new crops and improve the loss
slightly, but not accuracy. Multiple crops of a recording are correlated and
must not be treated as independent test recordings. Policies were selected
using **full-view** validation, so this is transfer testing of those selected
policies, not selection for cropped-view accuracy.

## What this suggests

The unchanged teacher classifier scores only about **22% on the random training
crops**, compared with 81.88% on full training recordings. Cropped test accuracy
is similarly low. This is a substantial distribution change for a classifier
that was trained solely on full-recording pooled features. Selection changes
only a few tokens and does not retrain that classifier.

Therefore this result is evidence against this particular combination of wide
crop ranges, frozen full-view teacher, small crop bank, and full-view validation
selection. It does **not** show that crop augmentation is generally unhelpful.

The next controlled step would be to train the teacher classifier on both full
and cropped features, verify its crop accuracy, then generate selector targets
with that classifier. Out-of-fold teachers would additionally keep a recording
out of the teacher's fitting data. A mixed full/crop validation criterion could
then select for robustness explicitly. Neither of those changes was implemented
in this experiment; the classifier was deliberately held fixed to isolate
augmentation of the selector.

## Implementation and artifacts

See [the crop augmentation commands](SET_SELECTION.md#spatial-and-temporal-crop-augmentation).
New code is in `crop_selection.py`, with training integration in
`set_experiment.py`. The default `--crop-views 0` preserves the fixed-view path.

Outputs are under `outputs/hierarchical_mae/set_selection54_crops4`:

- `results.json`: full-view fixed-head and fresh-probe results.
- `unseen_crops.json` and `.pt`: matched-crop metrics and per-view predictions.
- `policy_swaps*.pt`, `probe_swaps*_<seed>.pt`, `last_swaps*.pt`: final and
  continuation checkpoints, loadable with the existing `load_policy` helper.
- `validation_selection.json`, `fit_diagnostics.json`, epoch histories, and
  `validation_curves.png`.

Crop-target banks are in `outputs/hierarchical_mae_selector_crops`. Four banks
occupy approximately 34.5 MB total and took about 154 seconds to generate on this
machine. They are keyed by data/checkpoint/crop/source provenance. Rendered
random patches are not retained. Logs are `outputs/set_selection54_crops4.log`
and `outputs/set_selection54_crop_transfer.log`.

**101 tests passed**, including crop reproducibility, unchanged evaluation
views, exact view coverage and matched epoch size, fixed-view sampler
compatibility, and independently verified agreement between a crop's policy
inputs and its classification-loss targets. Full-view live inference also
matched cached oracle lookups for the actual validation and test datasets.
