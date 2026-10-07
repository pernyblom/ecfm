# THU activity-54 multi-swap oracle — 2026-10-07

Implemented and evaluated greedy, beam-width-4, and exhaustive label-assisted
search on the full training, validation, and test splits. **Two swaps reach
55.51% test accuracy and four reach 56.07%, compared with 52.21% for one swap.**
These are oracle diagnostics, not trained selector performance.

## Controlled comparison

The encoder, original activity-54 classifier, recording splits, fixed evaluation
crops, and activity tie handling are unchanged from the one-swap experiment.
No classifier was refitted. The baseline checkpoint is
`outputs/hierarchical_mae/linear_probe_activity54/best.pt`, SHA-256
`0a73fb3c395a235e16608c92d27a01508fde16eb84360eb1d88b38d35e2d8a70`.

The candidate pool remains fixed for each recording: remove from the four
lowest-ranked members of activity-54 and add from the six highest-ranked
excluded tokens. Up to four distinct removals and additions are allowed. The
transformer always receives exactly 54 tokens, with replacements evaluated
together. This is not unrestricted search over all 411 tokens.

There are 1, 24, 90, 80, and 15 distinct sets at exactly zero through four swaps,
respectively: **210 sets total**. All sets were evaluated once and cached, making
it possible to compare approximate searches with the exact minimum-loss result.

Greedy stops if its best next replacement does not reduce loss. Beam search
keeps four states at each exact depth and can explore temporarily worse states.
Each reported result retains the best state seen at any earlier depth, including
the unchanged baseline. Selection uses the true label to minimize cross-entropy.

## Accuracy

All three search methods obtained the same aggregate accuracy at each allowance
on these splits. This does not mean that they always chose the same token sets.

| Maximum swaps | Training (1,518) | Validation (268) | Test (544) | Test correct |
|---|---:|---:|---:|---:|
| 0 — activity | 81.88% | 45.52% | 44.12% | 240 |
| 1 | 88.41% | 55.97% | 52.21% | 284 |
| 2 | 89.66% | 58.96% | 55.51% | 302 |
| 3 | 90.38% | 60.45% | 56.25% | 306 |
| 4 | 90.38% | 60.45% | 56.07% | 305 |

The fourth-swap allowance lowers loss further but loses one correct test
prediction. This is expected to be possible: improving the true class's
probability does not necessarily make it the most probable class. The oracle
optimizes cross-entropy, not accuracy. The three-swap row is reported as part of
the pre-specified search curve, not used to select a deployable model.

## Test cross-entropy and search cost

| Maximum swaps | Greedy loss | Beam-4 loss | Exhaustive loss | Greedy mean evaluations | Beam mean evaluations | Exhaustive evaluations |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 2.17844 | 2.17844 | 2.17844 | 1 | 1 | 1 |
| 1 | 1.81178 | 1.81178 | 1.81178 | 25 | 25 | 25 |
| 2 | 1.68159 | 1.68100 | 1.68100 | 39.56 | 67.20 | 115 |
| 3 | 1.63638 | 1.63485 | 1.63169 | 46.07 | 88.58 | 195 |
| 4 | 1.63212 | 1.63056 | 1.62740 | 47.43 | 95.02 | 210 |

Evaluation counts include the baseline and deduplicate equivalent token sets.
They describe evaluations each search would require independently. Actual GPU
extraction evaluated all 210 sets once, shared across methods: 318,780 training,
56,280 validation, and 114,240 test evaluations. Recorded split extraction times
were approximately 36.9, 9.8, and 15.0 seconds, including data loading and GPU
evaluation but excluding model setup and subsequent CPU search summaries. These
timings use the existing patch cache and batched GPU execution, and are not
measurements of a live sequential oracle's latency.

At four swaps, greedy missed the exact minimum by more than 1e-6 for 21/544 test
recordings; beam missed it for 2/544. The largest gap for either was 1.0373 in
cross-entropy, so rare interactions still matter despite the small average gap.
On validation, greedy missed the minimum for 7/268 recordings and beam found it
for every recording. The single differing test prediction between either
approximate search and exhaustive search did not change correctness.

The exhaustive four-swap oracle actually used an average of **2.36 swaps** on
test. Its chosen counts were:

| Actual swaps | Test recordings |
|---|---:|
| 0 | 14 |
| 1 | 83 |
| 2 | 192 |
| 3 | 201 |
| 4 | 54 |

## Interpretation

There is additional room beyond one swap: allowing two adds 18 correct test
predictions over the one-swap oracle, and allowing four adds 21. Returns diminish
after two or three changes in this small boundary pool. Greedy search captures
most of the available loss improvement, suggesting that a sequential predictor
of the benefit of the next swap is a reasonable training experiment. Its inputs
would need to describe the current selected set or prior swaps; independently
scored single-swap benefits should not simply be added.

The previous learned single-swap policies still only achieved approximately
baseline accuracy. These new results establish available improvement under
label-assisted choices, not that a policy can generalize well enough to recover
it. Out-of-fold supervision remains a useful next step before expanding the
candidate pool or making many unconstrained changes.

## Reproduction and artifacts

```powershell
python -m experiments.hierarchical_mae.swap_oracle `
  --baseline outputs/hierarchical_mae/linear_probe_activity54/best.pt `
  --reference outputs/hierarchical_mae/swap54_boundary/results.json `
  --output-dir outputs/hierarchical_mae/swap54_multi_oracle
```

Use a different output directory for a new completed run. See
[usage and search semantics](SWAP_SELECTION.md#multi-swap-oracle-diagnostics).

Results, per-recording losses/predictions, and chosen sets are saved under
`outputs/hierarchical_mae/swap54_multi_oracle`; console output is in
`outputs/swap54_multi_oracle.log`. `results.json` includes every budget from zero
through four and the provenance of each split cache. Saved choices are labelled
oracle outputs, not classifier or policy checkpoints.

Verification: **93 tests passed**. New tests verify the 210 unique sets, exact
one-swap compatibility, constant token budget, distinct replacements, monotonic
best-so-far loss, and a synthetic interaction where beam crosses a barrier that
stops greedy. The actual run also reproduced the earlier baseline validation
loss and the original single-swap oracle loss and accuracy on all three splits.
