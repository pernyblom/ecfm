# Selection by spatiotemporal structure

These deterministic selectors implement three ideas from
[information_selection_ideas.md](information_selection_ideas.md), with no policy
training or label access. They rank eligible tokens during MAE pretraining,
frozen linear probing, or encoder finetuning. They do not change reconstruction
targets, strict masking exclusions, or the physical token budget.

## Run individual tests

Use a **pretrained MAE checkpoint** with the existing downstream pipeline:

```powershell
python -m experiments.hierarchical_mae.downstream --config experiments/hierarchical_mae/configs/thu_linear_probe_information54.yaml --checkpoint outputs/hierarchical_mae/best.pt --mode linear_probe --output-dir outputs/hierarchical_mae/information_support54
python -m experiments.hierarchical_mae.downstream --config experiments/hierarchical_mae/configs/thu_linear_probe_activity_information54.yaml --checkpoint outputs/hierarchical_mae/best.pt --mode linear_probe --output-dir outputs/hierarchical_mae/activity_information_support54
```

The first uses support alone; the second gives activity and support equal weight.
Both fit a fresh classifier with the same frozen encoder and 54-token budget.
Change `metric` to `entropy` or `autocorrelation` to compare other scores.

```yaml
data:
  information_selection:
    bins: [8, 8, 8]             # local x,y,t histogram, independent of patch size
    support_saturation: 3.0
downstream:
  selection:
    strategy: activity_information  # information for structure alone
    metric: support
    budget: 54
    combination: blend         # or product
    activity_weight: 0.5       # blend only: 1=activity, 0=structure
```

The same selection options work under top-level `selection` for MAE pretraining.
For finetuning use the same options with `--mode finetune`. If an information
strategy is configured without `data.information_selection`, its default settings
are applied automatically. An explicit data block computes all three scores even
for activity/all baselines, allowing the comparison suite to share patch caches.
Ordinary configurations compute no additional scores.

## Repeatable ablation suite

```powershell
python -m experiments.hierarchical_mae.information_experiment --checkpoint outputs/hierarchical_mae/best.pt --output-dir outputs/hierarchical_mae/information54
```

Defaults: THU's 500-epoch linear-probe configuration, budget 54, seeds 7/17/27,
and **18 variants / 54 independent probes**:

| Variant | Score |
| --- | --- |
| All | All tokens, reference with a larger attention budget |
| Activity | Highest `log1p(event_count)` |
| Coarse | Lowest hierarchy level first |
| Each metric alone | Raw structure score |
| Each metric blended with activity | Activity weights 0.25, 0.5, 0.75 |
| Each metric multiplied by activity | `log1p(event_count) * max(structure, 0)` |

Use `--metrics support --weights 0.5 --seeds 7` for six probes before committing
to the full suite. `--dry-run` prints the complete plan without reading the
checkpoint or training. `--config`, `--epochs`, `--device`, `--workers`, and
`--budget` override the defaults. An existing activity-probe checkpoint is not a
pretrained MAE checkpoint; use the original MAE file.

Each probe uses fixed full-recording views and the pretraining validation holdout.
Its best epoch is chosen by validation loss, then evaluated on test. Compare
metrics/weights using **validation**, reserving test comparisons for final
reporting; the suite does not automatically tune weights from test accuracy.
The same seeds pair classifier initialization and training order across variants.

`experiment.json` records configuration, selections, checkpoint digest, source
digests, and recording order/labels/file metadata for every split. Each
variant/seed gets its own checkpoints, metrics, and results. The
top-level `results.json` reports validation/test loss and accuracy, individual seed
values, and mean/population standard deviation across seeds. Repeating the same
command reuses completed probes and resumes interrupted ones from `last.pt`;
changed configuration or provenance requires a new output directory.

For a bounded pipeline check with a matching smoke MAE checkpoint:

```powershell
python -m experiments.hierarchical_mae.information_experiment --config experiments/hierarchical_mae/configs/smoke.yaml --checkpoint outputs/hierarchical_mae_smoke/best.pt --output-dir outputs/hierarchical_mae_smoke/information --budget 6 --metrics support --weights 0.5 --seeds 7 --epochs 1 --max-batches 1 --workers 0
python -m pytest tests/test_information_selection.py tests/test_hierarchical_mae.py -q
```

Smoke results check execution; they do not establish representation quality.

## Scores and interpretation

The dataset constructs a raw, unnormalized, polarity-pooled 3-D event histogram
inside each voxel. Coordinates are relative to that voxel's full spatial/temporal
extent. Scores are computed once per voxel per level and copied to all of its
representation tokens. A CSTR mean-time channel is never mistaken for event mass.

* **Support:** event-mass-weighted `min(neighbor_mass / saturation, 1)`, averaged
  over the voxel's events. The neighborhood includes spatial offsets in a 3x3
  window and temporal offsets -1/0/+1. It excludes the center spatial position
  at **all** time offsets, leaving 24 bins. Thus events confined to one spatial
  bin cannot support each other just by repeating through time. Boundaries do not
  wrap; neighboring voxels are not consulted. Range: [0,1].
* **Entropy:** `1 - H(V/sum(V))/log(number_of_bins)`. Range: [0,1]. This measures
  concentration, not geometric coherence: a hot pixel can score highly, and
  sparse random voxels have a finite-sample concentration bias.
* **Autocorrelation:** sum of centered adjacent-bin products along the three
  positive axes, divided by total centered squared mass, minus its exact expected
  value under uniform permutation of the histogram bins. For K bins and P valid
  adjacent pairs per axis, the null expectation is `-P/(K*(K-1))`. This preserves
  the exact histogram values and event count and needs neither Monte Carlo
  shuffles nor a random seed. It is an excess correlation, **not a z-score** and
  not the null obtained by independently relocating every event. Negative values
  are possible. Constant histograms receive zero.

All metrics return zero for empty voxels. Pure metrics rank their raw scores.
Blends min-max normalize activity and structure **separately over eligible tokens
in each recording**, then sum with the configured weight. Constant signals become
zero. Excluded targets/padding cannot alter that normalization. Products use raw
log counts and clamp negative structure to zero. Ties retain canonical token
order, including when every candidate is empty.

The scores are heuristics for structure, not established measures of downstream
information. Support still grows with local density; entropy has sampling bias;
correlation can favor structured sensor artifacts. Fixed local histogram bins
also correspond to different physical sizes and durations across levels. Compare
the isolated scores with the hybrids to check whether information beyond event
count actually improves validation performance. The count-conditioned residual,
spectral flatness, and shuffle-standardized z-score ideas are not implemented.

## Data access and cache cost

`source['information_scores']` is a float32 `[N,3]` tensor, batched to `[B,N,3]`,
in `support, entropy, autocorrelation` order. The selector reads candidate raw
voxel structure throughout the hierarchy. This adds sensing/histogram cost and
is more input access than a policy restricted to root patches. Input rendering
still covers all candidates; this is not lazy rendering.

Patch caches include the effective histogram settings and score implementation,
preserve/validate the score tensors, and are shared across metrics and blend
weights with identical data settings. Feature caches additionally fingerprint
the selection settings and implementation. Existing caches are invalidated by
the implementation changes; enabling scores also changes the patch-cache key.
The automated checks cover equal-count coherence versus isolated noise, hot
pixels, boundaries, the exact permutation null, crop-relative binning, strict
masking, model loss, cache integrity, feature extraction batch invariance, and
an actual tiny six-variant probe suite.
