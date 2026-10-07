# Downstream learned token selection

This experiment trains a coarse-conditioned selection policy using class labels.
The default run freezes the pretrained MAE, trains the selector and classification
head, and retains exactly 27 tokens: the three root representations plus 24 selected
tokens from the remaining hierarchy. There is no change to the MAE pretraining
objective, masks or default token selection.

## Run the first experiment

Use a fixed pretrained MAE checkpoint so comparisons share the same encoder. The
snapshot already created for the deterministic selection comparisons can be used:

```powershell
$checkpoint = 'outputs/hierarchical_mae/selection_probes/best_epoch_797_68eabd6852.pt'
python -m experiments.hierarchical_mae.downstream --config experiments/hierarchical_mae/configs/thu_selector.yaml --checkpoint $checkpoint --mode selector_train
```

The default output is `outputs/hierarchical_mae/selector_train`. This is 100 epochs,
batch size 32, classifier learning rate 0.001, selector learning rate 0.0005, and a
frozen encoder. Validation and test use deterministic selection. `best.pt` is chosen
by validation classification loss, then evaluated on test. This is supervised
selector training with a frozen encoder; the learned policy makes it more than an
ordinary linear probe.

The first experiment uses the full recording for training as well as evaluation,
matching the fixed observations used in the existing linear-probe comparisons.
With the standard patch cache enabled, those patches are reusable every epoch.
Set `downstream.learned_selector.training_crops: true` to instead use the existing
training temporal/spatial crop settings and optional crop bank. Evaluation still
uses the fixed evaluation volume.

For 54 tokens with fresh 80–100% temporal and spatial training crops, use
[`thu_selector54_crop80.yaml`](configs/thu_selector54_crop80.yaml). It enables
`training_crops`, sets both crop ranges to `[0.8, 1.0]`, keeps evaluation full-view,
and configures 500 epochs. The classifier head is trained on these cropped views
alongside the selector. See [the commands and crop settings](SET_SELECTION.md#milder-crops-and-yaml-configuration)
for training, probing, and how this differs from the separate bounded-swap runner.

After training, freeze the learned policy and fit a **fresh** linear head using
saved features:

```powershell
python -m experiments.hierarchical_mae.downstream --config experiments/hierarchical_mae/configs/thu_selector_probe.yaml --checkpoint outputs/hierarchical_mae/selector_train/best.pt --mode selector_probe
```

This extracts deterministic selected-token features, fingerprints the trained
selector checkpoint and selector configuration, and trains only the new head.
The preset uses 500 epochs and classifier learning rate 0.01. Results go to
`outputs/hierarchical_mae/selector_probe`. It uses the same selected representations
as the learned policy; labels used to train that policy still constitute supervised
adaptation and should be reported when comparing with heuristic frozen probes.

Optionally finetune the policy, encoder and classifier together:

```powershell
python -m experiments.hierarchical_mae.downstream --config experiments/hierarchical_mae/configs/thu_selector_finetune.yaml --checkpoint outputs/hierarchical_mae/selector_train/best.pt --mode selector_finetune
```

This restores the trained head as well as the policy and backbone, creates a new
optimizer, uses encoder learning rate 0.00002, and enables training crops. It also
accepts a `selector_probe` checkpoint if its newly fitted head is preferred.

All three modes support `--output-dir` and `--resume`. To continue selector training:

```powershell
python -m experiments.hierarchical_mae.downstream --config experiments/hierarchical_mae/configs/thu_selector.yaml --resume outputs/hierarchical_mae/selector_train/last.pt
```

As with other downstream modes, epochs specify the desired total, checkpoints save
optimizer and loader state, and a resumed run retains its earlier best model.
The annealing schedule uses absolute epoch numbers and its own duration, so
extending the total epochs does not silently change the schedule. A new probe or
finetuning stage uses `--checkpoint`; `--resume` continues the same stage.

## Inputs and scoring

`context: patch` concatenates the root tokens' pretrained patch-projection outputs
before the transformer. With the current three-plane, 192-dimensional THU model,
this is a 576-dimensional global input. `context: transformer` instead obtains the
root features from a separate root-only transformer pass.

A shared MLP receives that global context together with each candidate's:

- Nine geometry/time metadata values, including actual token duration.
- `log1p(event_count)` and its ratio to the largest root log-count (clamped denominator).
- Small learned level and representation embeddings.

It predicts a correction to the activity score:

`score_i = activity_prior_weight * log1p(count_i) + correction_i`

The correction's final layer starts at zero. With the default positive activity
prior, initial **deterministic** selection matches activity top-k subject to keeping
all root tokens. The selector can subsequently change region, level and representation
preferences. Counts are not normalized by voxel area/duration. The root tokens are
included in the 27-token budget, and their features are also used to score candidates.
The policy never sees labels as inputs; labels supervise the classification loss.

## Hard selection with relaxed gradients

During training, one Gumbel perturbation is drawn per candidate and added to its
score. Hard top-k selects distinct non-root tokens. The backward surrogate uses
sequential softmax rows, updating logits with `log(1 - probability)` after each
relaxed draw. This follows the subset-relaxation construction of
[Xie and Ermon, IJCAI 2019](https://www.ijcai.org/proceedings/2019/544).

Let `H` be the hard one-hot selection matrix and `S` its soft relaxation. The
straight-through matrix is `H + (S - stop_gradient(S))`. Multiplying this matrix
by the candidate embeddings gives exactly the hard-selected token vectors in the
forward pass, while gradients flow through all candidate scores in the backward
pass. Geometry and count embeddings travel with each selected token. This is a
biased surrogate gradient; it is not a derivative through discrete top-k indices.

The transformer sees exactly the configured budget during both training and
evaluation. Training computes every candidate's patch embedding to support the
relaxed gradient. Evaluation computes selector context and embeds only the selected
tokens for the main transformer pass. Input patch rendering/cache loading currently
still covers all candidates; this is not lazy event rendering. The `transformer`
context option additionally runs the root-only transformer.

The frozen-backbone mode freezes **parameters**, not autograd through the backbone.
Gradients from classification must pass through its transformer to reach the
selection matrix. The final deterministic evaluation uses the ordinary sparse
encoder path and mean pooling; it does not pass soft selection weights to the head.

## Settings and diagnostics

```yaml
downstream:
  learned_selector:
    budget: 27
    context: patch              # or transformer
    hidden_dim: 128
    lr: 0.0005
    activity_prior_weight: 1.0
    temperature: [1.0, 0.25]
    noise_scale: [1.0, 0.0]
    anneal_epochs: 80
    training_crops: false
    grad_clip: 1.0
```

Temperature controls the relaxation; noise scale controls exploration. Each is
linearly interpolated from its first to second value over epochs 0–79, then stays
at its final value. Hard ranking does not depend on relaxation temperature.
Validation/test always use zero noise and hard top-k, regardless of the training
schedule. Ties follow canonical token order. The first level must be `[1,1,1]`,
and the budget must exceed its token count so there is at least one learned choice.

Training/validation metrics include tokens selected by level and representation,
overlap with a root-retaining activity baseline, and the number of distinct tokens
used across the epoch. Full per-token selection frequencies are retained in
`metrics.jsonl` and `results.json`; the terminal prints only compact diagnostics.
Representation statistics follow `sorted(unique representations)` and level
statistics follow the configured hierarchy order. These help detect policies that
never move beyond activity selection or collapse onto a small part of the hierarchy.

Selector-training metrics use stochastic selections until the noise schedule reaches
zero; validation metrics use deterministic selections throughout. Probe training
uses cached vectors and therefore does not repeat token diagnostics each epoch.

The default run caches input patches, not pooled backbone features: a changing
selection changes transformer interactions. Only `selector_probe`, where both the
encoder and policy are frozen, uses the pooled-feature cache. Changing the learned
checkpoint or its configuration produces a different cache key.

For bounded pipeline debugging, `downstream.max_batches` and
`downstream.max_val_batches` limit training and validation batches (`0` means all).
Leave both at zero for reported experiment results. No full selector training run
is implied by the smoke checks; compare validation performance against activity-27,
coarse-27 and the all-token baseline to assess the learned policy.
