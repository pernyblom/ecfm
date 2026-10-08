# Learn the activity/information mixture

Two selectors learn how much to trust activity versus an information score:

| `downstream.learned_selector.kind` | Activity weight | Predictor inputs |
| --- | --- | --- |
| `global_blend` | One scalar shared by every recording | None; one trainable logit |
| `coarse_blend` | One scalar per recording | Concatenated root patch embeddings, or root transformer features |

Both use `score_i = alpha * norm(log1p(count_i)) + (1-alpha) * norm(info_i)`.
Normalization is min-max over valid candidate tokens in each recording, exactly
as in static `activity_information` selection. The information metric defaults
to `support`; `entropy` and `autocorrelation` also work. The sigmoid weight obeys
`0 < alpha < 1`. Each recording has one weight shared across its levels and
representations, rather than independent per-token corrections.

Presets select **216 tokens**. The local hierarchy has 411 candidates, so this is
approximately 53%. All valid tokens compete for the budget, with canonical tie
handling. Root context does not force root tokens into the selected set. A frozen
global weight therefore reproduces static blended selection at the same weight.

## Stage 1: fit the weight on training data

Use the existing supervised selector-training objective: train the weight and a
temporary classification head using the training split, with the pretrained MAE
frozen. Choose the checkpoint by validation classification loss. Labels supervise
the objective but never enter the predictor inputs. This is a supervised fit,
not an unsupervised noise-calibration objective.

The global model has exactly one trainable selector parameter and never reads
coarse features. The dynamic model predicts one weight from the existing coarse
context path using a small MLP. Its final layer starts with zero weights and a
bias corresponding to alpha=0.9, so both variants initially use the same blend.
Training uses hard top-k in the forward pass and the existing relaxed-top-k
gradient. The main encoder processes exactly 216 tokens.

```powershell
python -m experiments.hierarchical_mae.downstream --config experiments/hierarchical_mae/configs/thu_global_blend216.yaml --checkpoint outputs/hierarchical_mae/best.pt --mode selector_train --output-dir outputs/hierarchical_mae/global_blend216
python -m experiments.hierarchical_mae.downstream --config experiments/hierarchical_mae/configs/thu_coarse_blend216.yaml --checkpoint outputs/hierarchical_mae/best.pt --mode selector_train --output-dir outputs/hierarchical_mae/coarse_blend216
```

Supply a pretrained MAE checkpoint. For a different hierarchy or model size,
update the configuration to match that checkpoint, as for other downstream modes.

Presets train for 100 epochs. Global-logit learning rate is 0.01; the dynamic MLP
uses 0.0005. Selector weight decay is zero so it does not pull the global logit
toward alpha=0.5; the classifier retains its configured weight decay.
`score_scale: 10.0` multiplies the blend before relaxation, affecting training
gradients/exploration without changing deterministic ranking. Temperature anneals
from 1 to 0.25 and Gumbel noise from 0.1 to zero over 80 epochs. Evaluation always
uses deterministic hard selection.

To change the metric or initialization, use matching settings across stages:

```yaml
downstream:
  learned_selector:
    kind: global_blend           # or coarse_blend
    metric: support              # entropy or autocorrelation
    budget: 216
    initial_activity_weight: 0.9 # strictly between 0 and 1
    score_scale: 10.0
    context: patch              # coarse_blend only; transformer also works
```

Raw voxel information scores are enabled automatically. Their histogram settings
come from `data.information_selection` and remain fixed across stages. The score
definitions and their limitations are in [INFORMATION_SELECTION.md](INFORMATION_SELECTION.md).

## Stage 2: freeze the selector and fit a fresh linear probe

```powershell
python -m experiments.hierarchical_mae.downstream --config experiments/hierarchical_mae/configs/thu_global_blend_probe216.yaml --checkpoint outputs/hierarchical_mae/global_blend216/best.pt --mode selector_probe --output-dir outputs/hierarchical_mae/global_blend_probe216
python -m experiments.hierarchical_mae.downstream --config experiments/hierarchical_mae/configs/thu_coarse_blend_probe216.yaml --checkpoint outputs/hierarchical_mae/coarse_blend216/best.pt --mode selector_probe --output-dir outputs/hierarchical_mae/coarse_blend_probe216
```

Encoder and selector parameters are frozen, and a fresh head trains for 500
epochs on cached features. Global alpha is constant. Dynamic alpha is recomputed
from each recording's root features during extraction; freezing the predictor's
parameters does not make its output constant. Caches include the trained
checkpoint digest, selector configuration, and implementation.

Compare with the equal-budget activity baseline:

```powershell
python -m experiments.hierarchical_mae.downstream --config experiments/hierarchical_mae/configs/thu_linear_probe_activity216.yaml --checkpoint outputs/hierarchical_mae/best.pt --mode linear_probe --output-dir outputs/hierarchical_mae/linear_probe_activity216
```

Use the same encoder, split, probe seed, and training settings across these runs.
Blend-predictor initialization preserves the classifier's random state, so the
same seed gives matching initial classifier weights across the three variants.
Repeat with multiple `train.seed` values. Choose metric/settings using validation
and reserve test comparisons for final reporting.

## Finetune with the fitted selector

```powershell
python -m experiments.hierarchical_mae.downstream --config experiments/hierarchical_mae/configs/thu_global_blend_finetune216.yaml --checkpoint outputs/hierarchical_mae/global_blend216/best.pt --mode selector_finetune --output-dir outputs/hierarchical_mae/global_blend_finetune216
python -m experiments.hierarchical_mae.downstream --config experiments/hierarchical_mae/configs/thu_coarse_blend_finetune216.yaml --checkpoint outputs/hierarchical_mae/coarse_blend216/best.pt --mode selector_finetune --output-dir outputs/hierarchical_mae/coarse_blend_finetune216
```

These modes update encoder and classifier while freezing the blend selector's
parameters. They start with the head in the supplied selector checkpoint; a
completed selector-probe checkpoint also works. Global alpha remains constant.
Dynamic alpha is evaluated on each crop and can change as the finetuned encoder's
root features evolve. Selection remains hard, with no gradient through indices.
The presets enable spatial/temporal training crops; validation/test use full views.

The activity finetune baseline is `thu_finetune_activity216.yaml`. The original
`kind: coarse` selector retains its existing joint selector/encoder finetuning
behavior. Freezing after stage 1 applies to the two new blend kinds.

## Weights, diagnostics, and continuation

`selection.activity_weight` logs mean alpha across recordings, and
`activity_weight_squared` logs mean squared alpha. Their difference
`E[alpha^2] - E[alpha]^2` estimates variance (clamp tiny negative roundoff to zero).
Token frequencies, level/representation counts, and overlap with ordinary
activity top-k are logged alongside them.

Global-blend `results.json` also reports the checkpoint's exact `activity_weight`.
To use this constant without the wrapper, copy it into `downstream.selection`
with `strategy: activity_information`, `combination: blend`, the matching metric
and histogram settings, and `budget: 216`. Fit an ordinary fresh probe using the
original MAE checkpoint. Frozen selector-probe results include `feature_selection`
diagnostics for every split, preserved on cache hits; these describe the weights
used during extraction, not changing weights during head training.

Continue any stage with its configuration and `--resume <stage>/last.pt`.
Increase `downstream.epochs` to the desired total. New stages use `--checkpoint`.
Cross-stage loading checks kind, metric, budget, context, predictor size, score
settings, and effective histogram settings. Resume additionally applies the
existing data and training compatibility checks.

## Verification

```powershell
python -m pytest tests/test_information_selection.py tests/test_hierarchical_mae.py -q
```

Tests cover gradients to both predictors, global fitting without coarse access,
independent per-recording dynamic weights, static/global equivalence, hard-forward
and sparse-evaluation agreement, budgets, frozen selectors in downstream stages,
cache-hit diagnostics, checkpoint compatibility, and continuation.

The bounded local run in `outputs/hierarchical_mae/blend216_smoke_matched_20261008` uses
the local pretrained checkpoint and GPU: two training epochs, two batches of four
recordings per epoch, eight validation/test recordings, and one bounded finetuning
epoch per variant. Weights and a same-head activity comparison are in
`summary.json`. This verifies the 216-token path and weight updates; the briefly
trained heads do not establish accuracy gains.
