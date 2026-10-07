# Conservative one-swap selection

This downstream experiment starts from an **existing activity linear-probe
checkpoint**, including its frozen encoder and trained classifier. It learns to
replace at most one selected token, retaining the exact activity set otherwise.
Pretraining and the existing Gumbel selector are unchanged.

```powershell
python -m experiments.hierarchical_mae.swap_experiment `
  --baseline outputs/hierarchical_mae/linear_probe_activity54/best.pt `
  --output-dir outputs/hierarchical_mae/swap54_boundary
```

Defaults: 54 tokens, the four lowest-ranked selected tokens as removal
candidates, and the six highest-ranked unselected tokens as insertion candidates.
This evaluates 24 single swaps plus the unchanged baseline. Activity ties use
canonical token order, exactly as the existing activity heuristic; no extra
root-retention rule changes the baseline.

The command runs two policy variants:

- `coarse`: concatenated root patch projections, plus both candidates' geometry,
  count, relative count, activity rank, level, and representation.
- `local`: the same inputs plus the two candidates' patch projections.

Training targets are the decrease in cross-entropy under the frozen activity
classifier, measured by actually running each alternative through the encoder.
Only training labels form regression targets. The policy never receives labels,
alternative transformer outputs, or measured loss improvements as inputs.
Normalization statistics also come only from training examples.

The zero-initialized policy starts with no changes. Validation loss chooses the
epoch and a threshold from a fixed grid; the unchanged activity policy is an
explicit candidate. A swap occurs only if its predicted gain exceeds the
threshold. Policies are locked before test extraction. Each policy and activity
then receives a fresh linear probe for seeds 7, 17, and 27, using the baseline
checkpoint's probe settings and validation-loss checkpoint selection. The seed-7
activity probe should reproduce the original run.

Outputs include:

- `results.json`: fixed-head and fresh-probe results, overlap diagnostics, and
  a clearly labelled oracle diagnostic using actual labels.
- `validation_selection.json`: policy choices saved before test access.
- `policy_coarse.pt`, `policy_local.pt`: normalization, scorer weights,
  threshold, proposal settings, and baseline checkpoint path/digest.
- `probe_<variant>_<seed>.pt`: fresh classifier weights.
- Fingerprinted `train_*.pt`, `validation_*.pt`, and `test_*.pt` feature banks.
  Re-running an interrupted experiment reuses compatible completed banks and
  deterministically restarts training. Completed result directories are protected
  against accidental overwriting. This is not optimizer-state resume.

Use `--drops`, `--adds`, `--hidden`, `--policy-epochs`, `--policy-lr`, and
`--probe-seeds` for controlled experiments. More proposals increase extraction
cost. The default bank occupies roughly 400 MB for THU, in addition to existing
patch caches. Use a new output directory for each experiment.

For inference, `swap_selection.selected_features` accepts the frozen backbone,
an event view, trained `SwapPolicy`, threshold, and saved proposal settings. It
scores candidates from cheap patch projections, chooses zero or one swap, and
runs **one** transformer pass on exactly the selected budget. It does not
evaluate all alternatives. The baseline checkpoint must remain available to
restore its encoder; apply either the original classifier or the saved fresh
probe head to the returned pooled features. These checkpoints use the standalone
swap experiment format, not the existing `downstream --mode selector_probe`
format.

For example, after moving a batch's `source` view to the same device:

```python
import torch
from experiments.hierarchical_mae.downstream import Classifier
from experiments.hierarchical_mae.feature_cache import file_digest
from experiments.hierarchical_mae.model import HierarchicalMAE
from experiments.hierarchical_mae.swap_selection import SwapPolicy, selected_features

device = 'cuda'
saved = torch.load('outputs/hierarchical_mae/swap54_boundary/policy_local.pt',
                   map_location='cpu', weights_only=True)
baseline_path = saved['baseline']  # Update this if transferring machines.
assert file_digest(baseline_path) == saved['baseline_digest']
baseline = torch.load(baseline_path, map_location='cpu', weights_only=True)
model = Classifier(HierarchicalMAE(saved['config']),
                   saved['config']['downstream']['num_classes'], True).to(device)
model.load_state_dict(baseline['model'])
model.eval().requires_grad_(False)
policy = SwapPolicy(saved['input_dim'], saved['hidden']).to(device).eval()
policy.load_state_dict(saved['policy'])
# Optional: use the separately fitted classifier instead of the original head.
probe = torch.load('outputs/hierarchical_mae/swap54_boundary/probe_local_7.pt',
                   map_location='cpu', weights_only=True)
model.head.load_state_dict(probe['head'])
features, choices = selected_features(
    model.backbone, view, policy, saved['threshold'], saved['budget'],
    saved['drops'], saved['adds'], saved['inputs'])
logits = model.head(features)
```

The labelled oracle asks whether useful swaps exist in the evaluated proposal
set. It is not a deployable selector or an achievable accuracy guarantee.
Single swaps can interact; this implementation deliberately does not combine
independently scored changes. A limited candidate set can miss useful changes
farther down the activity ranking.

## Regularized linear control

To test whether the MLP policies overfit, run a ridge-regression scorer on the
same completed feature banks:

```powershell
python -m experiments.hierarchical_mae.swap_ridge `
  --source outputs/hierarchical_mae/swap54_boundary `
  --output-dir outputs/hierarchical_mae/swap54_ridge
```

This fits standardized linear scorers with five regularization strengths for
each input variant. Validation loss selects the input variant, regularization,
and threshold, including the unchanged activity baseline as a fallback. The
selected policy receives three fresh probes with the same seeds. All selection
trials are saved. The prior run's test results are not used for fitting or model
selection, but repeated experiments on this test set are exploratory.

Its `policy.pt` uses `swap_ridge.RidgePolicy(saved['input_dim'])`, with the saved
state dictionary, rather than `SwapPolicy`; the same `selected_features` inference
function works. Fresh heads are saved as `probe_7.pt`, `probe_17.pt`, and
`probe_27.pt`. All other proposal and baseline fields have the same meaning.

## Multi-swap oracle diagnostics

```powershell
python -m experiments.hierarchical_mae.swap_oracle `
  --baseline outputs/hierarchical_mae/linear_probe_activity54/best.pt `
  --reference outputs/hierarchical_mae/swap54_boundary/results.json `
  --output-dir outputs/hierarchical_mae/swap54_multi_oracle
```

This keeps the original classifier frozen and uses labels to search for lower
cross-entropy selections at budgets zero through four swaps. The candidate pool
is fixed per recording: the four lowest-ranked selected activity tokens and the
six highest-ranked excluded tokens. All resulting sets still have 54 tokens.
There are 1, 24, 90, 80, and 15 distinct sets at exactly 0, 1, 2, 3, and 4 swaps.
The 210 sets are evaluated once per recording and losses/predictions are cached.

- Greedy accepts the best next swap only if it reduces loss, stopping otherwise.
- Beam search (default width 4) retains four exact-depth candidates and expands
  them, including temporarily worse sets that could lead to a better combination.
- Exhaustive search supplies the exact minimum loss within this fixed pool.

Every method reports its best-so-far set, retaining no-change and earlier
solutions. Thus reported per-example loss cannot increase with the swap budget;
accuracy need not be monotonic because the objective is cross-entropy. Swaps
remove distinct members of the original selected pool and add distinct members
of the original excluded pool; candidates are not re-ranked between steps.

`--reference` checks the baseline checkpoint digest and reproduces the original
single-swap oracle loss and accuracy. The command defaults to full training,
validation, and test splits; `--splits validation test` restricts extraction.
`--drops`, `--adds`, `--max-swaps`, and `--beam-width` allow further diagnostics.
Larger pools grow combinatorially. Completed output directories are protected;
interrupted runs reuse compatible completed split caches.

`results.json` records accuracy, loss, average actual swaps, and cumulative
unique candidate evaluations per search budget. Search evaluation counts are
simulated from the cached losses: actual GPU extraction evaluates all 210 sets
once, shared by all methods. The report separately records actual extraction
time and evaluation count. `<split>_choices.pt` saves chosen state indices and
their removed/added pool positions; the referenced bank contains original token
IDs, candidate IDs, predictions, and losses so selections can be reconstructed.

These are label-assisted diagnostics only. They neither train a deployable
selector nor refit a classifier on label-selected features.

See [the completed local multi-swap results](MULTI_SWAP_RESULTS_2026-10-07.md)
for accuracy, loss, search cost, and interpretation.
