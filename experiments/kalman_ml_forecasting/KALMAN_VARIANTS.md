# Training Kalman variants and using them in residual models

The new variants use the existing 8-D state
`[cx, cy, w, h, vx, vy, vw, vh]`. Run all examples from the repository root with
the project Python environment active. `optimize_coupled_kalman.py` now trains
fixed CV, separate learned, and full learned dynamics, despite its historical
filename. Existing commands retain their defaults: full learned dynamics and
fixed legacy scalar noise unless `--optimize-noise` is supplied.

The existing 12-D constant-acceleration baseline and scalar optimizers remain
available. No learned 12-D generator or new 12-D covariance variants were added.

## What the choices mean

`P` is the evolving estimation-error covariance. The learned inputs are `P0`
(initial uncertainty), `Q` (process uncertainty), and `R` (measurement
uncertainty). Each can be frozen or learned independently. `P` itself is updated
by the filter rather than independently fitted at every step.

### Dynamics

| `--dynamics` | Behavior | Possible benefit | Possible shortcut/failure |
| --- | --- | --- | --- |
| `constant_velocity` | Fixed, separate motion equations for each coordinate | Strong baseline with limited ability to invent motion priors; cheap transition | Can look excellent on interpolated, nearly linear trajectories and miss real turns |
| `per_channel` | Learn a 2x2 continuous-time dynamics block for each coordinate | Can model damping, persistence, or coordinate-specific motion without directly using width to predict x | Absolute position or size can still drive its own velocity; it can learn location-dependent or size-dependent priors |
| `full` | Learn the complete 8x8 generator | Can use legitimate relationships between apparent size, position, and motion | Can use width/height or frame location as a proxy for scene, distance, camera motion, or typical trajectory direction |

Learned dynamics use `F(dt) = matrix_exp(A * dt)` for actual timestamp intervals.
The separate blocks use indices `[0,4]`, `[1,5]`, `[2,6]`, `[3,7]`, not contiguous
state slices. A mask enforces zero connections between separate blocks during
training and export. All new structures start at CV dynamics and diagonal noise
when the input config provides only scalar standard deviations.

### Covariance

| `--covariance-structure` | P0 and Q | R | Possible benefit | Possible shortcut/failure |
| --- | --- | --- | --- | --- |
| `legacy_diagonal` | Diagonal with existing tied scales | Existing tied diagonal scales | Reproduces the old optimizer's parameterization; few parameters | Tying x/y and width/height can hide different noise or motion statistics |
| `diagonal` | Independent scale for every state entry | Independent scale per coordinate | More flexible trust without adding cross-coordinate correlations | Can fit dataset-specific smoothness or axis imbalance |
| `per_channel` | Dense 2x2 blocks within each coordinate | Independent diagonal | Models position/velocity uncertainty together while retaining coordinate isolation with separate dynamics | Local filtering can still exploit how the cleaned track was smoothed or interpolated |
| `full` | Full 8x8 matrices | Full 4x4 matrix | Models correlated tracking errors; permits one measurement to help another coordinate | The filter can use size/location correlations during history assimilation even when dynamics are fixed CV |

These are global learned parameters, shared across tracklets, not a neural
network producing sample-specific noise. Full covariance does not directly
insert a new future motion equation. It changes how measurements infer the
state and velocity during history, which can nevertheless create shortcuts.
With no future measurements, Q/R do not change the future mean directly; they
affect the history estimate and propagated uncertainty.

Four independent filters are obtained with separate dynamics, local P0/Q, and
diagonal R. Separate dynamics with global covariance uses a **joint update**;
it has separate motion equations but coupled estimates. Full dynamics can
create cross-coordinate posterior P even when P0/Q/R are diagonal.

Dense covariance uses `C = L L^T`, with positive log-parameterized factor
diagonals and bounded relative factor off-diagonals. `--min-std` and
`--max-std` bound factor diagonals, not covariance eigenvalues. Explicit warm
start matrices must be positive definite and compatible with the requested
structure. Their Cholesky diagonals must be within those bounds and factor
off-diagonal/row-diagonal ratios must have magnitude below 10.

Process noise preserves the existing timestep convention:
`Q(dt) = D(dt) Q_base D(dt)^T`, with
`D = diag(dt, dt, dt, dt, 1, 1, 1, 1)` and the existing tiny-dt clamp.
This is a compatibility convention, not an exact continuous-time diffusion
integral. Keep the timestep and forecast sampling setup consistent when
comparing variants.

## Train any of the twelve combinations and export its residual config

The following PowerShell helper trains one Kalman variant and writes a complete
residual training config with separate output paths. It does not start residual
training automatically. Change `$baseConfig` to your own full dataset/model
config. It must use an 8-D Kalman motion model and contain valid train/val/test
split paths for subsequent residual training.

Sequence-held-out tuning groups all tracks in each training folder together.
It needs at least two folders with eligible samples; more folders give a more
useful validation set. Use `--tune-split track` for the existing track-held-out
protocol if a sequence split is impractical, and report that distinction.

```powershell
$baseConfig = "experiments/kalman_ml_forecasting/configs/base.yaml"

function Train-KalmanVariant {
    param([string]$Name, [string]$Dynamics, [string]$Covariance)
    $report = "outputs/kalman_variants/$Name.json"
    $residualConfig = "outputs/kalman_variants/${Name}_residual.yaml"
    $trainingArgs = @(
        "--config", $baseConfig,
        "--dynamics", $Dynamics,
        "--covariance-structure", $Covariance,
        "--optimize-noise",
        "--epochs", "50", "--lr", "1e-3", "--batch-size", "512",
        "--tune-split", "sequence", "--tune-val-fraction", "0.2",
        "--seed", "123", "--device", "cuda",
        "--objective-weights", "ade_bbox_px=0.25,fde_bbox_px=1,miou=-25",
        "--transition-l2", "0.01", "--covariance-l2", "0.01",
        "--output-json", $report
    )
    python experiments/kalman_ml_forecasting/optimize_coupled_kalman.py @trainingArgs
    if ($LASTEXITCODE -ne 0) { throw "Kalman training failed: $Name" }
    python experiments/kalman_ml_forecasting/coupled_report_to_config.py $report `
        --base-config $baseConfig --use-for-residual-rollout `
        --training-output-dir "outputs/residual_variants/$Name" `
        --output $residualConfig
    if ($LASTEXITCODE -ne 0) { throw "Config export failed: $Name" }
}
```

Run the individual commands you want; running all twelve launches twelve
training jobs sequentially. Each call includes training and config generation.

```powershell
# Fixed CV dynamics: increasing covariance flexibility.
Train-KalmanVariant cv_legacy constant_velocity legacy_diagonal
Train-KalmanVariant cv_diagonal constant_velocity diagonal
Train-KalmanVariant cv_local constant_velocity per_channel
Train-KalmanVariant cv_full constant_velocity full

# Learned separate coordinate dynamics.
Train-KalmanVariant separate_legacy per_channel legacy_diagonal
Train-KalmanVariant separate_diagonal per_channel diagonal
Train-KalmanVariant separate_local per_channel per_channel
Train-KalmanVariant separate_full per_channel full

# Learned fully coupled dynamics.
Train-KalmanVariant coupled_legacy full legacy_diagonal
Train-KalmanVariant coupled_diagonal full diagonal
Train-KalmanVariant coupled_local full per_channel
Train-KalmanVariant coupled_full full full
```

The regularization values above are starting points for validation, not measured
best settings. `--transition-l2` penalizes departure from CV; `--covariance-l2`
penalizes covariance correlations. Neither penalty is included in reported
ADE/FDE/mIoU or the validation selection score. For a controlled structural
comparison, also compare runs with both penalties set to zero.

Then train the residual model using the generated config, for example:

```powershell
python experiments/kalman_ml_forecasting/train.py --config outputs/kalman_variants/cv_local_residual.yaml
python experiments/kalman_ml_forecasting/train.py --config outputs/kalman_variants/coupled_full_residual.yaml
```

Replace the filename with any of the twelve names above. The exporter copies
the base model/data settings, imports the selected Kalman matrices, and sets
`model.initial_state_source: kalman_filter` and
`model.rollout_transition: configured_kalman`. The Kalman filter is frozen during
residual training. `--training-output-dir` redirects checkpoints, logs, and
metrics and clears `resume_checkpoint`. The base config must enable image,
history, or filter-state features to train a residual model; a config with no
features instead runs the baseline-only path.

Exports refuse to overwrite existing YAML files unless `--force` is supplied.
Optimizer JSON output can be overwritten, so use distinct names for seeds and
ablations. An exported matrix is authoritative over scalar noise values for
its covariance. Scalar values remain fallback inputs for other covariances.
Older coupled reports and scalar optimizer reports are still accepted.

## More informative ablations

You can choose covariance structures separately instead of using the shorthand.
For example, global process noise with separate initial and measurement noise:

```powershell
python experiments/kalman_ml_forecasting/optimize_coupled_kalman.py `
    --config experiments/kalman_ml_forecasting/configs/base.yaml `
    --dynamics constant_velocity --optimize-noise `
    --initial-covariance-structure per_channel `
    --process-covariance-structure full `
    --measurement-covariance-structure diagonal `
    --tune-split sequence --epochs 50 `
    --objective-weights "ade_bbox_px=0.25,fde_bbox_px=1,miou=-25" `
    --output-json outputs/kalman_variants/cv_full_q.json
```

This tests whether correlations in motion uncertainty help without also adding
correlated measurement errors. To learn only full P0, holding Q/R and dynamics
fixed:

```powershell
python experiments/kalman_ml_forecasting/optimize_coupled_kalman.py `
    --config experiments/kalman_ml_forecasting/configs/base.yaml `
    --dynamics constant_velocity --initial-covariance-structure full `
    --optimize-initial --no-optimize-process --no-optimize-measurement `
    --tune-split sequence --epochs 50 `
    --objective-weights "ade_bbox_px=0.25,fde_bbox_px=1,miou=-25" `
    --output-json outputs/kalman_variants/cv_full_p0.json
```

`--optimize-initial`, `--optimize-process`, and `--optimize-measurement` override
`--optimize-noise`; their `--no-optimize-*` forms freeze the respective input.
`--freeze-dynamics` freezes an imported custom generator. At least one component
must be trainable. If importing a prior report's YAML, structures and matrices
are retained unless explicitly overridden; a matrix incompatible with a newly
requested structure is rejected rather than silently projected.

Export either ablation using the same converter:

```powershell
python experiments/kalman_ml_forecasting/coupled_report_to_config.py `
    outputs/kalman_variants/cv_full_q.json `
    --base-config experiments/kalman_ml_forecasting/configs/base.yaml `
    --use-for-residual-rollout --training-output-dir outputs/residual_variants/cv_full_q `
    --output outputs/kalman_variants/cv_full_q_residual.yaml
```

Start with CV legacy/diagonal/local variants, then compare CV full covariance
and coupled diagonal covariance. That comparison separates gains from correlated
history updates from gains in future motion equations. Try fully coupled
dynamics plus full covariance after either individual change survives held-out
sequence evaluation. The learned separate variants are useful for checking
whether damping or per-coordinate dynamics alone explain the improvement.

## Preserving the existing 12-D constant-acceleration variant

Copy your full base YAML and set `kalman.motion_model: constant_acceleration`.
Leave the new covariance structures absent or `legacy_diagonal`, and remove
custom generators or explicit covariance matrices. Train with the existing
scalar optimizer, for example:

```powershell
python experiments/kalman_ml_forecasting/optimize_kalman_backprop.py `
    --config configs/kalman_ca.yaml --epochs 50 --lr 1e-2 `
    --objective-weights "ade_bbox_px=0.25,fde_bbox_px=1,miou=-25" `
    --output-json outputs/kalman_variants/ca_legacy.json

python experiments/kalman_ml_forecasting/coupled_report_to_config.py `
    outputs/kalman_variants/ca_legacy.json --base-config configs/kalman_ca.yaml `
    --use-for-residual-rollout --training-output-dir outputs/residual_variants/ca_legacy `
    --output outputs/kalman_variants/ca_legacy_residual.yaml

python experiments/kalman_ml_forecasting/train.py --config outputs/kalman_variants/ca_legacy_residual.yaml
```

The 12-D filter remains available for history initialization and baseline
forecasting. Its residual model still has an 8-D box/velocity state, so the
exporter selects `rollout_transition: constant_velocity` for CA reports. This
preserves the existing residual architecture; it does not introduce a learned
12-D residual rollout. The legacy random-search optimizer also remains usable.
The 8-D variant optimizer rejects CA inputs explicitly.

## Interpreting results when shortcuts may exist

The following are code-supported risks or hypotheses to test, not evidence that
a particular learned model has already exploited them:

1. **Interpolation can make motion artificially predictable.** The dataset
   interpolates cleaned-track x/y/width/height onto label timestamps using
   `np.interp`, then slices history and future windows from that sequence.
   Piecewise linear targets can favor CV and learned smoothing. If a history
   timestamp lies between track knots and the later knot is after the forecast
   anchor, the interpolated history can depend on a future track value. Evaluate
   native observations or strictly causal preprocessing before claiming
   deployment performance; the new variants do not change preprocessing.
2. **Track holdout does not remove shared scene priors.** Different tracks in
   the same sequence can share background, camera movement, distance, and flight
   direction. Sequence holdout tests that risk better. Different folders can
   still come from related recordings, so use recording/session-level split
   files when that metadata is available. Keep the final test split disjoint.
3. **Size may encode motion without being a robust cause.** Full dynamics can
   learn that a particular width predicts a direction or speed. Full covariance
   can infer x velocity from width measurements during history. Such relations
   may describe perspective correctly, or may identify a recording. Compare
   errors by size, position, direction, and sequence; inspect sensitivity when
   only the observed size changes while center history stays fixed.
4. **Separate filters still learn priors.** Learned 2x2 dynamics can map absolute
   coordinate values to their derivatives or damp velocities toward common
   trajectories. Use fixed CV plus local covariance as a lower-capacity
   comparison. A mask removes direct cross-coordinate links, not every shortcut.
5. **More covariance parameters do not guarantee calibrated uncertainty.**
   Training optimizes forecast errors and overlap, not a probabilistic
   likelihood. Multiplying P0/Q/R together by the same positive scale leaves
   Kalman gains and mean predictions unchanged. Their absolute scale therefore
   cannot be identified by these metrics alone. Treat exported covariance
   features as learned filtering features until calibration is evaluated.
6. **Residual models can reintroduce shortcuts.** Image background, raw box
   history, and filter-state positions/sizes can all convey scene identity.
   Compare filter-only, history-only, image-only where feasible, and combined
   residual models on the same sequence split. `filter_covariance_features:
   full` adds uncertainty features; it does not alter the selected filter's
   covariance structure. With fixed linear parameters and identical timestamps,
   covariance is primarily a function of timing/history length, so its features
   can also reveal sampling patterns rather than image-specific uncertainty.

Use the same history/horizon, timestamp grid, sample caps, seeds, objective,
and comparable optimization budgets across variants. Center-only objectives
do not directly assess width/height quality; include box ADE/FDE and overlap.
Reduce adjacent-window redundancy when evaluating confidence in improvements;
the effective number of independent tracks/sequences matters more than the
number of overlapping windows.

The Kalman optimizers read cached `dataset.samples` directly. Their existing
dataset builder does **not** forward `data.box_augmentation` or sample
decorrelation options, and live per-fetch augmentations are bypassed. Do not
assume changing those settings in a base YAML makes the Kalman sweep causal,
decorrelated, or noise-robust. Those settings require a separate optimizer/data
change or a prepared input dataset. They can still apply in residual training
through its own dataset path.

For the selected variant only, add `--run-test-on-best` to its final optimizer
run to evaluate the validation-selected model on the configured test split.
Avoid choosing architectures or regularization from repeated test results.
Reports include the selected config, split mode, seed, sample counts, effective
trainable parameter count, regularization, and optimization history. There are
no claimed dataset-wide performance gains from implementation alone.
