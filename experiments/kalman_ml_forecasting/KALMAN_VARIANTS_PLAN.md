# Kalman variant audit and implementation plan

Reviewed against the working tree on 2026-10-14. This is the original audit and
proposal. The 8-D variants have since been implemented; see
[KALMAN_VARIANTS.md](KALMAN_VARIANTS.md) for current behavior and training examples.
The proposed 12-D extensions remain outside the implementation scope.

## Terminology and current behavior

The box coordinates are center x, center y, width, and height. The CV state is
`[cx, cy, w, h, vx, vy, vw, vh]`; CA appends four accelerations.

- `F(dt)` is the state transition. The coupled model learns a continuous-time
  generator `A`, with `F(dt) = matrix_exp(A * dt)`.
- `P` is the evolving state estimation error covariance, not a fixed noise matrix.
- `P0` is the initial state error covariance; `Q` is process noise covariance;
  `R` is measurement noise covariance. These are separate configuration and
  optimization choices. Learn `P0`, `Q`, and `R`, then propagate `P` with the
  Kalman equations rather than making every posterior `P` a free parameter.

`models/kalman_filter.py` already stores and updates complete 8x8 or 12x12 `P`
matrices. Its initialization, prediction, and measurement update nevertheless
construct **diagonal** `P0`, `Q`, and `R` using `diag_embed` and squared standard
deviations. Several standard deviations are tied between coordinates.

CV/CA dynamics never mix coordinates. Their `P` develops position/velocity/
acceleration correlations within each coordinate but remains block diagonal
when grouped by coordinate. These are mathematically four separate filters,
implemented together in a larger matrix.

`models/coupled_kalman_filter.py` learns all 64 entries of an 8x8 generator.
`--optimize-noise` learns nine tied scalar standard deviations, not dense
covariance entries. A coupled transition can generate cross-coordinate entries
in `P` despite diagonal `P0`, `Q`, and `R`.

The fixed runtime accepts an arbitrary 8x8 generator, including a supplied
generator with four separate coordinate blocks. There is no optimizer mode
that constrains learned dynamics to those blocks. Coupled CA is also absent:
custom generators are currently restricted to 8x8.

`model.filter_covariance_features: full` only exposes covariance entries to the
residual model. It does not change how noise is configured or learned.

## Combination audit

Here, covariance structure refers to configurable/learnable `P0` and `Q`.
`R` is chosen separately: one scalar per coordinate for separate measurement
noise, or a dense 4x4 matrix for correlated measurement noise.

| Dynamics | Existing tied diagonal covariance | Dense covariance within each coordinate | Dense covariance across coordinates |
| --- | --- | --- | --- |
| Fixed separated CV, 8-D | Implemented | Missing | Missing |
| Fixed separated CA, 12-D | Implemented | Missing | Missing |
| Learned separated CV blocks, 8-D | Missing optimizer constraint; supplied generator works | Missing | Missing |
| Learned full generator, 8-D | Implemented | Missing | Missing |
| Learned full generator, 12-D | Missing | Missing | Missing |

A fully dense *global* covariance and four independent filter updates are
incompatible: each update must account for cross-coordinate uncertainty.
Use either four dense local covariance blocks with separate filters, or a joint
Kalman update with separated motion equations and dense global covariance.
The second option preserves separate dynamics but couples the estimates.

If the request concerns only dense initial `P0`, that capability is also missing
and should be independently selectable without requiring dense `Q` or `R`.

## Proposed configuration

Keep the existing `motion_model` values and scalar configuration valid. Add
independent structure choices, with the existing behavior as the default:

- `dynamics_structure: full | per_channel` for custom/learned generators.
  Fixed CV/CA already have per-coordinate structure.
- `initial_covariance_structure: legacy_diagonal | diagonal | per_channel | full`.
- `process_covariance_structure: legacy_diagonal | diagonal | per_channel | full`.
- `measurement_covariance_structure: legacy_diagonal | diagonal | full`.
- Explicit `initial_covariance`, `process_covariance`, and
  `measurement_covariance` matrices for exported/fixed runtimes.
- Independent optimizer controls for learning dynamics, `P0`, `Q`, and `R`.

`legacy_diagonal` retains tied scalar parameters; `diagonal` gives each entry
its own scale. `per_channel` permits complete 2x2 CV or 3x3 CA covariance
blocks. In the existing state order, the CV index groups are `[0,4]`, `[1,5]`,
`[2,6]`, `[3,7]`; CA adds indices 8, 9, 10, and 11 to the respective groups.
These blocks are not contiguous slices of the current state layout.

Validate matrix dimensions, finite values, symmetry, positive definiteness,
and zeros required by the selected structure. Define explicit matrices as
authoritative over legacy scalar defaults; reject conflicting explicitly
supplied scalar and matrix inputs with an actionable error. Reject unknown
covariance/structure keys rather than silently dropping them.

## Implementation sequence

1. **Refactor shared covariance construction and runtime primitives.**
   In `models/kalman_filter.py`, introduce one common covariance resolver and
   matrix-based predict/update path used by functional, tensor-parameter,
   configured, and learnable runtimes. Preserve Joseph updates and linear
   solves. Keep the public standard-deviation APIs as compatibility wrappers;
   distinguish new matrix arguments explicitly rather than guessing their
   meaning from shape. Initialize with `P0` and use full `R` in innovation and
   Joseph calculations.

   Preserve the existing timestep convention initially. Its process covariance
   generalization is `Q(dt) = D(dt) Q_base D(dt)^T`, where `D` scales the first
   four state entries by the existing clamped `dt` and leaves derivatives at
   one. With diagonal `Q_base = diag(process_std**2)`, this reproduces existing
   behavior. Record this convention in reports; changing to continuous-time
   diffusion integration is a separate experiment.

2. **Add learnable covariance structures.**
   Create a shared covariance parameter module. Parameterize dense covariance
   with `C = L L^T`, using a lower-triangular `L` with positive diagonal via
   softplus plus a small floor. Use four local factors for `per_channel` mode,
   independent scales for `diagonal`, and existing tied log-scales for legacy
   mode. Initialize to current noise covariances so added structure starts with
   zero cross-correlations. Apply sensible scale bounds and optional penalties
   on correlations; do not clamp individual covariance entries arbitrarily.

3. **Implement and expose the variants in this order.**
   First add separated CV/CA with dense local `P0`/`Q` blocks and diagonal `R`.
   Then add fixed separated dynamics with global dense `P0`/`Q`, and optional
   dense `R`, using a joint update. Finally add learned per-channel CV
   generators and learned full CV generators with every covariance structure.
   Enforce the dynamics mask in the forward parameterization so forbidden
   connections stay zero throughout training. Keep 12-D learned generators as
   a follow-up; they are not necessary for the requested CV combinations.

4. **Connect optimization, export, and consumers.**
   Extend `optimize_kalman_backprop.py` and `optimize_coupled_kalman.py` to use
   the shared learnable covariance module and independent learning controls.
   Prefer backprop for dense matrices; keep `optimize_kalman.py` random search
   for scalar modes and make unsupported optimization choices explicit.
   Export exact matrices, state layout, structure choices, timestep convention,
   and a report schema version. Extend `coupled_report_to_config.py` and
   `_format_yaml` to support those matrices and older scalar-only reports.
   Route the matrices through `ConfiguredBoxKalmanFilter`, baseline evaluation,
   visualization, residual history initialization, and covariance features.
   Preserve existing checkpoint loading and the frozen-filter behavior in
   residual training. Existing configured residual rollout remains 8-D.

5. **Verify structure and numerical behavior.**
   Add regression checks that legacy configurations and diagonal matrix inputs
   reproduce existing states, covariances, and predictions for irregular times.
   Compare per-channel mode against four explicit reference filters, including
   CA indexing. Verify that a cross-coordinate measurement influences another
   estimate under correlated covariance with fixed separated dynamics, while
   independent mode preserves isolation. Check finite gradients into dense
   off-diagonal parameters and zero forbidden dynamics connections. Test
   covariance symmetry/positive definiteness over long histories and small
   noise scales. Cover malformed matrices, device/dtype movement, report/YAML
   round trips, and optimizer-to-fixed-runtime parity for every new structure.

6. **Run a controlled variant sweep.**
   Extend `configs/kalman_motion_model_sweep.yaml` or add a covariance sweep.
   Compare legacy CV/CA, local dense CV/CA, separated dynamics with global
   covariance, existing full dynamics with diagonal noise, and full dynamics
   with dense covariance. Include variants that change only `P0`, only `Q`,
   or only `R` to identify the source of improvement. Use identical track splits,
   history/horizon settings, seeds, optimization budgets, and validation
   selection; evaluate the held-out test split only after selection. Report
   center and box ADE/FDE, mIoU, parameter count, and inference time. Include
   box/overlap terms when judging width and height performance.

## Audit verification

- Existing targeted tests: `10 passed, 31 deselected` using
  `pytest tests/test_kalman_ml_forecasting.py -k 'coupled or kalman_constant_acceleration or kalman_tensor_parameters' -q`.
- A float64 history check confirmed nonzero within-coordinate position/velocity
  covariance and exactly zero cross-coordinate covariance for CV and CA.
  Introducing cross-coordinate generator entries produced nonzero
  cross-coordinate `P` in the coupled runtime.
- Passing explicit `initial_covariance`, `process_covariance`, and
  `measurement_covariance` keys to the current config parser retained none of
  them: these capabilities are absent and unknown keys are silently ignored.
