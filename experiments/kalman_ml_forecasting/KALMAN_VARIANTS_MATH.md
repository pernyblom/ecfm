# Kalman filter equations and variant matrices for UAV forecasting

The UAV forecasting experiment estimates an aircraft's image bounding box and
predicts how that box will move and change size. Its twelve main Kalman variants
share one filtering algorithm. They differ in the motion matrix and in the
structure of three uncertainty matrices. This companion assumes familiarity
with vectors, matrix multiplication, transpose, and inverse; it introduces the
probabilistic notation as it is needed.

The equations below describe the implemented eight-state filters. A final
section covers the separate twelve-state constant-acceleration baseline.
For a less mathematical introduction, see the
[audience guide](KALMAN_VARIANTS_GUIDE.md); for commands, see
[training and sweep instructions](KALMAN_VARIANTS.md).

## An intuitive introduction

A motion model supplies a prediction: if the aircraft was moving right, where
should its box be now? A measurement supplies new evidence: where does the
observed box say it is? The Kalman filter maintains both an estimate and an
uncertainty about that estimate, then uses their uncertainties to combine the
prediction and measurement.

In one dimension, with a predicted position variance of $p$ and a measurement
error variance of $r$, the correction weight is $p/(p+r)$. A relatively uncertain
prediction receives a larger correction from the measurement. In several
dimensions, that scalar weight becomes the **Kalman gain matrix**. Its entries
can also let evidence about one state value correct another state value.

A **covariance matrix** describes uncertainty. Its diagonal entries are
variances, the squares of standard deviations. An off-diagonal entry describes
how two estimation errors vary together. A positive position–velocity covariance,
for example, means that underestimating position tends to accompany
underestimating velocity. It lets a position measurement revise velocity even
though velocity is not directly measured.

Under the standard linear Gaussian model, the filter propagates the conditional
mean and covariance exactly. This implementation additionally clips normalized
box values, as described below, so its complete behavior includes that practical
constraint as well as the linear filter equations.

## State and matrix reference

Write the box and its rates of change as column vectors:

$$
q_k=\begin{bmatrix}c_x&c_y&w&h\end{bmatrix}^{\mathsf T},
\qquad
v_k=\begin{bmatrix}v_x&v_y&v_w&v_h\end{bmatrix}^{\mathsf T},
\qquad
x_k=\begin{bmatrix}q_k\\v_k\end{bmatrix}\in\mathbb R^8.
$$

Centers, widths, and heights are normalized by the image dimensions. Time is in
seconds, so CV velocities are normalized box units per second. Width and height
rates describe changes in apparent image size, rather than physical aircraft
dimensions. The learned motion models retain this state layout but can change
the relationship between the velocity entries and the derivative of $q$.

| Symbol | Shape | Meaning |
| --- | --- | --- |
| $x_k$, $\hat x_k$ | $8\times1$ | Latent state and its estimate. |
| $z_k$ | $4\times1$ | Observed box $[c_x,c_y,w,h]^{\mathsf T}$. |
| $F_k$ | $8\times8$ | Transition over one timestamp interval. |
| $A$ | $8\times8$ | Continuous-time dynamics generator, with $F_k=\exp(A\Delta t_k)$ for learned dynamics. |
| $H$ | $4\times8$ | Selects box coordinates from the state. |
| $P_0$ | $8\times8$ | Initial state estimation-error covariance. |
| $P_k^-$, $P_k^+$ | $8\times8$ | Covariance before and after a measurement update. |
| $Q_{\mathrm{base}}$, $Q_k$ | $8\times8$ | Process covariance parameter and its timestep-scaled value. |
| $R$ | $4\times4$ | Measurement error covariance. |
| $S_k$ | $4\times4$ | Covariance of the measurement innovation. |
| $K_k$ | $8\times4$ | Kalman gain, mapping box discrepancies to state corrections. |

Here $I_d$ denotes the $d\times d$ identity matrix, $0$ a zero block of the
indicated size, and $\operatorname{diag}$ a diagonal matrix. Superscript
$\mathsf T$ means transpose. Superscripts $-$ and $+$ mean **before** and
**after** assimilating a measurement, rather than negative and positive values.

The state and observation model is

$$
x_k=F_kx_{k-1}+\eta_k,\qquad z_k=Hx_k+\epsilon_k,
$$

where the usual modeling assumptions are zero-mean, temporally independent
process and measurement errors, mutually independent, with covariances $Q_k$
and $R$. There is no control-input matrix in this experiment. All twelve
variants use the same observation matrix:

$$
H=\begin{bmatrix}I_4&0\end{bmatrix}
=\begin{bmatrix}
1&0&0&0&0&0&0&0\\
0&1&0&0&0&0&0&0\\
0&0&1&0&0&0&0&0\\
0&0&0&1&0&0&0&0
\end{bmatrix}.
$$

Thus a measurement directly observes all four box coordinates and none of the
four velocities. In Kalman fitting, $z_k$ comes from aligned cleaned-track
boxes, not from loading an image or running a detector.

## Prediction and measurement update

For each observed timestamp after the first, let
$\Delta t_k=\max(t_k-t_{k-1},10^{-6})$. Starting from the previous corrected
estimate, the **prediction** is

$$
\hat x_k^-=F_k\hat x_{k-1}^+,
\qquad
P_k^-=F_kP_{k-1}^+F_k^{\mathsf T}+Q_k.
$$

The first term propagates existing uncertainty through the motion equations.
The second adds uncertainty about departures from those equations.

For the **measurement update**, compute the innovation, its covariance, and
the gain:

$$
e_k=z_k-H\hat x_k^-,\qquad
S_k=HP_k^-H^{\mathsf T}+R,\qquad
K_k=P_k^-H^{\mathsf T}S_k^{-1}.
$$

Then update the state and covariance:

$$
\hat x_k^+=\hat x_k^-+K_ke_k,
$$

$$
P_k^+=(I_8-K_kH)P_k^-(I_8-K_kH)^{\mathsf T}+K_kRK_k^{\mathsf T}.
$$

The last equation is the **Joseph form**. In exact arithmetic it agrees with
the shorter covariance update $(I_8-K_kH)P_k^-$, but is more robust numerically.
The implementation obtains $K_k$ through a linear solve rather than explicitly
forming $S_k^{-1}$.

### Initialization and clipping

Every tracklet starts a fresh filter. Before assimilating its first observation,
the code sets

$$
\hat x_0^-=\begin{bmatrix}z_0\\0_{4\times1}\end{bmatrix},
\qquad P_0^-=P_0,
$$

and performs the measurement update with $z_0$. Its innovation is zero, so the
mean initially stays the same, but the covariance changes. $P_0$ therefore
means the covariance **before** this first update; it is not the first posterior
covariance.

After each history update, the four estimated box values are clipped to
$[0,1]$. Velocities are not clipped by that operation. The covariance is not
adjusted to describe a truncated Gaussian. The same box clipping occurs after
each forecast step. The linear equations here describe the filter before these
clipping operations.

### Forecasting without future measurements

At the forecast anchor, let $(\hat x_a^+,P_a^+)$ be the final history estimate.
Future timestamps use the prediction equations repeatedly, with no innovations
or measurement updates:

$$
\hat x_{a+j}=F_{a+j}\hat x_{a+j-1},\qquad
P_{a+j}=F_{a+j}P_{a+j-1}F_{a+j}^{\mathsf T}+Q_{a+j}.
$$

The predicted box is the first four state values, clipped as above. $Q$ and
$R$ influence future mean forecasts through the state inferred from history;
once that state and $F$ are fixed, future process covariance changes the
uncertainty but does not directly add a displacement to the mean.

## Three motion matrix structures

### Fixed constant velocity

The `cv` variants use

$$
A_{\mathrm{CV}}=\begin{bmatrix}0&I_4\\0&0\end{bmatrix},
\qquad
F_{\mathrm{CV}}(\Delta t)=\exp(A_{\mathrm{CV}}\Delta t)
=\begin{bmatrix}I_4&\Delta t I_4\\0&I_4\end{bmatrix}.
$$

Thus $q_{k+1}=q_k+\Delta t v_k$ and $v_{k+1}=v_k$. Horizontal position,
vertical position, width, and height each follow their own motion equation.
None of the entries of $A_{\mathrm{CV}}$ is learned.

### Learned separate dynamics

For `separate` variants, define four diagonal matrices
$D_\alpha=\operatorname{diag}(\alpha_1,\ldots,\alpha_4)$ and similarly
$D_\beta,D_\gamma,D_\delta$. The generator has structure

$$
A_{\mathrm{sep}}=\begin{bmatrix}D_\alpha&D_\beta\\D_\gamma&D_\delta\end{bmatrix},
\qquad F_{\mathrm{sep}}(\Delta t)=\exp(A_{\mathrm{sep}}\Delta t).
$$

There are 16 learned entries. To see the four independent motion blocks,
reorder the state to

$$
\tilde x=[c_x,v_x,c_y,v_y,w,v_w,h,v_h]^{\mathsf T}=\Pi x,
$$

where $\Pi$ is a permutation matrix. Then

$$
\Pi A_{\mathrm{sep}}\Pi^{\mathsf T}
=\operatorname{blockdiag}(A_1,A_2,A_3,A_4),
\qquad
A_i=\begin{bmatrix}\alpha_i&\beta_i\\\gamma_i&\delta_i\end{bmatrix}.
$$

Each pair obeys $\dot q_i=\alpha_iq_i+\beta_iv_i$ and
$\dot v_i=\gamma_iq_i+\delta_iv_i$. Damping is possible through $\delta_i$;
absolute coordinate values can influence motion through $\alpha_i$ and
$\gamma_i$. These are general learned linear blocks, rather than a model that
only learns acceleration. They start at
$\alpha_i=\gamma_i=\delta_i=0$, $\beta_i=1$, reproducing CV.

In the code's original zero-based indexing, the permitted pairs are
`[0,4]`, `[1,5]`, `[2,6]`, `[3,7]`. A mask keeps every forbidden connection zero
throughout fitting and export. Matrix exponentiation preserves this separation.

### Learned coupled dynamics

The `coupled` variants learn all 64 entries of

$$
A_{\mathrm{full}}=
\begin{bmatrix}A_{qq}&A_{qv}\\A_{vq}&A_{vv}\end{bmatrix},
\qquad A_{qq},A_{qv},A_{vq},A_{vv}\in\mathbb R^{4\times4},
\qquad F_{\mathrm{full}}(\Delta t)=\exp(A_{\mathrm{full}}\Delta t).
$$

For example, an entry in $A_{vq}$ can let observed width influence the evolution
of horizontal velocity. All blocks are initialized to CV: $A_{qv}=I_4$ and
the others zero. This allows learned cross-coordinate motion relationships,
including relationships that may depend on a particular scene.

Here $\exp$ is the **matrix exponential**, not the elementwise exponential.
The same generator supplies a transition for each actual timestamp interval.
Bounds and regularization on generator entries do not guarantee that its
dynamics are stable over arbitrarily long forecasts.

## Four covariance structures

The sweep applies one structure family to $P_0$, $Q_{\mathrm{base}}$, and $R$,
with the local family using diagonal $R$. Their values are learned globally
across tracklets. The evolving $P_k$ is calculated by the filtering equations;
it is not a separate collection of fitted matrices.

### Legacy diagonal covariance

The legacy family uses tied standard deviations. Its exact eight-state pattern
is

$$
P_0=\operatorname{diag}(s_p^2,s_p^2,s_s^2,s_s^2,
s_v^2,s_v^2,s_v^2,s_v^2),
$$

$$
Q_{\mathrm{base}}=\operatorname{diag}(q_p^2,q_p^2,q_s^2,q_s^2,
q_v^2,q_v^2,q_{sv}^2,q_{sv}^2),
\qquad
R=\operatorname{diag}(r_p^2,r_p^2,r_s^2,r_s^2).
$$

Here $s_p,s_s,s_v$ are `initial_pos_std`, `initial_size_std`, and
`initial_vel_std`; the four $q$ values correspond to `process_pos_std`,
`process_size_std`, `process_vel_std`, and `process_size_vel_std`; and the two
$r$ values are `measurement_pos_std` and `measurement_size_std`.
There are nine trainable scalar scales in total.

### Independent diagonal covariance

The diagonal family removes the tying:

$$
P_0=\operatorname{diag}(p_1,\ldots,p_8),\quad
Q_{\mathrm{base}}=\operatorname{diag}(q_1,\ldots,q_8),\quad
R=\operatorname{diag}(r_1,\ldots,r_4),
$$

with all entries positive. These symbols denote variances, rather than standard
deviations. There are $8+8+4=20$ parameters. It can assign horizontal and vertical
position different uncertainty without introducing correlated input errors.

### Local position and velocity covariance

For either $C=P_0$ or $C=Q_{\mathrm{base}}$, the local family permits

$$
C=\begin{bmatrix}
\operatorname{diag}(a_1,a_2,a_3,a_4)&\operatorname{diag}(b_1,b_2,b_3,b_4)\\
\operatorname{diag}(b_1,b_2,b_3,b_4)&\operatorname{diag}(d_1,d_2,d_3,d_4)
\end{bmatrix}.
$$

Equivalently,

$$
\Pi C\Pi^{\mathsf T}=\operatorname{blockdiag}(C_1,C_2,C_3,C_4),
\qquad C_i=\begin{bmatrix}a_i&b_i\\b_i&d_i\end{bmatrix},
$$

where $a_i>0$, $d_i>0$, and $a_id_i-b_i^2>0$ ensure positive definiteness.
Position and velocity errors can correlate within one coordinate, but errors
across coordinates cannot correlate in these input matrices. Measurements have
one value per coordinate, so $R=\operatorname{diag}(r_1,r_2,r_3,r_4)$.
The parameter count is $4\cdot3+4\cdot3+4=28$.

### Full covariance

For $C=P_0$ or $C=Q_{\mathrm{base}}$, full covariance allows

$$
C=\begin{bmatrix}C_{qq}&C_{qv}\\C_{qv}^{\mathsf T}&C_{vv}\end{bmatrix},
$$

where $C_{qq}$ and $C_{vv}$ are symmetric $4\times4$ matrices and $C_{qv}$
is an unrestricted $4\times4$ block, subject to $C$ being positive definite.
$R$ is a full symmetric positive-definite $4\times4$ matrix. There are
$36+36+10=82$ parameters.

Full covariance can change which measurements influence each estimated state
value, even when $F$ remains fixed CV. It does not, by itself, change the future
motion equations.

### Timestep scaling of process covariance

All four families use the implementation's common scaling convention:

$$
D(\Delta t)=\begin{bmatrix}\Delta t I_4&0\\0&I_4\end{bmatrix},
\qquad Q_k=D(\Delta t_k)Q_{\mathrm{base}}D(\Delta t_k)^{\mathsf T}.
$$

Thus a position variance receives a factor $\Delta t^2$, a position–velocity
covariance receives a factor $\Delta t$, and a velocity variance receives no
additional factor. The diagonal scaling entries are clamped below by $10^{-6}$
in code. This convention preserves the earlier scalar implementation. It is
not the continuous-time diffusion discretization
$\int_0^{\Delta t}\exp(A\tau)Q_c\exp(A^{\mathsf T}\tau)\,d\tau$.
Keep the timestamp grid consistent when interpreting comparisons.

### Positive definiteness during learning

Nonlegacy covariance parameters use $C=LL^{\mathsf T}$, where $L$ is lower
triangular and

$$
L_{ii}=\exp(\ell_i),\qquad
L_{ij}=10L_{ii}\tanh(u_{ij})\quad (i>j\text{ and permitted}),
$$

with other entries zero. Diagonal mode permits no off-diagonal entries; local
mode permits only entries belonging to the same coordinate pair; full mode
permits all lower-triangular entries. The optimizer bounds $L$'s diagonal
scales and the unconstrained off-diagonal parameters. Bounds on factor diagonals
are not bounds on all covariance eigenvalues. Legacy scales are learned in log
space and squared to obtain variances.

## Matrix specification of all twelve variants

The covariance labels below refer to the matrix families defined above.
Every row uses the same $H$ and the same prediction and Joseph update equations.

| Sweep case | Generator or transition | $P_0$ | $Q_{\mathrm{base}}$ | $R$ | Learned parameters |
| --- | --- | --- | --- | --- | --- |
| `cv_legacy` | Fixed $F_{\mathrm{CV}}$ | Legacy | Legacy | Legacy | $0+9=9$ |
| `cv_diagonal` | Fixed $F_{\mathrm{CV}}$ | Diagonal | Diagonal | Diagonal | $0+20=20$ |
| `cv_local` | Fixed $F_{\mathrm{CV}}$ | Local | Local | Diagonal | $0+28=28$ |
| `cv_full` | Fixed $F_{\mathrm{CV}}$ | Full | Full | Full | $0+82=82$ |
| `separate_legacy` | $\exp(A_{\mathrm{sep}}\Delta t)$ | Legacy | Legacy | Legacy | $16+9=25$ |
| `separate_diagonal` | $\exp(A_{\mathrm{sep}}\Delta t)$ | Diagonal | Diagonal | Diagonal | $16+20=36$ |
| `separate_local` | $\exp(A_{\mathrm{sep}}\Delta t)$ | Local | Local | Diagonal | $16+28=44$ |
| `separate_full` | $\exp(A_{\mathrm{sep}}\Delta t)$ | Full | Full | Full | $16+82=98$ |
| `coupled_legacy` | $\exp(A_{\mathrm{full}}\Delta t)$ | Legacy | Legacy | Legacy | $64+9=73$ |
| `coupled_diagonal` | $\exp(A_{\mathrm{full}}\Delta t)$ | Diagonal | Diagonal | Diagonal | $64+20=84$ |
| `coupled_local` | $\exp(A_{\mathrm{full}}\Delta t)$ | Local | Local | Diagonal | $64+28=92$ |
| `coupled_full` | $\exp(A_{\mathrm{full}}\Delta t)$ | Full | Full | Full | $64+82=146$ |

The counts are effective trainable entries under the sweep's `--optimize-noise`
setting, excluding frozen parameters and masked generator entries. They count
the parameterization, not statistically identifiable degrees of freedom.

The `cv_legacy`, `cv_diagonal`, `cv_local`, `separate_legacy`,
`separate_diagonal`, and `separate_local` variants preserve coordinate separation
through both prediction and measurement update. In the permuted ordering they
are four independent two-state filters. Diagonal covariance inputs can develop
position–velocity correlations within each pair during prediction.

`cv_full` and `separate_full` have separate motion equations but joint estimates.
Their full covariances allow cross-coordinate corrections during history.
Every `coupled` variant can produce cross-coordinate posterior covariance
through $FPF^{\mathsf T}$, even with diagonal or local covariance inputs.

## How an observed position updates an unobserved velocity

Consider one coordinate pair just before a measurement update:

$$
\hat x^-=\begin{bmatrix}\hat q\\\hat v\end{bmatrix},\qquad
P^-=\begin{bmatrix}p&c\\c&s\end{bmatrix},\qquad
H_i=\begin{bmatrix}1&0\end{bmatrix},\qquad R_i=r.
$$

The shared update equations reduce to

$$
S=p+r,\qquad
K=\frac{1}{p+r}\begin{bmatrix}p\\c\end{bmatrix},
\qquad
\hat q^+=\hat q+\frac{p}{p+r}(z-\hat q),\quad
\hat v^+=\hat v+\frac{c}{p+r}(z-\hat q).
$$

For an illustrative normalized-coordinate example, take
$\hat q=0.5$, $\hat v=0.1$, $p=0.04$, $c=0.006$, $s=0.01$,
$r=0.01$, and $z=0.52$. The innovation is $0.02$ and
$K=[0.8,0.12]^{\mathsf T}$, giving

$$
\hat x^+=\begin{bmatrix}0.516\\0.1024\end{bmatrix},
\qquad
P^+=\begin{bmatrix}0.008&0.0012\\0.0012&0.00928\end{bmatrix}.
$$

The box was observed slightly farther along than expected, so both position
and velocity increase. A CV forecast $0.1$ seconds later is
$0.516+0.1(0.1024)=0.52624$. Even if input covariances start diagonal,
CV prediction can create the needed $c$: for a diagonal prior, the
position–velocity block of $FPF^{\mathsf T}$ contains $\Delta t s$.

In a full filter, the analogous correction is

$$
\Delta\hat v_x
=K_{v_x,c_x}e_{c_x}+K_{v_x,c_y}e_{c_y}
+K_{v_x,w}e_w+K_{v_x,h}e_h.
$$

Local filters force the last three coefficients to zero for this row. Full
covariance or coupled dynamics can make them nonzero. This explains how width
evidence can change inferred horizontal velocity through the history update,
even when the subsequent forecast uses fixed CV equations.

## What is fitted and what stays fixed

The sweep fits global parameters by differentiating through the filter and its
future rollout. It does not fit a different $A$, $P_0$, $Q$, or $R$ for every
sample. The training objective is

$$
\mathcal L=0.25\,\mathrm{ADE}_{\mathrm{box,px}}
+\mathrm{FDE}_{\mathrm{box,px}}-25\,\mathrm{mIoU}
+0.01\,\Omega_A+0.01\,\Omega_C,
$$

where $\Omega_A=\|A-A_{\mathrm{CV}}\|_F^2/64$. The covariance penalty sums,
over nonlegacy covariance modules, the mean squared entries of
$\operatorname{Corr}(C)-I$, with
$\operatorname{Corr}(C)_{ij}=C_{ij}/\sqrt{C_{ii}C_{jj}}$.
For diagonal modules this penalty is zero. These penalties affect fitting;
validation selection uses the forecast metric terms without the penalties.

The twelve variants start from CV dynamics and the same base scalar noise
settings. New covariance correlations start at zero. Entire training folders
are reserved for internal validation, and the sweep leaves the canonical test
split out of fitting and model selection.

Forecast error and overlap do not identify absolute uncertainty scale:
scaling $P_0,Q,R$ together by a positive constant leaves the Kalman gains and
mean predictions unchanged. The learned covariance matrices therefore require
separate calibration evaluation before interpreting confidence regions as
probability guarantees. Interpolation of cleaned tracks also affects the
causality and difficulty of the forecasting task; see the
[data interpretation discussion](KALMAN_VARIANTS.md#interpreting-results-when-shortcuts-may-exist).

### Optional neural residual stage

After exporting a fitted filter, the residual model can initialize from its
history estimate and use its configured transition. With a learned four-value
acceleration correction $a_\theta$ at a future step, its un-clipped update is

$$
\hat x_{k+1}=F(\Delta t)\hat x_k+
\begin{bmatrix}\tfrac12\Delta t^2 I_4\\\Delta t I_4\end{bmatrix}a_\theta.
$$

The neural model can use configured image, history, and filter features. The
Kalman parameters remain frozen in this stage. This adds a learned mean
correction; it does not automatically provide a covariance model for the neural
correction. `model.filter_covariance_features: full` exposes covariance entries
as neural input features and does not change the filter's covariance structure.

## The separate constant acceleration baseline

The older constant-acceleration model appends four acceleration values:

$$
x_{\mathrm{CA}}=\begin{bmatrix}q\\v\\a\end{bmatrix}\in\mathbb R^{12},
\qquad
F_{\mathrm{CA}}(\Delta t)=\begin{bmatrix}
I_4&\Delta t I_4&\tfrac12\Delta t^2 I_4\\
0&I_4&\Delta t I_4\\
0&0&I_4
\end{bmatrix},
\qquad
H_{\mathrm{CA}}=\begin{bmatrix}I_4&0&0\end{bmatrix}.
$$

The same predict/update equations apply with $P,Q\in\mathbb R^{12\times12}$,
$K\in\mathbb R^{12\times4}$, and $I_{12}$ in the Joseph update. Initial
velocities and accelerations are zero. Legacy $P_0$ adds two tied center
acceleration variances and two tied size acceleration variances, and legacy
$Q_{\mathrm{base}}$ adds corresponding process acceleration variances. $R$
keeps its four-coordinate legacy pattern. Its process scaling is
$D_{\mathrm{CA}}=\operatorname{diag}(\Delta t I_4,I_4,I_4)$, subject to
the same small-timestep clamp.

Only legacy scalar covariance fitting is implemented for this twelve-state
baseline; learned twelve-state generators and the new covariance families are
outside the twelve-case sweep. Its optional residual stage still uses an
eight-state box/velocity rollout with a CV transition, rather than a
twelve-state neural rollout.

## Implementation references

The matrix structures and implementation details correspond to these sources:

- [Filter equations, initialization, timestep scaling, and clipping](models/kalman_filter.py).
- [Learned dynamics masks, trainability, and exported matrices](models/coupled_kalman_filter.py).
- [Covariance factors and correlation regularization](models/kalman_covariance.py).
- [Optimization objective and validation selection](optimize_coupled_kalman.py).
- [Neural residual update](models/kalman_residual.py).
- [The twelve-case sweep](configs/kalman_variants_sweep.yaml).
