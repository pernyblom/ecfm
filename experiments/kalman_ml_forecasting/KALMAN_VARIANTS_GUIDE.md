# Understanding the Kalman variants for UAV forecasting

The experiment predicts where an unmanned aerial vehicle (UAV) will appear in
an image and how large its bounding box will be. A Kalman filter combines a
motion prediction with observed boxes, accounting for uncertainty in both.
The twelve variants test two separate questions: how much freedom the motion
equations need, and how much freedom the uncertainty model needs. More freedom
may improve forecasts, but the comparison must establish whether it works on
sequences the model has not seen.

## What the filter sees and predicts

Each sample supplies a short history of boxes. A box has four numbers: horizontal
center, vertical center, width, and height. The filter also estimates how quickly
each number changes, giving eight state values in total. Growing width, for
example, is a change in apparent image size; it is not a direct measurement of
the aircraft's physical speed or distance.

The sweep uses a nominal 0.4 seconds of history to forecast the next 0.8 seconds.
At the dataset's usual cadence this gives 12 observed boxes, including the final
observation, followed by 24 future boxes. During the history the filter repeatedly
predicts, compares its prediction with an observed box, and adjusts its estimate.
During the forecast it receives no future observations.

FRED provides the box histories through `cleaned_tracks.txt`, aligned to label
timestamps. Kalman training uses these boxes and timestamps without loading
camera images or event streams. This measures forecasting from prepared tracks;
performance also depends on the quality and preprocessing of those tracks.

## Three ways to describe motion

**Fixed constant velocity**, abbreviated **CV**, extends each estimated rate of
change into the future. A UAV moving right is expected to keep moving right at
the same image speed. Width and height also have their own rates. Training tunes
uncertainty while keeping these motion equations fixed.

**Learned separate dynamics** fits a small motion rule for each box coordinate
and its velocity. It can learn that horizontal motion tends to slow down, for
example. The horizontal rule cannot directly read vertical position, width, or
height. Each coordinate's absolute value can still influence its own velocity,
so this model can learn common locations or sizes in the training sequences.

**Learned coupled dynamics** fits connections among all eight state values.
Changing apparent size can help predict horizontal motion, for example. This
could capture useful perspective effects, but it can also learn relationships
specific to particular scenes or recordings. These are global learned motion
rules shared across samples, rather than rules produced from each sample's image.

## Four ways to describe uncertainty

The filter has three uncertainty inputs: **initial uncertainty** describes how
unsure it is when a history starts; **process uncertainty** describes departures
from its motion prediction; **measurement uncertainty** describes errors in
observed boxes. In technical descriptions these are called **P0**, **Q**, and
**R**. The filter updates its current uncertainty, **P**, as observations arrive.

The covariance structure determines which errors may be related. Two errors
are correlated when knowing about one changes what we expect about the other.
For example, an uncertain position estimate may be associated with an uncertain
velocity estimate.

| Name in run labels | What can be learned | What it tests |
| --- | --- | --- |
| `legacy` | Diagonal uncertainty with several scales shared between coordinates. | Whether the existing compact uncertainty model is sufficient. |
| `diagonal` | A separate uncertainty scale for each state value and each observed coordinate. | Whether different coordinates need different amounts of trust. |
| `local` | Position and velocity uncertainty may be related within each coordinate; observed-coordinate errors remain separate. | Whether local position–velocity relationships help estimate motion. |
| `full` | Initial and process uncertainty may connect all eight state values; measurement uncertainty may connect all four observed coordinates. | Whether relationships among tracking errors help estimate the complete state. |

All three uncertainty inputs are trained in this sweep. These are shared
parameters across tracklets, rather than uncertainty predictions from a neural
network. A diagonal input does not force the evolving uncertainty P to stay
diagonal: the motion equations can create relationships as the filter runs.

## What each of the twelve variants tests

Each row below corresponds to one case in
[`configs/kalman_variants_sweep.yaml`](configs/kalman_variants_sweep.yaml).
Generated filenames add a numeric prefix to these names.

| Variant | Motion and uncertainty | Main comparison question |
| --- | --- | --- |
| `cv_legacy` | Fixed CV with shared diagonal scales. | How well does the existing compact model work under the common training protocol? |
| `cv_diagonal` | Fixed CV with independent scales. | Does giving every state value its own uncertainty improve the compact model? |
| `cv_local` | Fixed CV with local position–velocity uncertainty. | Can better estimates of each coordinate's motion help without connections to other coordinates? |
| `cv_full` | Fixed CV with full uncertainty. | Can relationships among observed errors improve history estimates while future motion equations stay fixed? |
| `separate_legacy` | Learned separate motion with shared diagonal scales. | Do learned motion rules help without expanding the uncertainty model? |
| `separate_diagonal` | Learned separate motion with independent scales. | Do separate motion rules and separate trust levels work well together? |
| `separate_local` | Learned separate motion with local uncertainty. | How far can four independent coordinate filters go? |
| `separate_full` | Learned separate motion with full uncertainty. | Do shared history updates help when future motion equations remain separate? |
| `coupled_legacy` | Learned coupled motion with shared diagonal scales. | Do connections between motion coordinates help with compact uncertainty inputs? |
| `coupled_diagonal` | Learned coupled motion with independent scales. | Do coupled motion rules help when each state value has its own uncertainty scale? |
| `coupled_local` | Learned coupled motion with local uncertainty inputs. | Are local uncertainty inputs sufficient when the motion equations connect coordinates? |
| `coupled_full` | Learned coupled motion with full uncertainty. | Does combining both forms of flexibility improve forecasts on unseen sequences? |

`separate_full` still uses a joint history update: separate equations for future
motion do not make its estimates independent. Conversely, `coupled_local` can
develop uncertainty relationships between coordinates through its motion
equations, even though its uncertainty inputs contain only local connections.

Compare `cv_local` with `cv_full` to examine relationships in history estimation.
Compare `cv_diagonal` with `coupled_diagonal` to examine learned connections in
motion. Compare `separate_diagonal` with `coupled_diagonal` to ask whether those
connections add value beyond learning a rule for each coordinate.

## How training and comparison work

The supplied sweep creates twelve runs with 50 training epochs each, a common
learning rate and batch size, and seed 123. All start from constant-velocity
motion and the same scalar uncertainty settings inherited from `base.yaml`.
The learned variants initially have zero additional correlations. The objective
combines average box error, final box error, and box overlap. Small penalties
discourage large departures from CV and strong uncertainty correlations; their
values are starting settings, not established optimal values.

Training reads the canonical FRED training split. It reserves 20 percent of its
folders, rounded to a whole number, for internal validation. All tracklets from
one folder stay together. The best epoch is selected using validation forecasts.
The canonical test split is reserved for evaluating the chosen approach after
model selection. This protocol helps test generalization beyond tracks in a
shared scene, although folders from related recordings can still share context.

Report **ADE**, the average error over forecast steps; **FDE**, the error at the
last forecast step; and **mIoU**, the mean overlap between predicted and target
boxes. Lower ADE/FDE and higher mIoU are better. This implementation reports
center errors and errors over the four box values in pixels; box error also
accounts for width and height. Compare the same metric, history, and horizon
across variants. Parameter counts and inference time help judge the cost of any
improvement. Repeating the sweep with additional seeds helps assess stability.

Two limitations matter when interpreting the results. First, interpolation of
cleaned tracks can make trajectories unusually smooth and can let an interpolated
history value depend on a track observation after the forecast anchor. Evaluation
with native observations or strictly causal preprocessing is needed before
claiming performance in live use. Second, forecast error and overlap do not
establish that uncertainty values are calibrated probabilities. Better forecasts
alone do not justify a claim such as “95 percent confidence.”

## Kalman training and residual training

Kalman training fits the shared motion and uncertainty parameters described
above. An optional second stage trains a neural residual model to correct the
filter's future motion using the configured features, which can include images
or event representations. The fitted Kalman parameters stay frozen during that
stage. Exporting a report creates the runtime filter configuration and can also
prepare that residual training configuration.

The separate 12-state constant-acceleration baseline also exists. It adds a rate
of change for each velocity. It uses the older scalar optimizer and is outside
this twelve-variant sweep; learned 12-state dynamics and the new covariance
structures are not implemented for it.

For generation, launch, batch export, and residual training commands, see
[Training Kalman variants](KALMAN_VARIANTS.md#generate-and-launch-the-complete-sweep).
