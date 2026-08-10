# Incremental ML Overhead Benchmark

This document describes
[`benchmark_incremental_ml_overhead.py`](benchmark_incremental_ml_overhead.py),
a focused CUDA benchmark for the additional computation introduced by the
image-conditioned residual forecasting path after the observation history has
already been processed by a Kalman filter.

The benchmark is deliberately narrower than an end-to-end latency benchmark.
It does not answer how long the complete forecasting system takes from loading
an image and a box history to returning predictions. Instead, it answers:

> Given an already filtered state at the forecast anchor, how much CUDA work is
> added by extracting visual features and applying learned residual dynamics?

This boundary avoids directly subtracting a CPU Kalman measurement from a CUDA
ResNet measurement. It is useful when the research question concerns the
incremental computational price of adding learned image-conditioned dynamics
to an existing Kalman forecasting pipeline.

## Quick usage

### Windows installation from a fresh checkout

This benchmark requires an NVIDIA GPU and a working CUDA build of PyTorch.
Compiled results additionally require a Windows Triton wheel compatible with
the installed PyTorch minor version. The following known-working combination
is used by this repository:

| Component | Version |
|---|---|
| Python | 3.12 (64-bit) |
| PyTorch | 2.6.0+cu124 |
| torchvision | 0.21.0+cu124 |
| triton-windows | 3.2.x |

From PowerShell in a fresh repository checkout:

```powershell
py -3.12 -m venv .venv312
. .\.venv312\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install --index-url https://download.pytorch.org/whl/cu124 "torch==2.6.0+cu124" "torchvision==0.21.0+cu124"
python -m pip install "triton-windows>=3.2,<3.3"
python -m pip install -e .
python -m pip check
```

Do not replace `triton-windows` with the ordinary `triton` package on Windows.
The PyTorch/Triton minor versions must remain paired; PyTorch 2.6 corresponds
to Triton 3.2. These wheels bundle the CUDA and small C compiler components
needed for this CUDA workload, so installing the full CUDA Toolkit or Visual
Studio Build Tools is not normally necessary. A compatible, up-to-date NVIDIA
driver is still required.

Confirm that the intended virtual environment and GPU are active:

```powershell
python -c "import sys, torch, torchvision, triton; print(sys.executable); print(torch.__version__, torchvision.__version__, triton.__version__); print(torch.version.cuda, torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No CUDA GPU')"
```

Expected core output includes `2.6.0+cu124`, `0.21.0+cu124`, `3.2.x`, CUDA
`12.4`, and `True`. The CUDA version displayed by `nvidia-smi` may be newer;
that value is the driver's maximum supported CUDA version, not the runtime
bundled in the PyTorch wheel.

Run the benchmark from the repository root:

```bash
python experiments/kalman_ml_forecasting/benchmark_incremental_ml_overhead.py
```

The first compiled run can pause while TorchInductor creates and caches its
kernels. This compilation occurs before the timed measurements.

### Windows troubleshooting

- If `torch.cuda.is_available()` is `False`, check that `python` resolves to
  `.venv312\Scripts\python.exe`, run `nvidia-smi`, and confirm that the CUDA
  PyTorch wheel—not a CPU-only wheel—is installed.
- `ModuleNotFoundError: No module named 'triton'` means `triton-windows` was not
  installed in the active environment. The distribution is named
  `triton-windows`, but its Python import is `triton`.
- `FileExistsError: [WinError 183]` from a TorchInductor cache rename is a
  known PyTorch 2.5 Windows issue. Confirm that `torch.__version__` is 2.6.0
  and that Triton is 3.2.x; installing Triton alone does not fix PyTorch 2.5.
- After changing Python, PyTorch, Triton, CUDA, or compiler versions, stale
  caches can be removed from
  `%LOCALAPPDATA%\Temp\torchinductor_<Windows-user>` and
  `%USERPROFILE%\.triton\cache` before retrying. Only remove these generated
  cache directories, and do so while no Python benchmark process is running.
- A warning that there are not enough SMs for `max_autotune_gemm` is not a
  compilation failure. This benchmark requests `reduce-overhead`, and results
  are valid when the JSON contains normal component measurements under
  `compiled` rather than `available: false`.

The script is self-contained and does not read an experiment YAML file. Its
protocol is defined by constants near the top of the script. The defaults are:

- CUDA execution;
- batch size 1;
- 12 history steps;
- 24 future steps;
- one step every `1/30` second;
- an 800 ms forecast horizon;
- 64x64, 224x224, and 1280x720 image inputs;
- 500 warm-up iterations per measured component;
- 5000 measured iterations per component.

Image dimensions are written as `(width, height)` pairs:

```python
IMAGE_SIZES = ((64, 64), (224, 224), (1280, 720))
```

The tensors supplied to PyTorch follow the normal
`[batch, channels, height, width]` convention. Therefore `(1280, 720)` creates
a tensor with shape `[1, 3, 720, 1280]`.

## Benchmark boundary

The benchmark separates historical filtering from future prediction:

```text
box history
    |
    v
Kalman history filtering                 excluded from timing
    |
    v
filtered position/velocity state
    |
    +--> image encoding                  timed
    +--> box-history feature encoding    timed
    +--> learned residual rollout        timed
    +--> zero-residual control rollout   timed separately
```

The Kalman filter processes the 12 historical boxes once before any timed
region. It produces an eight-dimensional state containing normalized box
position and size followed by their velocities:

```text
[cx, cy, w, h, vx, vy, vw, vh]
```

Only this final state is passed into the rollout benchmarks. Kalman covariance
propagation, measurement updates, and historical filtering are therefore not
part of the reported learned overhead.

All measured tensors are already resident on the GPU. Dataset access, image
decoding, spatial crop construction, host-to-device transfer, model loading,
and compilation are excluded.

## Model instantiated by the script

For every configured image size, the script creates a new
`KalmanResidualForecaster` with one `cstr3` representation and a ResNet-18
backbone. It uses the experiment's normal model components with fixed settings:

- ResNet-18 with three input channels and a 128-dimensional pooled output;
- a 256-dimensional image-fusion output;
- relative box-history features;
- a 128-dimensional history encoding;
- a residual head with 256 hidden units and two hidden layers;
- four predicted acceleration residuals, covering center and box size;
- initialization from the final Kalman-filter state.

The weights are randomly initialized. This is appropriate for computational
timing because trained values do not change tensor shapes or the sequence of
operations. This script is not intended to evaluate forecasting accuracy.

## Timed components

### Image encoding

The image stage runs ResNet-18 once and applies the image-fusion projection:

```text
input image -> ResNet-18 -> global pooling -> projection -> image feature
```

The resulting image feature is computed once per forecast request. It is then
reused at all 24 rollout steps. ResNet-18 is not rerun for each future step.

The reported `image_encoding` time includes both the backbone and the image
fusion layer.

### History encoding

The history stage converts the 12 past boxes to relative-position features,
adds relative timestamps, flattens the sequence, and applies the history MLP.
It is evaluated once per forecast request.

This stage is reported separately because it is ML-specific post-filter work,
even though it is much smaller than the image backbone and recurrent residual
rollout.

### Learned residual rollout

The learned rollout starts from the final Kalman state. At each of the 24
future steps it concatenates:

- the image feature reused from the image stage;
- the history feature reused from the history stage;
- the current position/velocity state;
- the current step duration.

The residual MLP predicts acceleration for `cx`, `cy`, `w`, and `h`. Position
and velocity are then updated with constant-acceleration equations:

```text
p_next = p + v*dt + 0.5*a*dt^2
v_next = v + a*dt
```

The new state is used by the following step, so the rollout is autoregressive.
The image and history encoders still execute only once; only the residual head
and state update repeat 24 times.

### Zero-residual control rollout

The zero-residual rollout starts from the same final Kalman state, uses the
same 24 step durations, and produces predictions through:

```text
p_next = p + v*dt
v_next = v
```

It does not evaluate the residual MLP, predict acceleration, propagate a
Kalman covariance, add process noise, or perform measurement updates. It is
therefore best described as a constant-velocity state rollout initialized by
Kalman, rather than a complete Kalman prediction rollout.

Its purpose is to estimate the common overhead of the autoregressive loop,
state slicing, position update, clamping, prediction collection, and stacking.

## Derived measurements

The script derives the incremental rollout cost from medians:

```text
incremental residual rollout
    = median learned residual rollout
    - median zero-residual rollout
```

It then estimates total post-filter learned overhead:

```text
total ML overhead
    = median image encoding
    + median history encoding
    + incremental residual rollout
```

The raw component measurements should be treated as the primary observations.
The differences are useful summaries, but subtracting two noisy latency
distributions can amplify measurement uncertainty. Do not reconstruct the
derived result by subtracting rounded numbers from a table; use the value
written directly by the script.

The learned and zero-residual rollouts also follow different state trajectories
after the first predicted acceleration. Their tensor shapes and control flow
remain comparable, but they are not mathematically identical workloads with
only one instruction removed.

## Eager and compiled execution

Every component is first measured using ordinary eager PyTorch. The script then
attempts to compile the image, history, learned-rollout, and control-rollout
modules with:

```python
torch.compile(module, mode="reduce-overhead", fullgraph=True)
```

Compilation is triggered before the compiled warm-up and measured iterations.
Compilation time is never included in latency results. The compiled result is
intended to estimate an optimized fixed-shape deployment in which Python and
CUDA kernel-launch overhead can be reduced.

`torch.compile` requires a compatible PyTorch Inductor and Triton environment.
If compilation is unavailable, the script preserves the eager measurements and
writes a structured `compiled` entry containing `available: false` and the
error. This is expected on some Windows installations and should not invalidate
the eager benchmark.

Eager and compiled values must be clearly labeled. A paper should not silently
select the faster value without stating which execution mode was used.

## Timing method

Each component receives its own warm-up followed by repeated CUDA-event timing.
CUDA events measure elapsed GPU-stream time rather than Python wall-clock time.
The script does not synchronize between individual future steps. It records an
event around the complete component and synchronizes after the set of measured
iterations.

For every component it reports:

- mean latency;
- median latency;
- standard deviation;
- 5th percentile;
- 95th percentile.

Median latency is used for derived values because it is less sensitive than the
mean to occasional GPU scheduling and clocking outliers. The full distribution
should still be retained when comparing close results.

## Interpreting image-size results

ResNet-18 accepts arbitrary spatial dimensions. A 64x64 input is not resized or
padded to 224x224 by this benchmark. Adaptive average pooling produces the
fixed-size final feature vector directly from each backbone output.

At batch size 1, a smaller image is not guaranteed to have a measurably lower
latency in every run. Kernel-launch overhead, GPU occupancy, convolution kernel
selection, clock ramping, thermal state, and run order can dominate small
differences. Very close latency distributions should be reported as similar,
not ranked based on a single median.

For a strong image-size comparison:

1. Run the entire script several times in fresh processes.
2. Reverse or randomize image-size order between process runs.
3. Report medians across process-level repetitions.
4. Include percentile ranges or another uncertainty measure.
5. Keep the GPU model, power mode, software versions, and batch size fixed.

The current script uses random untrained models constructed separately for each
size. Weight values do not affect the operation count, but separate process
runs remain useful for capturing system-level variation.

## What the benchmark can support

The benchmark supports claims about post-filter computational overhead, such
as:

- how much time one image encoding adds to a forecast;
- how much time the learned autoregressive residual calculation adds beyond a
  simple constant-velocity state rollout;
- how input crop size affects ResNet feature-extraction latency;
- how eager and compiled execution differ for fixed shapes;
- whether image encoding or residual rollout dominates learned overhead.

It does not by itself support claims about:

- end-to-end application latency;
- image loading, event rendering, or crop-generation cost;
- CPU/GPU transfer cost;
- Kalman history-filtering latency;
- energy use or memory consumption;
- forecast accuracy;
- throughput for batch sizes other than the hard-coded batch size;
- latency under dynamically varying shapes.

If the application-level question matters, report a separate end-to-end
benchmark alongside this focused result rather than adding unrelated work to
the incremental-overhead number.

## Presentation at several levels of detail

The same result can be communicated at different levels depending on the
audience and available space. In every form, identify the device, image size,
batch size, forecast horizon, and eager or compiled mode.

### One-sentence result

Use this in an abstract, caption, or concise discussion:

> After Kalman history filtering, the image-conditioned model added **T ms** of
> median CUDA latency for an 800 ms forecast: **I ms** for one ResNet-18 image
> encoding, **H ms** for history encoding, and **R ms** of incremental residual
> rollout beyond the matched constant-velocity control.

Replace `T`, `I`, `H`, and `R` with
`derived_median_total_ml_overhead_ms`, `image_encoding.median_ms`,
`history_encoding.median_ms`, and
`derived_median_incremental_rollout_ms` respectively.

### Short paragraph

Use this in the main experimental text:

> We measured only the learned computation after the 12-frame observation
> history had been processed by the Kalman filter. For batch-size-one inference
> on **GPU**, one **W x H** input was encoded once by ResNet-18 and its feature
> reused throughout a 24-step, 800 ms autoregressive rollout. Median image and
> history encoding took **I ms** and **H ms**. Learned rollout took **L ms**,
> compared with **C ms** for a matched zero-residual constant-velocity state
> rollout. We therefore estimate incremental residual processing at **L-C ms**
> and total post-filter learned overhead at **T ms** in **eager/compiled** mode.

### Table presentation

A compact comparison table can use:

| Input | Execution | Image encoding (ms) | History encoding (ms) | Learned rollout (ms) | Zero-residual rollout (ms) | Incremental rollout (ms) | Total ML overhead (ms) |
|---|---|---:|---:|---:|---:|---:|---:|
| 64x64 | eager | ... | ... | ... | ... | ... | ... |
| 224x224 | eager | ... | ... | ... | ... | ... | ... |
| 64x64 | compiled | ... | ... | ... | ... | ... | ... |
| 224x224 | compiled | ... | ... | ... | ... | ... | ... |

Use median values in the main cells. Add `p05-p95` in parentheses, error bars,
or a supplementary table if space permits. If compilation is unavailable,
write `N/A` and identify the missing compiler dependency in a note rather than
dropping the row without explanation.

### Figure presentation

A stacked bar chart can show image encoding, history encoding, and incremental
residual rollout as the three contributions to total learned overhead. Place
the zero-residual rollout beside it as a separate reference bar; do not include
the full control rollout inside the learned-overhead stack because it has
already been subtracted from the incremental residual term.

Use separate panels or visually distinct groups for eager and compiled results.
Include percentile whiskers only on directly measured components. Derived
differences require paired or repeated-run uncertainty estimation before they
can receive statistically meaningful error bars.

### Full methodology description

Use language similar to the following in a paper or technical report:

> We isolated the incremental inference cost of learned residual forecasting
> from historical Kalman filtering. Twelve observations sampled at 30 Hz were
> filtered before timing, and the resulting position/velocity state initialized
> two CUDA rollouts over 24 future steps (800 ms). The learned path encoded one
> image with ResNet-18, encoded the relative box history, and reused both feature
> vectors while applying an autoregressive residual-acceleration MLP at every
> future step. A matched control propagated the same initial state with constant
> velocity without covariance propagation or residual evaluation. Dataset I/O,
> crop construction, host-to-device transfers, model initialization, Kalman
> history filtering, compilation, and warm-up were excluded. Components were
> measured independently with CUDA events over **N** iterations at batch size
> one. We report distribution statistics for all directly measured components
> and estimate incremental residual cost as the difference between median
> learned and control rollout latencies.

This detailed version makes clear that the control is not a complete Kalman
forecast and that the result measures additional learned computation rather
than total system latency.

## Reporting checklist

Before publishing or comparing results, record:

- GPU model and power/performance mode;
- PyTorch, CUDA, cuDNN, Triton, and driver versions;
- eager or compiled execution;
- input width and height;
- batch size;
- history and future step counts;
- temporal step size and resulting forecast horizon;
- warm-up and measured iteration counts;
- whether TF32 or mixed precision was enabled;
- medians and dispersion for directly measured components;
- number of independent process-level repetitions;
- any change to model architecture or benchmark constants.

The script currently enables high float32 matrix-multiplication precision via
`torch.set_float32_matmul_precision("high")`. If numerical precision settings
are changed, report them because they can affect both latency and hardware
utilization.

## Recommended interpretation

Use this benchmark as an algorithmic decomposition of learned overhead. Keep a
separate end-to-end benchmark for operational latency. Together they answer two
different and complementary questions:

1. **Focused overhead:** What does learned image-conditioned correction add
   after filtering is complete?
2. **Deployment latency:** How long does the complete application take on the
   devices chosen for actual deployment?

Avoid dividing the focused CUDA overhead by an unrelated CPU Kalman latency or
describing the derived overhead as a complete system speedup. The focused
result is strongest when presented as a transparent sum of measured model
components with the zero-residual rollout serving as an explicit control.
