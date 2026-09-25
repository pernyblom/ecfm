# Event Camera Foundation Model (ECFM)

This repo contains a PyTorch project for learning representations from event
camera streams. The main pretraining path is a masked region-token autoencoder
with learned relative attention bias. See `DESIGN.md` for architecture details.

## Region MAE Path

An event stream is sampled into spatio-temporal regions. Each region becomes a
two-channel histogram patch plus normalized `(x, y, dx, dy, t, dt)` metadata
and a projection-plane ID. During pretraining, selected patch contents are
replaced by a learned mask token. Region metadata and plane identity remain
visible so the model knows which region it is reconstructing.

With `use_relative_bias: true`, every encoder attention head receives a learned
pairwise bias based on relative position and log scale ratios. Absolute token
position embeddings are normally disabled for this path because token order is
not spatial order. Padded regions are excluded from encoder and decoder
attention through each sample's `valid_mask`.

The implementation is MAE-style but intentionally differs from the original
visible-tokens-only MAE architecture: masked region tokens remain in the
encoder after their patch content has been hidden.

## Layout

- `src/ecfm`: library code
- `configs`: YAML configs
- `scripts`: training and utility entrypoints
- `data`: placeholder for datasets
- `tests`: unit and training-path regression tests
- `docs`: extra docs

## Quick Start

1) Create a venv and install dependencies.
2) Run a dry training loop with synthetic events.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .
python scripts\train.py --config configs\small.yaml
```

In the current trainer, `train.num_steps` is the number of complete passes over
the data loader. Console `step` values and reconstruction `step_*` directories
therefore correspond to zero-based epochs.

## CUDA Setup (Windows)

The following setup is tested with both ordinary CUDA execution and
`torch.compile` in the Kalman incremental-overhead benchmark. Use the versions
as a matched set: PyTorch 2.6 uses Triton 3.2. In particular, PyTorch 2.5 has a
known Windows TorchInductor cache-renaming failure even when Triton itself is
installed correctly.

Prerequisites:

- 64-bit Python 3.12, available through the Windows `py` launcher;
- an NVIDIA CUDA-capable GPU;
- a current NVIDIA driver (`nvidia-smi` should run successfully);
- PowerShell opened in the repository root.

The PyTorch and Triton wheels include the CUDA components needed by this
setup. A separate CUDA Toolkit or Visual Studio C++ installation is not
required for this GPU-only benchmark.

```powershell
py -3.12 -m venv .venv312
. .\.venv312\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install --index-url https://download.pytorch.org/whl/cu124 "torch==2.6.0+cu124" "torchvision==0.21.0+cu124"
python -m pip install "triton-windows>=3.2,<3.3"
python -m pip install -e .
python -m pip check
python -c "import torch, torchvision, triton; print('torch:', torch.__version__); print('torchvision:', torchvision.__version__); print('triton:', triton.__version__); print('CUDA runtime:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
```

The expected core versions are `torch 2.6.0+cu124`,
`torchvision 0.21.0+cu124`, and `triton 3.2.x`. The CUDA version reported by
`nvidia-smi` describes the newest runtime supported by the driver and does not
need to equal the `12.4` runtime bundled with PyTorch.

If PowerShell blocks activation scripts, either adjust the execution policy
for the current process or invoke the environment's interpreter explicitly:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
. .\.venv312\Scripts\Activate.ps1

# Equivalent form without activation:
.\.venv312\Scripts\python.exe -m pip check
```

These pins intentionally favor a known-working, reproducible Windows compile
environment over automatically selecting the newest releases. When upgrading
PyTorch later, upgrade Triton to the matching minor version as documented by
the `triton-windows` project and retest `torch.compile`.

## THU-EACT-50-CHL Smoke Run

If the dataset is placed at `datasets/THU-EACT-50-CHL`, run:
```powershell
python scripts\train.py --config configs\thu_smoke.yaml
```

Both `configs/thu_smoke.yaml` and `configs/thu_pretrain_small.yaml` exercise the
relative-attention-bias path. For DVS-Lip, use:

```powershell
python scripts\train.py --config configs\dvslip_pretrain_small.yaml
```

For a bounded THU learning check with random `xt`/`yt` regions, variable token
counts, and reconstruction images after every epoch, run:

```powershell
python scripts\train.py --config configs\thu_pretrain_relative_bias_recon.yaml
```

The run writes a fixed example and mask to
`outputs/thu_relative_bias_recon/recon/step_*/`. Each masked token has a
`*_gt.png` and `*_pred.png` image, making reconstruction changes comparable
across epochs. Periodic checkpoints are written under
`outputs/thu_relative_bias_recon/checkpoints/`.

## Region Sensing World Model

The separate [region world-model experiment](experiments/region_worldmodel/README.md)
learns action-conditioned latent changes by rotating or resizing sensing regions
and extracting new patches from the same THU event volume. It includes configurable
axes and magnitudes, masked region-token encoding, latent anti-collapse training,
THU linear probing and finetuning, and held-out-action evaluation.

## Tests

Run the complete suite from the repository root:

```powershell
python -m pytest -q
```

The region-MAE tests cover relative-bias shapes and gradients, token-order
equivariance, masked forward/backward execution, variable region counts, and
padding isolation. Region sampling tests cover spatial and temporal boundaries.

## Image Folder Media
Convert a name-sorted slice of an image folder to MP4 or GIF:

```powershell
python scripts\images_to_media.py --image-dir outputs\frames --output outputs\clip.mp4 --start-index 100 --frame-count 300 --fps 30
python scripts\images_to_media.py --image-dir outputs\frames --output outputs\clip.gif --start-index 100 --frame-count 90 --fps 12
```

## Config Sweeps
Generate expanded YAML configs, a manifest, and `run_all.ps1`/`run_all.sh`
from a base config plus a grid spec:

```powershell
python scripts\generate_config_sweep.py --base-config experiments\kalman_ml_forecasting\configs\base.yaml --spec experiments\kalman_ml_forecasting\configs\sweep_example.yaml --output-dir outputs\kalman_ml_sweeps\example
```

The spec supports `grid`, static `overrides`, output path `templates`, and a
launcher `command` using `{config}`, `{config_path}`, `{name}`, `{output_dir}`,
and `{result_json}` (an automatically quoted `results/<name>.json` path).
Generated sweep configs include
`train.log_file: "{output_dir}/runs/{name}/train.log"` by default unless the
spec overrides that template.

The Kalman forecasting experiment includes
`configs/kalman_motion_model_sweep.yaml` to compare and backprop-optimize
constant-velocity and constant-acceleration filters while leaving constant
velocity as the base-config default.

After runs finish, collect configs and results into table-ready records:

```powershell
python scripts\sweep_results.py --sweep-dir outputs\kalman_ml_sweeps\example --output-csv outputs\kalman_ml_sweeps\example\results.csv
```

From Python:

```python
from pathlib import Path
from scripts.sweep_results import sweep_dataframe, load_sweep_runs

df = sweep_dataframe(sweep_dir=Path("outputs/kalman_ml_sweeps/example"))
print(df[["name", "test.loss", "test.fde_center_px", "config.data.representations"]])

runs = load_sweep_runs(sweep_dir=Path("outputs/kalman_ml_sweeps/example"))
print(runs[0].config)
print(runs[0].test_results)
```
