# Event Camera Foundation Model (ECFM)

This repo scaffolds a PyTorch project for a transformer-based foundation model
over event camera data. See `DESIGN.md` for the design draft.

## Layout
- `src/ecfm`: library code
- `configs`: YAML configs
- `scripts`: training and utility entrypoints
- `data`: placeholder for datasets
- `tests`: basic unit tests
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

## CUDA Setup (Windows)
If your default Python is 3.14, install a CUDA-enabled PyTorch build in a
Python 3.12 venv (CUDA wheels are not available for 3.14).

```powershell
py -3.12 -m venv .venv312
. .\.venv312\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install --upgrade --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -e .
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## THU-EACT-50-CHL Smoke Run
If the dataset is placed at `datasets/THU-EACT-50-CHL`, run:
```powershell
python scripts\train.py --config configs\thu_smoke.yaml
```

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
launcher `command` using `{config}`.

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
