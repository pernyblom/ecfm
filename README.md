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

## THU-EACT-50-CHL Smoke Run
If the dataset is placed at `datasets/THU-EACT-50-CHL`, run:
```powershell
python scripts\train.py --config configs\thu_smoke.yaml
```
