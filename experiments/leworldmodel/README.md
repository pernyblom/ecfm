LeWorldModel-Inspired SSL on FRED

This experiment trains a LeWorldModel-style latent predictor on rendered FRED
event-image representations without actions.

What it does
- Uses `cleaned_tracks.txt` plus frame timestamps from `Event_YOLO/*.txt`.
- Builds track-consistent windows for visible drones only.
- Requires rendered representations for every frame in the window.
- Encodes the sequence into latent embeddings and predicts future latents.
- Regularizes the latent space with a configurable anti-collapse term:
  `simple`, `vicreg`, or `sigreg`.
- Optionally trains a lightweight downstream forecasting head that predicts
  future boxes from the latent rollout plus the input box history.

Default starter setup
- Representations: `xt_my`, `yt_mx`, `cstr3`
- Encoder: `small_cnn`
- Predictor: `mlp`
- Context steps: `1`
- Future steps: `12`
- Regularizer: `sigreg`

Render data
1) Train split:
```powershell
python scripts/render_fred_splits.py --split-file datasets/FRED/dataset_splits/canonical/train_split.txt --output-root outputs/fred_reps --representation "xt_my;yt_mx;cstr3" --window 33333 --window-mode trailing --num-workers 4 --include-empty
```

2) Validation split:
```powershell
python scripts/render_fred_splits.py --split-file datasets/FRED/dataset_splits/canonical/test_split.txt --output-root outputs/fred_reps --representation "xt_my;yt_mx;cstr3" --window 33333 --window-mode trailing --num-workers 4 --include-empty
```

Train
```powershell
python experiments/leworldmodel/train.py --config experiments/leworldmodel/configs/base.yaml
```

Config notes
- `data.context_steps` controls how many previous latent frames seed the predictor.
- `data.max_frame_gap_s` drops windows with large timestamp gaps.
- `model.encoder_type` supports `small_cnn`, `resnet18`, and `vit_tiny`.
- `model.predictor_type` supports `mlp` and `transformer`.
- `regularizer.type` supports:
  - `simple`: mean/std matching toward a standard normal target
  - `vicreg`: variance + covariance anti-collapse regularization
  - `sigreg`: sketched Gaussian matching using the Epps-Pulley statistic over
    random 1D projections
- `downstream.forecasting.enabled` toggles the inline box-forecasting head.

Recommendation
- Start with `sigreg` for the main LeWorldModel-style run.
- Use `simple` only as a cheap sanity baseline.
- Use `vicreg` when you want a stronger but easier-to-interpret non-SIGReg
  anti-collapse baseline.
