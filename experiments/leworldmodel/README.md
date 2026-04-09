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
- SSL image steps: `1 -> 1`
- SSL future offset: `8`
- Forecast boxes: `12 -> 24`
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
- `data.ssl_context_steps` and `data.ssl_future_steps` are only for SSL on image
  tuples. With the default `1 -> 1`, one rendered image tuple predicts the next
  rendered image tuple.
- `data.ssl_future_offset_steps` inserts a gap between the last SSL context
  image and the first SSL target image. With the default value `8`, the first
  SSL future target is 8 frame steps after the anchor frame.
- `data.forecast_history_steps` and `data.forecast_future_steps` are only for
  the box forecasting task.
- `data.max_frame_gap_s` drops windows with large timestamp gaps.
- `model.encoder_type` supports `small_cnn`, `resnet18`, and `vit_tiny`.
- `model.predictor_type` supports `mlp` and `transformer`.
- `regularizer.type` supports:
  - `simple`: mean/std matching toward a standard normal target
  - `vicreg`: variance + covariance anti-collapse regularization
  - `sigreg`: sketched Gaussian matching using the Epps-Pulley statistic over
    random 1D projections
- `downstream.forecasting.enabled` toggles the inline box-forecasting head.
- `downstream.forecasting.use_ssl_features` controls whether the forecasting
  head uses SSL latent context.
- `downstream.forecasting.use_history_boxes` controls whether the forecasting
  head uses past box history.
- At least one of those two must be `true`.

About the rendered event window
- The `--window` value used in `render_fred_splits.py` is fixed at render time.
  The LeWorldModel code does not reinterpret that duration later.
- If you rendered with `8 * 33333`, each saved image already contains a longer
  temporal integration. Training then treats that saved image as one timestep.
- FRED frame timestamps are exactly `33333` microseconds apart in the label
  files, so the implicit frame step is exactly `33.333 ms`.
- For the setup you described, `ssl_context_steps: 1`,
  `ssl_future_steps: 1`, and `ssl_future_offset_steps: 8` means one integrated
  event image tuple predicts a target image tuple whose timestamp is
  `8 * 33.333 ms = 266.664 ms` later.
- With trailing windows rendered at `8 * 33333 us`, that means the future
  target window starts right where the context window ends, up to the 1 us
  rounding already present in the source timestamps.

Standalone downstream forecasting
- You can train forecasting from an SSL checkpoint without running SSL and
  forecasting jointly:
```powershell
python experiments/leworldmodel/downstream_forecasting.py --config experiments/leworldmodel/configs/base.yaml --checkpoint outputs/leworldmodel_ckpt/best.pt
```
- By default this freezes the SSL backbone and only trains `forecast_head`.
  Change `downstream.forecasting.freeze_ssl_backbone` if you want to finetune
  the full model instead.
- Useful ablations:
  - boxes only: `use_ssl_features: false`, `use_history_boxes: true`
  - latents only: `use_ssl_features: true`, `use_history_boxes: false`
  - fused: `use_ssl_features: true`, `use_history_boxes: true`

Recommendation
- Start with `sigreg` for the main LeWorldModel-style run.
- Use `simple` only as a cheap sanity baseline.
- Use `vicreg` when you want a stronger but easier-to-interpret non-SIGReg
  anti-collapse baseline.
