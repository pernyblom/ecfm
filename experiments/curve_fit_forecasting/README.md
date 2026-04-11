Curve-Fit Forecasting

This experiment forecasts drone boxes from a single anchor pair of event-history
images (`xt_my`, `yt_mx`) plus box history for the same track.

What it does
- Builds track-aligned samples from `cleaned_tracks.txt` and `Event_YOLO` frame times.
- Uses one anchor image at forecasting time `t` and includes box history up to `t`.
- Fits guided curves in `XT` and `YT`, then predicts future centers in `XY`.
- Uses a simple box-size heuristic for now (`mean`, configurable).
- Falls back to a history-only curve when the image-based fit disagrees too much
  with the observed history boxes.

Why the dataset shape differs from `experiments/forecasting`
- The existing forecasting dataset is centered on windows of images.
- This method needs the anchor `xt_my` / `yt_mx` image at time `t` plus a
  history trajectory to guide curve extraction inside that image.
- That matches the forecasting definition in `docs/fred_paper/main.tex`: past
  information up to `t`, future predictions after `t`.

Defaults
- `data.image_window_ms = 400`: assumes the rendered event-history image covers
  400 ms by default.
- `data.verify_render_manifest = true`: requires `render_manifest.json` next to
  rendered images and validates window mode and duration for each anchor frame.
- `data.history_ms = 400`: uses 400 ms of history boxes ending at the anchor.
- `data.forecast_ms = 400`: predicts 400 ms ahead by default.
- Set `data.forecast_ms = 800` for the mid-term benchmark.

Run
```bash
python experiments/curve_fit_forecasting/eval.py --config experiments/curve_fit_forecasting/configs/base.yaml
```

Notes
- `scripts/render_fred_splits.py` now writes per-folder `render_manifest.json`
  files and a split-level aggregate manifest under the output root. Re-render if
  your existing image directories do not have these manifests yet.
- `data.image_window_mode` should stay `trailing` for forecasting. Other modes
  can leak future information into the event image.
- The curve-fit settings are all exposed under `curve_fit` so thresholds and
  guidance weights can be tuned later, or learned by a downstream model.
- Size prediction is intentionally simple for now. The next logical extension is
  a learned size head conditioned on history and the event images.
