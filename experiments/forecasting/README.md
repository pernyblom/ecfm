Forecasting Experiments (FRED)

This folder is intentionally isolated from the main training code. It contains a
minimal, modular baseline for trajectory forecasting using event
representations (xt, yt, xy, cstr2, cstr3, etc.).

Key ideas
- Use multiple representations per timestep (no bbox crop for xt/yt).
- Per-representation CNNs -> fused temporal model -> future box prediction.
- Train on normalized boxes (cx, cy, w, h in [0,1]).

Expected inputs
- Rendered representation images from `scripts/render_evt3_yolo_frames.py`
  in a single folder, named like: `Video_0_frame_100032333_xt.png`
- YOLO labels in `datasets/FRED/0/Event_YOLO`

Quick start
1) Render images:
   python scripts/render_evt3_yolo_frames.py `
     datasets\\FRED\\0\\Event\\events.raw `
     datasets\\FRED\\0\\Event_YOLO `
     outputs\\fred_reps `
     --representation "xt;yt;xy"

2) Train:
   python experiments\\forecasting\\train.py --config experiments\\forecasting\\configs\\base.yaml

3) Eval:
   python experiments\\forecasting\\eval.py --config experiments\\forecasting\\configs\\base.yaml

Notes
- This baseline assumes one primary drone per frame and selects the largest box.
- It is intentionally simple for fast iteration. Add track IDs or multi-object
  handling when ready.
