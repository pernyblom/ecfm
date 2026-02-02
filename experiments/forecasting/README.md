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
     datasets/FRED/0/Event/events.raw `
     datasets/FRED/0/Event_YOLO `
     outputs/fred_reps `
     --representation "xt;yt;xy"

2) Train:
   python experiments/forecasting/train.py --config experiments/forecasting/configs/base.yaml

3) Eval:
   python experiments/forecasting/eval.py --config experiments/forecasting/configs/base.yaml

Notes
- This baseline assumes one primary drone per frame and selects the largest box.
- It is intentionally simple for fast iteration. Add track IDs or multi-object
  handling when ready.

Tracks-based training (tracks.txt)
Use this when you want track-consistent supervision instead of YOLO boxes.

1) Render images for the split (all folders):
   python scripts/render_fred_splits.py `
     --split-file datasets/FRED/dataset_splits/canonical/train_split.txt `
     --output-root outputs/fred_reps `
     --representation "xt;yt"

   python scripts/render_fred_splits.py `
     --split-file datasets/FRED/dataset_splits/canonical/test_split.txt `
     --output-root outputs/fred_reps `
     --representation "xt;yt"

2) Train using tracks:
   python experiments/forecasting/train.py --config experiments/forecasting/configs/tracks.yaml

3) Notes on alignment:
   - Track timestamps are assumed to be seconds.
   - Label/frame timestamps come from Event_YOLO filenames and are treated as microseconds.
   - The two time sources are not guaranteed to share the same zero-time. To make them
     comparable, we shift each track timeline so that the first timestamp in that track
     matches the first label timestamp in the folder:
       aligned_t = track_t + (first_label_t - first_track_t)
     This is a simple per-track offset that puts tracks and frame times on the same axis.
   - We then interpolate each track box to the exact label/frame timestamps. This is
     needed because track samples and label frames rarely land on identical times.
     Interpolation ensures every frame in a training window has a box.
   - If you have known sensor geometry, set `data.frame_size` in `tracks.yaml` to avoid
     relying on inferred sizes from tracks.
