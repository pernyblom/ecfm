# Relative pose inference with a calibrated static camera

Use a trained FRED checkpoint to estimate a UAV's position, up vector, and motion relative to a fixed event camera. Supply the measured intrinsics and camera orientation directly at inference. You can run this setup without retraining or adding the scene to the FRED scene table.

## Prepare the inputs

1. Use a detector or tracker to get the UAV box at the anchor timestamp `t0`. Express it as `(cx, cy, w, h)` in full-frame event-camera pixels. The anchor-only checkpoint takes that single box. The box-history checkpoint takes 12 boxes for the same target at roughly 30 Hz, ordered oldest to newest and ending at `t0`. Use the actual box timestamps.
2. Gather events from the 100 ms immediately before `t0` on the same clock as the boxes. The renderer counts events in six bins per polarity, applies the checkpoint's normalization, and produces a 640 × 360 representation. It then takes a 64 × 64 crop centered on the anchor box. The resulting input has shape `[1, 12, 64, 64]`.
3. Express the events, boxes, and intrinsics in the same pinhole image coordinates. The example uses the checkpoint's 1280 × 720 frame. If you rectify or remap your input, use the intrinsics for the resulting full frame. Keep `K` in those full-frame pixel units when you crop the event tensor. If the boxes come from a separate RGB camera, register them to the event-camera image first.

The model uses camera axes **right, down, forward**. Let `R_wc` be the camera-to-world rotation expressed in that convention. In a Z-up world, `R_wc.T @ [0, 0, 1]` gives world up in camera coordinates. If your calibration gives a world-to-camera rotation, multiply world up by that rotation directly. The camera's world translation is only needed to express the inferred position in world coordinates. [Pinhole camera and extrinsic conventions](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html)

## Run the checkpoint

Run this example from the repository root in the experiment's Python environment. Set `variant` to `"with_history"` to use the 12-box checkpoint. The public `estimate_state()` method uses a FRED scene ID. The example calls `_estimate_state()` with the measured camera parameters directly.

```python
import numpy as np
import torch

from ecfm.utils.event_representations import render_event_tensor
from experiments.fred_relative_pose.model import RelativePoseModel, project_body
from experiments.kalman_ml_forecasting.data.track_dataset import _prepare_event_tensor

variant = "anchor_only"
checkpoint = torch.load(
    f"outputs/fred_relative_pose/canonical_crop64_events100/{variant}/best.pt",
    map_location="cpu",
    weights_only=True,
)
config = checkpoint["config"]
data = config["data"]
model = RelativePoseModel(config, checkpoint["scenes"]["up_priors"])
model.load_state_dict(checkpoint["model"])
model.eval()


@torch.inference_mode()
def infer(events, boxes_px, box_times_us, K, R_wc):
    # events: [N, 4] as x, y, timestamp_us, polarity (0 or 1).
    # boxes_px: [1, 4] for anchor_only or [12, 4] for with_history.
    width, height = data["frame_size"]
    box_times_us = np.asarray(box_times_us, dtype=np.int64)
    t0 = int(box_times_us[-1])
    past = np.asarray(boxes_px, dtype=np.float32) / np.array(
        [width, height, width, height], dtype=np.float32
    )

    rep = "event_stack"
    full = render_event_tensor(
        events, rep, width=width, height=height,
        start_us=t0 - 100_000, end_us=t0, output_size=(640, 360),
        options=data["representation_options"][rep],
    )
    crop = _prepare_event_tensor(
        full, (64, 64), source_size=(640, 360), rep=rep,
        frame_size=(width, height), anchor_box=past[-1],
        spatial_cutout=data["spatial_cutout"],
    )
    batch = {
        "past": torch.from_numpy(past)[None],
        "past_time": torch.as_tensor(
            (box_times_us - t0) * 1e-6, dtype=torch.float32
        )[None],
        "events": {rep: crop[None]},
    }
    camera = {
        "focal": torch.tensor([[K[0, 0], K[1, 1]]], dtype=torch.float32),
        "center": torch.tensor([[K[0, 2], K[1, 2]]], dtype=torch.float32),
        "gravity_up": torch.as_tensor(
            R_wc.T @ np.array([0., 0., 1.]), dtype=torch.float32
        )[None],
    }

    state, _ = model._estimate_state(batch, camera, mask_events=False)
    times_s = state.position.new_tensor([[0., 0.1, 0.2, 0.4]])
    trajectory = model.propagate(state, times_s, camera["gravity_up"])
    projected_boxes_px = project_body(
        trajectory.position, trajectory.up, camera, model.half_height
    )
    return state, trajectory, projected_boxes_px
```

Call `infer(events, boxes_px, box_times_us, K, R_wc)` with your synchronized sensor data and calibration. The code needs only data up to the anchor. The forecast queries are elapsed times in seconds. The first projected box is at the anchor, and the other three are 100, 200, and 400 ms ahead.

## Interpret and check the result

| Output | Meaning |
| --- | --- |
| `state.position` | Position relative to the optical center in assumed body diameters |
| `state.up` | Unit vector along the UAV's up axis in camera coordinates |
| `state.velocity` | Linear velocity in assumed body diameters per second |
| `state.angular_velocity` | Tilt angular velocity in radians per second |
| `state.thrust` | Thrust per unit mass in assumed body diameters per second squared |
| `trajectory.acceleration[:, 0]` | Anchor acceleration in assumed body diameters per second squared |
| `projected_boxes_px` | Anchor and forecast boxes as full-frame pixel `(cx, cy, w, h)` |

To express the inferred up vector in world coordinates, use `R_wc @ up_cam`. If `C_w` is the camera center in world meters and `d_m` is the body-diameter scale in meters, use `C_w + R_wc @ (d_m * position_cam)` for approximate world position.

Known calibration fixes the viewing geometry. The range estimate still depends on the inferred state and assumed UAV size. The FRED evaluation measured forecast accuracy in the image plane. For a new scene, overlay the projected boxes on the tracker output and compare range and up direction separately with independent measurements.

See the [experiment README](README.md#use-the-flight-state) for the state model and training details.
