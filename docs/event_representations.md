# Event image representations

`ecfm.utils.event_representations.render_event_representation` renders an
`[N, 4]` array of `(x, y, timestamp_us, polarity)` to an RGB uint8 array.
Polarity `1` is positive; `0` and `-1` are negative. All inputs are restricted
to the half-open window `[start_us, end_us)` and sensor bounds. Output sizes
are `(width, height)`. Integer timestamp differences are computed before float
conversion. No events or decoded files are written to disk.

The existing formats include XY/XT/YT polarity histograms, rotated projections,
mean-coordinate projections (`xt_my`, `yt_mx`), and mean-timestamp/count images
(`cstr2`, `cstr3`, `cstr3_fixed`). The following temporal formats also work in
FREDDataset, the FRED batch renderer, the label-free EVT3 renderer, and indexed
Kalman residual training:

| Name | Red, green, blue channels | Scaling |
| --- | --- | --- |
| `time_surface` | Latest positive, latest of either polarity, latest negative | `exp(-(end_us - latest_us) / tau_us)`; unseen pixels are zero |
| `time_surface_linear` | Latest positive, latest of either polarity, latest negative | `(latest_us - start_us) / window_us`; unseen pixels are zero |
| `multi_channel_time_surface` | Latest event of either polarity at three decay scales | Exponential surfaces with decay constants `tau/4`, `tau/2`, `tau` |
| `event_stack` | Event counts in the early, middle, late thirds of the window | Divided by the maximum count across all three channels |
| `voxel_grid` | Signed event accumulations at three temporal bin centers | Linear interpolation in time; joint absolute maximum scaling, then `round(128 + 127 * value)` |

The three-channel choices preserve compatibility with the project's CNN/ResNet
encoders and PNG storage. Voxel-grid byte 128 represents zero activity, byte 1
negative saturation, and byte 255 positive saturation. Empty voxel grids are
neutral gray; the other temporal formats are black when empty. Time-surface
values use the actual window end, including when the last event precedes it;
they do not depend on input ordering. Stacks ignore polarity, while voxel grids
allow opposite polarities to cancel. The voxel-grid temporal coordinate spans
the configured window rather than stretching the first and last observed event.

The default exponential decay constant is one third of the window duration.
Set `time_surface_tau_us` in the shared renderer, or `time_surface_tau_ms` in
FREDDataset and forecasting data configuration, to override it. The batch
renderers use the default decay constant.

Every new format supports grid aliases, for example
`multi_channel_time_surface_4x3`. Each spatial cell uses the same time window;
Counts and voxel grids are normalized per cell. Use `output_size` to set the final canvas size.

```python
from ecfm.utils.event_representations import render_event_representation

image = render_event_representation(
    events, "multi_channel_time_surface",
    width=1280, height=720, start_us=100_000, end_us=133_333,
    output_size=(640, 360), time_surface_tau_us=10_000,
)
```

For background on polarity surfaces and temporally interpolated voxel grids,
see [Gehrig et al., ICCV 2019](https://openaccess.thecvf.com/content_ICCV_2019/html/Gehrig_End-to-End_Learning_of_Representations_for_Asynchronous_Event-Based_Data_ICCV_2019_paper.html).
