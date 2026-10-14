# Event image representations

`ecfm.utils.event_representations.render_event_tensor` produces float32 tensors
shaped `[C, H, W]`. `C` is determined by the representation settings and can be
any positive channel count. Indexed forecasting and `FREDDataset` use this API
directly, without converting event tensors through PIL or an RGB preview.

The separate `render_event_representation` API renders an
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

The table describes the legacy RGB image API. Its three-channel choices preserve compatibility with the project's CNN/ResNet
encoders and PNG storage. Voxel-grid byte 128 represents zero activity, byte 1
negative saturation, and byte 255 positive saturation. Empty voxel grids are
neutral gray; the other temporal formats are black when empty. Time-surface
values use the actual window end, including when the last event precedes it;
they do not depend on input ordering. Stacks ignore polarity, while voxel grids
allow opposite polarities to cancel. The voxel-grid temporal coordinate spans
the configured window rather than stretching the first and last observed event.

The float tensor API uses the following settings. Pass a single representation's
settings as `options` to `render_event_tensor`, or use
`data.representation_options` in forecasting YAML. Exact representation names
take precedence over base-name settings for grid aliases.

| Representation | Tensor settings | Channel count |
| --- | --- | --- |
| `time_surface`, `time_surface_linear` | `include_activity: true` (default); exponential surfaces also accept `tau_ms` | 3 (positive/all/negative), or 2 (positive/negative) when activity is omitted |
| `multi_channel_time_surface` | `decay_ms: [5, 10, 20, 40]`, optional `split_polarity: true` | Number of decay constants, doubled when polarity is split |
| `event_stack` | `bins: 5`, optional `split_polarity: true` | Number of bins, doubled when polarity is split |
| `voxel_grid` | `bins: 5`, optional `split_polarity: true` | Number of bins, doubled when polarity is split |

Defaults are three bins, unsplit polarity, and three decay constants at
`tau/4`, `tau/2`, `tau`. Split tensors are **polarity-major**: all positive
bins or decay scales first, then all negative bins or decay scales. Temporal
bins run from early to late; decay scales follow the supplied list order.
Stacks and voxel grids accept `normalization: max` (default, joint absolute
maximum per spatial cell) or `normalization: none` to retain counts. Unsplit
voxel grids are signed floats; zero activity is exactly zero. Split voxel
channels accumulate each polarity separately and are nonnegative. Resizing
preserves signs, full float precision, and the channel count. Other existing
formats retain their RGB-compatible encoding in the tensor API.

```python
tensor = render_event_tensor(
    events, "voxel_grid", width=1280, height=720,
    start_us=100_000, end_us=133_333, output_size=(640, 360),
    options={"bins": 5, "split_polarity": True},
)  # float32 [10, 360, 640]
```

Float event tensors can be saved as `.npy` arrays or `.npz` archives with a
`tensor` key. Forecasting looks for `<label_stem>_<representation>.npz`, then
`.npy`, then the legacy `.png`. Tensor files contain `[C,H,W]` floating point
arrays; they are not normalized or quantized when loaded. The normal configured
resizing, cutouts, and count normalization still apply. For custom stored
tensors whose channel counts cannot be inferred from renderer settings, set
`data.representation_channels: {my_representation: 12}`. Encoder channel counts
are resolved before loading data, and mismatches fail with an explicit error.

Use the training YAML to export label-aligned tensors and separate PNG previews:

```bash
python scripts/render_fred_tensors.py --config path/to/config.yaml --sequence 8 --output-dir outputs/fred_tensors --max-frames 100
```

Omit `--sequence` to use `data.split_files.train`, or choose another `--split`.
The exporter writes render manifests compatible with forecasting validation.
It refuses to overwrite an existing manifest; use a new output directory for a
new set of settings. Previews have a `_preview.png` suffix so they cannot be
mistaken for training inputs. Set `event_image_source: rendered` and point
`images_root` at the export directory to train from the stored tensors.

## Visualization

`visualization.event_images` controls the tensor exporter, forecasting cutout
inspector, and track visualization. These settings never affect model inputs,
encoder channels, or sample-cache keys.

The default `mode: auto` preserves ordinary unsigned RGB, shows two channels
or polarity-split tensors in positive red/negative blue, and groups larger
unsigned tensors into three equally sized channel groups whose means become
R/G/B. Signed voxel tensors display positive activity in red and negative
activity in blue. Compact previews keep the source spatial dimensions, allowing
box overlays. `grid` displays every selected channel in a labeled grayscale
panel; signed grid channels map `[-1,1]` to black through gray to white.

```yaml
visualization:
  event_images:
    mode: auto
    normalization: auto
    by_representation:
      event_stack:
        mode: grid
        columns: 5
        labels: true
      voxel_grid:
        mode: rgb
        channels: [0, 2, 4]
      multi_channel_time_surface:
        mode: temporal_rgb
```

Modes are `auto`, `rgb` (exactly three selected channels), `temporal_rgb`
(channel-group means), `polarity`, `mean`, `max`, and `grid`. Optional `channels`
is a list of zero-based indices, applied before reduction in the supplied order;
omit it to include every channel. `polarity` pools positive/negative parts of
signed tensors, or the first/second halves of an unsigned selected tensor.
`normalization` accepts `auto`, `none`, `max`, `min_max`, or
`per_channel_min_max`. Auto retains bounded values and scales larger magnitudes
by their absolute maximum. Signed values use a centered mapping in RGB and
grayscale modes; explicit min/max modes rescale them into unsigned intensities.
Exact names in `by_representation` override base names for grid aliases.

Grid previews have panel dimensions rather than sensor dimensions. Use them
for inspection or standalone exports. The cutout inspector requires a spatial
preview for `--draw-anchor-box`; track composition retains grid dimensions and
reports incompatible layer sizes rather than resizing panels into sensor space.

SmallCNN and ResNet-18 branches select their input channels independently in
forecasting and object-detection model factories. Only their input convolution
depends on the channel count; feature fusion and prediction heads are unchanged.
Existing three-channel configurations keep their channel counts. Changing bins
or polarity splitting changes first-convolution weight shapes, so checkpoints
trained with a different channel configuration cannot be loaded unchanged.
Indexed voxel inputs now use signed float values rather than the earlier RGB
byte mapping; retrain an earlier voxel model when switching to this tensor path.

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
