# Indexed FRED rendering and image comparison

The shared [IndexedRawReader](../../src/ecfm/utils/evt3_index.py) now reads bounded
event windows directly from RAW using an existing OpenEB v2 `.raw.tmp_index`.
No temporary decoded event file is produced. The frame and split renderers expose
this as `--event-source indexed`; the existing `auto/raw/npz` choices retain their
previous meaning. Indexed image rendering currently requires little-endian RAW
and `event_unit=1` (microseconds).

## Comparison with existing images

Run from the repository root:

```powershell
.venv312/Scripts/python.exe scripts/compare_fred_indexed_frames.py
```

Default output: `outputs/fred_reps_indexed_33ms/index.html`. The gallery links full
PNGs and shows existing/indexed/amplified-difference columns, including a zoom
around the largest labeled UAV. JPEG contact sheets are previews only; metrics
are computed on decoded original PNG pixels. Parameters come from the reference
manifests, including representation-specific sizes (which differ across folders).

The default seed 42 selects three eligible UAV-containing frames from each of four
randomly ordered sequences. A minimum normalized box area of 0.0003 favors visible
objects. All requested representations must be recorded in the manifest and have
an existing PNG. Orphaned images whose representation is absent from the current
manifest are excluded because their rendering settings are unknown.

On 2026-10-09, sequences **116, 202, 13 and 10** provided 12 frames:

* 33,333 us trailing windows; `cstr3`, `xt_my`, `yt_mx`.
* **36/36 images pixel-identical** to `outputs/fred_reps`.
* **12/12 event counts identical** to the saved manifests.
* Difference mean/max and changed-pixel fraction were all zero.
* Original PNGs and manifests were not modified.

`comparison_results.json` records sample IDs, counts, offsets, bytes read and
pixel metrics. `missing_image_check.json` records a separate live check: remove
one newly generated CSTR3 image, reconstruct it identically, preserve the other
representations, then repeat the request with zero event-window reads.

These results validate the tested render compatibility, not all possible RAW
streams or independent SDK decoder correctness.

## Create missing images on demand

```python
import json
from pathlib import Path
from scripts.render_evt3_yolo_frames import ensure_rendered_frames

reference = json.loads(Path('outputs/fred_reps/13/render_manifest.json').read_text())
manifest = ensure_rendered_frames(
    raw=Path('datasets/FRED/13/Event/events.raw'),
    yolo_dir=Path('datasets/FRED/13/Event_YOLO'),
    output_dir=Path('outputs/fred_reps_on_demand/13'),
    label_stems=['Video_13_frame_25166415'],
    render_params=reference['render_params'],
)
```

The helper creates the requested representations and an ordinary render manifest.
An existing compatible entry is reused only if its actual image files exist. A
missing representation is regenerated; complete frames skip event decoding. It
rejects conflicting source paths or rendering parameters in the same output folder.
An empty list is rejected, preventing accidental whole-sequence rendering.

The command-line renderer also accepts `--label-stems STEM [STEM ...]` to select
specific frames. Omitting this filter processes the sequence's labels as before.
Do not use parallel writers targeting the same output manifest. Training datasets
are not automatically changed to invoke this helper; it is the shared entry point
for adding such an input provider or preparing a batch of missing images.

## Read events without generating images

```python
from ecfm.utils.evt3_index import IndexedRawReader

reader = IndexedRawReader('datasets/FRED/13/Event/events.raw')
events = reader.read_window(25_133_082, 25_166_415)
# int64 [N,4]: x, y, shifted timestamp in microseconds, polarity
```

The default `integer` mode preserves integer microseconds through decoding and
window selection. Subtract the window origin before converting time to a float
tensor for the MAE. Keep the requested duration even for quiet or empty windows.

The existing-image renderer explicitly uses `legacy_float32` mode to reproduce
rounding in the old RAW-to-memmap pipeline. This is necessary for pixel-exact
comparisons with saved images; it does not claim those images retained original
microsecond precision. Boolean window selection promotes those rounded timestamps
to float64 to avoid rounding the window boundary itself. A regression test covers
an upper bound of 49,132,842 us, which cannot be represented exactly in float32.

Index metadata is loaded once per reader. Each read opens an independent RAW
handle, so no cursor is shared across workers. Prefix requests fall back to the
RAW header when no preceding valid bookmark exists; quiet/EOF windows can return
empty arrays. Decoder clock checks account for a chunk ending between TIME_HIGH
and TIME_LOW. A rolling overlap cache and native decoder acceleration remain
possible optimizations; neither is required for these comparisons.

Validation:

```powershell
.venv312/Scripts/python.exe -m pytest tests/test_evt3_index.py tests/test_evt3.py tests/test_render_evt3_yolo_frames.py -q
```

18 tests passed, including indexed/sequential equivalence across timestamp wraps,
unaligned decoder chunks, precision boundaries, reverse-order reads, empty/EOF
windows, corrupt/stale indexes and missing-image recreation without decoding on a
complete cache hit.
