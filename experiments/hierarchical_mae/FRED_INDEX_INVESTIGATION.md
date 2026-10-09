# Using FRED's existing RAW indexes

Investigation date: 2026-10-09. This follow-up strengthens the recommendation in
[FRED_RAW_FEASIBILITY.md](FRED_RAW_FEASIBILITY.md): **reuse the existing indexes
for random event-window access before building a new checkpoint index.** A bounded
prototype recovered exactly the same event arrays as sequential decoding in all
72 tested windows, without exporting decoded events or images.

## Current use in this repository

`scripts/render_evt3_yolo_frames.py` reads `.raw.tmp_index` for:

* `ts_shift_us`, to align RAW event time with the label timeline;
* approximate first/last indexed timestamps, primarily to detect an already
  shifted `output_events.npz`.

It does **not** use bookmark byte offsets to extract windows. Its RAW path still
calls `decode_evt3_raw_to_arrayfile`, decodes the complete recording to a temporary
float32 file, then uses `searchsorted` on the resulting event timestamps. The MAE
loader does not use the sidecar either.

## What the index contains and how seeking works

Inspected OpenEB commit: `9003b5416676e78ba994d912087486cfa94fae73`.

The existing Linux-generated v2 sidecars contain an ASCII header followed by packed
20-byte records: signed 64-bit timestamp, unsigned 64-bit absolute RAW byte offset,
and unsigned 32-bit CD-event count. Timestamps use the shifted clock. The count
relates to events since the preceding bookmark; it is not a cumulative event index.
The header includes the shift, RAW size, versions and bookmark period.

The SDK creates time slots at 2 ms intervals. Slots can repeat an earlier bookmark,
so the actual timestamp is not necessarily `slot * 2000`. Initial negative
timestamps are invalid placeholders. The final 20-byte record is a completion
marker, not a bookmark. Seeking selects a time slot and moves the file cursor to
its stored offset. See the pinned [index implementation](https://github.com/prophesee-ai/openeb/blob/9003b5416676e78ba994d912087486cfa94fae73/hal/cpp/src/facilities/i_events_stream.cpp)
and [record declaration](https://github.com/prophesee-ai/openeb/blob/9003b5416676e78ba994d912087486cfa94fae73/hal/cpp/include/metavision/hal/facilities/i_events_stream.h).

The SDK then resets the decoder timestamp to the reached bookmark. EVT3's reset
clears row/vector state and expects subsequent words to restore it before decoding
dependent events. Thus the index is usable without storing a complete decoder
snapshot, provided restart and filtering semantics are respected. Sources:
[seek example](https://github.com/prophesee-ai/openeb/blob/9003b5416676e78ba994d912087486cfa94fae73/hal/cpp/samples/metavision_hal_seek/metavision_hal_seek.cpp),
[EVT3 reset implementation](https://github.com/prophesee-ai/openeb/blob/9003b5416676e78ba994d912087486cfa94fae73/hal/cpp/include/metavision/hal/decoders/evt3/evt3_decoder.h).

This corrects the emphasis of the initial investigation: EVT3 is stateful, but
that does not mean we necessarily need our own full-state checkpoint index.
Existing SDK bookmarks are specifically intended to support decoder restarts.

## Local validation

Added [scripts/probe_fred_raw_index.py](../../scripts/probe_fred_raw_index.py), a
read-only comparison probe. It writes only a JSON report. It does not modify RAW
files or their sidecars, and is not wired into any training or rendering pipeline.

**Index audit:** all 222 available primary RAW recordings passed the probe's v2
layout, completion-marker, RAW-header/size, timestamp-order and offset checks.
Their indexes total **254.48 MiB**, about 0.25% of the 98.81 GiB RAW corpus. This
validates structural compatibility, not every decoded event or a content hash.

**Window comparisons:** sequences 0, 8 and 66, 24 windows each:

* 33,333 us and 400,000 us durations;
* anchors at 1, 10, 17, 60 and 110 seconds;
* windows straddling an absolute 24-bit EVT3 timestamp wrap;
* five additional deterministic random anchors per sequence.

For each window the probe selects the index slot 4 ms before the requested start,
seeks directly to its offset, initializes the repository decoder's time-high,
time-low and wrap fields from `bookmark.t + ts_shift_us`, and clears spatial
state. It decodes forward in 64 KiB buffers and retains only `[start, end)` events.
The actual preroll was generally around 6--8 ms because of bookmark placement.
The guard is an experimental precaution, not a proof that all malformed streams
resynchronize within 4 ms.

A separate reference pass decodes from the start of each recording, preserving
state across 1 MiB chunks. **All 72 indexed results matched the reference exactly
in shape, event order, coordinates, polarity and returned timestamp values.**

| Sequence | Exact matches | Median 33 ms extraction | Median 400 ms extraction | Sequential reference pass to ~110 s |
| --- | ---: | ---: | ---: | ---: |
| 0 | 24/24 | 12.8 ms | 32.3 ms | 33.1 s |
| 8 | 24/24 | 13.0 ms | 39.0 ms | 26.9 s |
| 66 | 24/24 | 13.1 ms | 39.6 ms | 19.0 s |

These are single-run CPU measurements with the existing Python decoder. Indexed
timing includes file open, seek, read, decode, filter and concatenate, but excludes
index parsing, patch rendering and GPU work. Reads follow the reference scan and
therefore benefit from the OS cache. The sequential column is one pass supplying
all reference windows, not the cost of one comparable indexed request; do not
interpret their ratio as end-to-end training speedup.

For a concrete access-cost example, sequence 0 at 60 seconds:

| Requested interval | Events | RAW bytes read | Extraction time |
| --- | ---: | ---: | ---: |
| `[59.966667, 60)` seconds | 3,091 | 64 KiB | 12.8 ms |
| `[59.6, 60)` seconds | 27,092 | 256 KiB | 51.7 ms |

The corresponding seeks start around RAW offsets 108.12 MB and 107.90 MB; that
prefix is skipped completely. Dense windows remain expensive: sequence 0's
`[0.6, 1)` seconds contains 1.48 million events and took 1.43 seconds to extract.
The index removes prefix scanning, not the cost of decoding requested events.

Reproduce from the project root:

```powershell
.venv312/Scripts/python.exe scripts/probe_fred_raw_index.py
```

Results: `outputs/fred_index_investigation/results.json`. Use `--sequences 0` for
a shorter comparison; `--sequences` with no values runs only the corpus index
audit. `--guard-us` controls the exploratory restart guard. The script requires
the project environment with `ecfm` installed, as used here.

## What these results do and do not establish

The direct-seek mechanism is demonstrably useful on these FRED files and works
without installing the SDK. However, both paths use the repository's decoder,
which emits float32 timestamps. Exact agreement establishes consistency with that
decoder, not independent SDK correctness or microsecond-accurate boundary handling.
An integer-timestamp decoder remains a prerequisite for a robust production reader.

The probe excludes invalid initial slots and out-of-range starts instead of
providing a start-of-file fallback. It does not cover all recordings, all bookmark
boundaries, physically empty intervals, corrupt streams or multiworker training.
These cases need validation before integration. Fully empty windows should be
handled using decoder clock progress, not by waiting indefinitely for a CD event.

SDK installation alone is not a guaranteed shortcut. The inspected Python HAL
bindings do not expose the C++ stream seek/reset methods; Python `RawReader`'s
forward-decoding seek remains a separate implementation. Also, the native index
loader checks platform and HAL/plugin versions, so loading these Linux-generated
4.6/5.1 indexes under a different SDK/platform may trigger rebuilding. Our format
probe reads their existing bytes directly. Sources: [stream Python bindings](https://github.com/prophesee-ai/openeb/blob/9003b5416676e78ba994d912087486cfa94fae73/hal/python/bindings/i_events_stream_python.cpp),
[decoder Python bindings](https://github.com/prophesee-ai/openeb/blob/9003b5416676e78ba994d912087486cfa94fae73/hal/python/bindings/i_events_stream_decoder_python.cpp),
[Python RAW reader](https://github.com/prophesee-ai/openeb/blob/9003b5416676e78ba994d912087486cfa94fae73/sdk/modules/core/python/pypkg/metavision_core/event_io/raw_reader.py).

## Recommended integration

Build a shared `FredRawWindowReader` around the existing index, with validated
sidecar metadata, integer timestamps and one reader/cache per worker. Its output
should be event arrays plus explicit window bounds and sensor geometry. Both
MAE rendering and downstream input providers can consume this interface.

1. Use indexed seeking for random pretraining crops and shuffled downstream
   anchors. Cache parsed index arrays; do not parse a sidecar for every sample.
2. Retain sequential decoding and a bounded RAM buffer for neighboring requests.
   Overlapping contexts and multiple tracks at one anchor can share reads.
3. Render histogram/CSTR tensors directly in memory. Detection labels and forecast
   tracks remain keyed by the shifted anchor time; no intermediate image is needed.
4. Validate against independent native decoding, exact time boundaries, start/end
   handling, wraps and worker isolation before training. Preserve half-open causal
   windows and fingerprint the source/index in any derived cache keys.
5. Benchmark preparation plus GPU consumption. If decoding is still limiting,
   accelerate the decoder while retaining this index contract; patch rendering
   requires its own profiling and optimization.

The bookmark event counts may also help estimate activity or schedule balanced
work, but are coarse slot statistics, not exact counts for arbitrary crop windows.
Spatial crop counts still require decoded coordinates.

The main implementation risk has narrowed: **we have evidence that the existing
sidecars can provide useful random access, so a new corpus-wide indexing pass is
not the first thing to build.** The remaining work is making that access robust
and connecting it to the shared renderer and task datasets.
