# Pretraining bottleneck investigation — 2026-10-03

The main bottleneck was CPU patch construction, with serial loading leaving the
GPU idle between batches. Reading the `.npy` event files was inexpensive. Model
size was not the cause of the observed wait.

Measurements used the local RTX 4070 SUPER, PyTorch 2.6.0+cu124, the 420-token THU
configuration, batch size 32, and the first 12 recordings in the seeded training
partition (27,795–736,891 events). The model benchmark used two warmup steps then
five synchronized forward/backward/optimizer steps. File reads use a warm OS cache;
these are bounded component benchmarks, not full-training throughput guarantees.

| Work | Measured time |
| --- | ---: |
| Original preparation, per recording | 346 ms |
| Original serial preparation, projected to a batch of 32 | 11.08 s |
| Raw event load/normalization, per recording | 3.7 ms |
| Optimized preparation without patch-cache hits, per recording | 82 ms |
| Optimized serial preparation, projected to a batch of 32 | 2.63 s |
| Warm patch-cache read/validation, per recording | 6.3 ms |
| GPU training step on an already prepared batch | 49 ms |
| Mask/selection planning alone | 13–15 ms |

The original sample profile spent roughly 58% of its time in per-token event
filtering/loop work, and 36% in `build_patch`, including another round of filtering,
histogram accumulation, interpolation and normalization. There were 420 full-crop
voxel selections per example, 411 histogram builders and 9 CSTR builders. The same
spatio-temporal voxel was selected again for every representation.

The optimized renderer partitions events once per level (three partitions for
this layout), preserves order within each voxel, and shares those events across
the representations. Common histogram planes use `numpy.bincount` over polarity
and pixel/bin indices. Rotated projections retain the existing renderer. CSTR
accumulation is unchanged. Uncached preparation was approximately **4.2× faster**
on these recordings. Tests compare histogram output and boundary assignments
against the existing implementation, including empty voxels and uneven grids.

The new optional crop bank caches final patch tensors plus geometry, duration and
log-count metadata. Warm-cache sample preparation was approximately **55× faster**
than original uncached preparation in this measurement. That is a data-preparation
ratio, not an end-to-end training speedup.

With four persistent workers, a second data-only pass over 128 cached recordings
took **0.213 seconds** in total; its first batch took 0.149 seconds. The first pass,
including Windows worker startup and many cache misses, took 4.06 seconds. A separate
two-epoch CUDA smoke run, limited to two training batches per epoch and using one
repeated crop per recording, reached **145 examples/second** in its warm second
epoch (mean data wait 99 ms/batch, completed step 122 ms/batch). Startup, rendering,
prefetching, disk cache state and CPU contention all affect these short measurements.
The recommended eight-view bank only repeats a given crop after eight epochs.

Secondary overhead remains in mask planning and many small per-group operations
with CPU/GPU synchronization. The full decoder still attends to all 420 candidates.
Those are candidates for later optimization once cached training makes them a
substantial share of runtime. Rendering every candidate is still required on a
cache miss; this change does not implement lazy rendering for a learned selector.

## Reproduce

```powershell
python -m experiments.hierarchical_mae.profile --config experiments/hierarchical_mae/configs/thu.yaml --samples 12 --steps 5 --output outputs/hierarchical_mae_profile.json
python -m experiments.hierarchical_mae.profile --config experiments/hierarchical_mae/configs/thu_cached.yaml --samples 12 --steps 5 --loader-batches 4 --output outputs/hierarchical_mae_profile_cached.json
```

The profiler does not read or overwrite training checkpoints. It creates a temporary
model in memory and, when caching is enabled, populates the requested samples' patch
cache. `--loader-batches` runs two data-only passes over a bounded prefix using the
configured worker count. JSON includes individual sample times, aggregate GPU and
mask times, loader timing, and a Python call profile. Original local measurements
are in `outputs/hierarchical_mae_profile_before.json`, optimized uncached measurements
in `outputs/hierarchical_mae_profile_render_optimized.json`, and cache measurements
in `outputs/hierarchical_mae_profile_cached.json` (ignored runtime artifacts).

Use the cache/resume instructions in [README.md](README.md) to enable the crop bank.
The default config keeps new training crops each epoch; its validation cache is
exactly reusable without reducing crop diversity.
