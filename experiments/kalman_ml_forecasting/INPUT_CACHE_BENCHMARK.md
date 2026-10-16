# Indexed input cache benchmark

Measured on 2026-10-16 using real FRED sequences `0`, `1`, and `3`. A warm
cache made actual GPU training epochs **8.28x faster** in this small experiment.
This is a throughput measurement, not a model accuracy or convergence study.

Setup: 128 deterministically sampled instances, batch size 32, three measured
passes after the first pass, and the base configuration with indexed event
inputs and 64x64 fixed cutouts. Event representations were `xt_my`, `yt_mx`, and
`cstr3`; the mixed case also loaded `padded_rgb`. Event windows were 400 ms,
history 400 ms and forecast 800 ms. Box augmentation and decorrelation were
disabled. Compression used lossless zlib level 1.

Hardware/software: Intel Core i5-14600KF, NVIDIA RTX 4070 SUPER, Windows 11,
Python 3.12.5, PyTorch 2.6.0+cu124. The main process used 14 PyTorch threads;
DataLoader workers used PyTorch's usual one thread each. Workers were
persistent; pinning was disabled for both paths.

All times below are seconds per 128 instances. Steady times are medians of
three passes. The uncached path retained the existing 32 MiB in-memory event
cache per process. OS filesystem caches were not flushed, so these are not
cold-disk measurements. Dataset construction is excluded; the first pass
includes worker startup and initial reader setup.

| Inputs | Workers | Uncached steady | Warm disk cache | Speedup | First uncached pass | Cache-building pass |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Events + RGB | 0 | 11.586 | 1.016 | 11.41x | 12.266 | 11.795 |
| Events + RGB | 2 | 6.086 | 0.513 | 11.87x | 10.871 | 10.801 |
| Events only | 0 | 10.613 | 0.0440 | 241.34x | 10.668 | 10.728 |
| Events only | 2 | 5.560 | 0.0305 | 182.34x | 10.448 | 10.543 |

The loading measurement includes batching, collation, worker communication and
SHA-256 hashing of input tensors and box/time tensors. Every pass produced
identical hashes with and without the cache; hashes also matched across worker
counts for each input configuration. Each case started with its own empty disk
cache. Subsequent measurement order alternated between cached and uncached.

GPU training used the base model's four ResNet18 branches and the actual
`train.py::_run_epoch` implementation: forward, backward, AdamW updates,
gradient clipping and metrics. It used two persistent workers, one CUDA
warm-up batch, and three measured epochs per path in alternating order. The
benchmark's AdamW used learning rate 0.001 and its default weight decay 0.01.
The same model continued updating across measurements; these timings do not
compare loss values. Timers synchronized CUDA at both boundaries.

| Actual training | Median epoch | Range | Throughput |
| --- | ---: | ---: | ---: |
| Indexed, uncached | 6.474 s | 6.421-6.542 s | 19.8 instances/s |
| Indexed, warm cache | 0.782 s | 0.719-0.868 s | 163.8 instances/s |

That is an **8.28x speedup, or 87.9% less epoch time**, excluding validation,
checkpointing, dataset construction and first-epoch cache building. Cache
building cost roughly one ordinary data pass, with no substantial extra cost
visible at this scale; the second pass already benefited.

Each cache stored 384 event cutouts in a 1.94 MiB SQLite database, including its
index/page overhead. Compressed tensor payloads occupied 1.17 MiB; the raw
float32 event tensors would occupy 18 MiB. RGB and sample metadata are excluded
from those sizes.

RGB reading accounts for most of the remaining warm-cache data-loading time:
the two-worker mixed case took 0.513 s versus 0.0305 s for events alone. Model
compute also limits the training gain relative to the loading-only gain. These
results describe a small subset that fits comfortably in the OS file cache;
larger datasets, different storage, crop sizes, representations and temporal
sequences need their own measurement before assuming the same speedups.

Reproduce from the repository root with a fresh output directory:

```powershell
python experiments/kalman_ml_forecasting/benchmark_input_cache.py --folders 0 1 3 --samples 128 --workers 0 2 --repeats 3 --training --output-dir outputs/input_cache_benchmark_new
```

The benchmark saves per-pass JSON timings, hashes, cache sizes and resolved
configuration overrides. The measured run's raw results are in
`outputs/input_cache_benchmark_20261016/results.json`.
