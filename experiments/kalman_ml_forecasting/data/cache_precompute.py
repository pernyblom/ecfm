"""Bounded, CPU-only precomputation of the selected instance cutouts."""
from __future__ import annotations

import faulthandler
import math
import multiprocessing
import time

import torch
from torch.utils.data import DataLoader, Dataset


def initialize_input_worker(worker_id: int) -> None:
    # Python exceptions already propagate through DataLoader. Native faults need
    # faulthandler in each process to leave a traceback on stderr as well.
    faulthandler.enable(all_threads=True)


def precompute_loader_options(config: dict) -> dict:
    """Keep these defaults independent of the training loader (including CUDA)."""
    workers = int(config.get("num_workers", 0))
    batch_size = int(config.get("batch_size", 32))
    prefetch = int(config.get("prefetch_factor", 1))
    timeout = float(config.get("timeout", 0))
    log_every = int(config.get("log_every", 100))
    if workers < 0 or batch_size < 1 or prefetch < 1 or log_every < 1:
        raise ValueError("data.instance_cache.precompute requires num_workers >= 0 and "
                         "batch_size, prefetch_factor, log_every >= 1.")
    if not math.isfinite(timeout) or timeout < 0 or (workers == 0 and timeout != 0):
        raise ValueError("data.instance_cache.precompute.timeout must be finite and >= 0 "
                         "(0 when num_workers is 0).")
    context = str(config.get("multiprocessing_context", "spawn"))
    if context not in {"spawn", "forkserver"}:
        raise ValueError("data.instance_cache.precompute.multiprocessing_context must be "
                         "spawn or forkserver to avoid inheriting SQLite/CUDA state.")
    if workers and context not in multiprocessing.get_all_start_methods():
        raise ValueError(f"data.instance_cache.precompute.multiprocessing_context {context!r} "
                         "is unavailable on this platform.")
    options = dict(batch_size=batch_size, num_workers=workers,
                   pin_memory=bool(config.get("pin_memory", False)), timeout=timeout,
                   shuffle=False, drop_last=False, collate_fn=sum,
                   generator=torch.Generator().manual_seed(0))
    if workers:
        options.update(prefetch_factor=prefetch, persistent_workers=False,
                       multiprocessing_context=context, worker_init_fn=initialize_input_worker)
    return options


class _CacheSamples(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset.samples[index]
        anchor = sample["past_boxes"][-1]
        for rep, references in sample["input_paths"].items():
            if not self.dataset._cache_representation(rep):
                continue
            for reference in references if isinstance(references, list) else [references]:
                try:
                    # Discard each tensor immediately. Only completion counts
                    # cross worker queues; no image batches or CUDA allocations.
                    self.dataset._prepared_input(rep, reference, anchor)
                except Exception as error:
                    raise RuntimeError(
                        f"Precomputing input cache failed at sample {index} "
                        f"({sample['folder']}/{sample['anchor_stem']}), "
                        f"representation {rep}, reference {reference!r}"
                    ) from error
        return 1


def populate_instance_cache(dataset) -> None:
    config = dict(dataset.instance_cache.get("precompute") or {})
    options = precompute_loader_options(config)
    cache = dataset._input_cache
    # No open parent connection or RAW event cache may reach another process.
    cache.close()
    dataset._indexed_source = None
    started = time.monotonic()
    print(f"Precomputing input cutouts for {len(dataset)} selected samples "
          f"(workers={options['num_workers']}, batch_size={options['batch_size']}): {cache.path}",
          flush=True)
    iterator = None
    completed = 0
    try:
        loader = DataLoader(_CacheSamples(dataset), **options)
        iterator = iter(loader)
        for step, count in enumerate(iterator, 1):
            completed += count
            if step % int(config.get("log_every", 100)) == 0 or completed == len(dataset):
                print(f"Input cutout precompute: {completed}/{len(dataset)} samples "
                      f"in {time.monotonic() - started:.1f}s", flush=True)
        if completed != len(dataset):
            raise RuntimeError(f"Incomplete input cache: visited {completed}/{len(dataset)} samples")
    finally:
        # Exhaustion shuts workers down normally. Also join them on an error or
        # Ctrl-C, before training can allocate GPU memory or start its loaders.
        try:
            if iterator is not None and hasattr(iterator, "_shutdown_workers"):
                iterator._shutdown_workers()
        finally:
            cache.close()
            dataset._indexed_source = None
    dataset._input_cache_complete = True
    print("Input cutout precompute complete; cache misses will fail during training.", flush=True)
