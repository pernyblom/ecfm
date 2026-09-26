"""Validation-only probes of the just-saved epoch, isolated from SSL state."""
from copy import deepcopy
from pathlib import Path
import random

import numpy as np
import torch


def run_periodic_probe(cfg, checkpoint, epoch):
    from .downstream import run
    probe_cfg = deepcopy(cfg)
    overrides = deepcopy(cfg['train']['linear_probe'])
    overrides.pop('every', None)
    probe_cfg['downstream'].setdefault('feature_cache', {}).update(enabled=True, storage='memory')
    def merge(target, source):
        for key, value in source.items():
            if key != 'regions' and isinstance(value, dict) and isinstance(target.get(key), dict):
                merge(target[key], value)
            else:
                target[key] = value
    merge(probe_cfg['downstream'], overrides)
    from .config import validate_downstream
    validate_downstream(probe_cfg)
    output = Path(cfg['train']['output_dir']) / 'probes' / f'epoch_{epoch:04d}'
    py_state, np_state = random.getstate(), np.random.get_state()
    # The probe has its own encoder and optimizer. Preserve all RNG streams
    # too, so enabling it cannot alter subsequent SSL training randomness.
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    print(f'Linear probe after {epoch + 1} completed epochs (checkpoint epoch {epoch})', flush=True)
    try:
        with torch.random.fork_rng(devices=devices):
            return run(probe_cfg, checkpoint, 'linear_probe', output,
                       evaluate_test=False, save_weights=False)
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
