"""Benchmark Kalman-only and image-conditioned forecasting inference.

The benchmark deliberately uses pre-created device tensors.  Dataset I/O and
spatial crop construction are excluded for both paths, so the result measures
the forecasting algorithms rather than storage/cache performance.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import platform
import statistics
import sys
import time
from typing import Any, Callable

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.kalman_ml_forecasting.models.factory import build_model
from experiments.kalman_ml_forecasting.models.kalman_filter import (
    ConfiguredBoxKalmanFilter,
)
from experiments.kalman_ml_forecasting.utils.config import (
    load_config,
    resolve_representation_image_sizes,
    resolve_representation_sequences,
)


def _is_kalman_only(cfg: dict[str, Any]) -> bool:
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    return (
        not list(data_cfg.get("representations", []))
        and not bool(model_cfg.get("use_filter_state_features", False))
        and str(model_cfg.get("filter_covariance_features", "none")).lower() == "none"
        and str(model_cfg.get("history_feature_mode", "raw")).lower() == "none"
    )


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))
    return ordered[index]


def _make_inputs(cfg: dict[str, Any], device: torch.device, batch_size: int, history_steps: int, future_steps: int, *, seed: int, step_s: float):
    dtype = torch.float32
    # Plausible, deterministic boxes avoid pathological singular inputs while
    # keeping identical trajectories across configurations.
    generator = torch.Generator(device="cpu").manual_seed(seed)
    centers = torch.rand((batch_size, history_steps, 2), generator=generator) * 0.6 + 0.2
    sizes = torch.rand((batch_size, history_steps, 2), generator=generator) * 0.1 + 0.05
    past_boxes = torch.cat((centers, sizes), dim=-1).to(device)
    dt = step_s
    past_times = torch.arange(history_steps, dtype=dtype).mul(dt).expand(batch_size, -1).to(device)
    future_times = (
        torch.arange(history_steps, history_steps + future_steps, dtype=dtype).mul(dt).expand(batch_size, -1).to(device)
    )
    channels = int(cfg["model"].get("backbone", {}).get("in_channels", 3))
    inputs = {}
    sequence_cfg = resolve_representation_sequences(cfg["data"])
    for rep, (width, height) in resolve_representation_image_sizes(cfg["data"]).items():
        shape = (batch_size, channels, height, width)
        if rep in sequence_cfg:
            shape = (batch_size, sequence_cfg[rep]["length"], channels, height, width)
        inputs[rep] = torch.randn(shape, generator=generator).to(device)
    return inputs, past_boxes, past_times, future_times


def _load_checkpoint(model: torch.nn.Module, path: Path, device: torch.device) -> None:
    state = torch.load(path, map_location=device)
    model.load_state_dict(state.get("model", state))


def benchmark(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    inference_cfg = dict(cfg.get("inference") or {})
    requested_device = str(args.device or inference_cfg.get("device", "cuda"))
    device = torch.device(requested_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; set --device to cpu.")
    batch_size = int(args.batch_size)
    history_steps = int(args.history_steps)
    future_steps = int(args.future_steps)
    warmup = int(args.warmup)
    iterations = int(args.iterations)
    if min(batch_size, history_steps, future_steps, iterations) < 1 or warmup < 0:
        raise ValueError("batch/history/future/iterations must be positive and warmup non-negative.")

    inputs, past_boxes, past_times, future_times = _make_inputs(
        cfg, device, batch_size, history_steps, future_steps, seed=args.seed, step_s=args.step_s
    )
    parameter_count = 0
    kalman_only = _is_kalman_only(cfg)
    if kalman_only:
        kalman_model = ConfiguredBoxKalmanFilter(cfg.get("kalman")).to(device).eval()
        operation: Callable[[], Any] = lambda: kalman_model(
            past_boxes, past_times, future_times
        )
    else:
        model = build_model(cfg, device).eval()
        if args.checkpoint:
            _load_checkpoint(model, args.checkpoint, device)
        parameter_count = sum(p.numel() for p in model.parameters())
        operation = lambda: model(inputs, past_boxes, past_times, future_times)

    with torch.inference_mode():
        for _ in range(warmup):
            operation()
        _sync(device)
        elapsed_ms: list[float] = []
        for _ in range(iterations):
            _sync(device)
            start = time.perf_counter_ns()
            operation()
            _sync(device)
            elapsed_ms.append((time.perf_counter_ns() - start) / 1.0e6)

    mean_ms = statistics.fmean(elapsed_ms)
    return {
        "path": "kalman_only" if kalman_only else "ml",
        "implementation": "configured_kalman" if kalman_only else "model_forward",
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
        "torch_version": torch.__version__,
        "batch_size": batch_size,
        "history_steps": history_steps,
        "future_steps": future_steps,
        "warmup": warmup,
        "iterations": iterations,
        "representation_sizes_wh": {k: list(v) for k, v in resolve_representation_image_sizes(cfg["data"]).items()},
        "parameter_count": parameter_count,
        "latency_batch_ms": {
            "mean": mean_ms,
            "median": statistics.median(elapsed_ms),
            "stdev": statistics.stdev(elapsed_ms) if len(elapsed_ms) > 1 else 0.0,
            "p05": _percentile(elapsed_ms, 0.05),
            "p95": _percentile(elapsed_ms, 0.95),
            "min": min(elapsed_ms),
            "max": max(elapsed_ms),
        },
        "mean_latency_per_sample_ms": mean_ms / batch_size,
        "throughput_samples_s": batch_size * 1000.0 / mean_ms,
        "scope": "forecast inference only; excludes dataset loading/cutout creation and host-to-device transfer",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument(
        "--device",
        help="Override inference.device from the config (which defaults to cuda).",
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--history-steps", type=int, default=12)
    parser.add_argument("--future-steps", type=int, default=40)
    parser.add_argument("--step-s", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()
    result = benchmark(load_config(args.config), args)
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered + "\n", encoding="utf-8")
        print(f"Wrote {args.output_json}")


if __name__ == "__main__":
    main()
