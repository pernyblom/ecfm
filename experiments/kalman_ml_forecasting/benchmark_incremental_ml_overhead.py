"""Standalone CUDA benchmark for post-filter learned forecasting overhead.

All experiment parameters are constants below; no experiment config is read.
The Kalman history pass and history feature encoding happen before timing. The
script measures eager PyTorch and, when supported, torch.compile execution.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import statistics
import sys
from typing import Callable

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.kalman_ml_forecasting.models.kalman_filter import (
    kalman_filter_history_tensor_params,
    kalman_std_tensors_from_config,
)
from experiments.kalman_ml_forecasting.models.kalman_residual import KalmanResidualForecaster


# Fixed benchmark protocol.
DEVICE = "cuda"
BATCH_SIZE = 1
HISTORY_STEPS = 12
FUTURE_STEPS = 24
STEP_S = 1.0 / 30.0
# (width, height) pairs. For example, add (1280, 720) to benchmark a full frame.
IMAGE_SIZES = ((64, 64), (224, 224), (1280, 720))
WARMUP = 500
ITERATIONS = 5000
SEED = 123


class ImageEncodingStage(nn.Module):
    def __init__(self, model: KalmanResidualForecaster) -> None:
        super().__init__()
        self.encoder = model.encoders["cstr3"]
        self.fusion = model.image_fusion

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        pooled = self.encoder(image).pooled
        return self.fusion(pooled)


class HistoryEncodingStage(nn.Module):
    def __init__(self, model: KalmanResidualForecaster) -> None:
        super().__init__()
        self.encoder = model.history_encoder

    def forward(self, past_boxes: torch.Tensor, past_times: torch.Tensor) -> torch.Tensor:
        relative_positions = past_boxes[..., :2] - past_boxes[:, -1:, :2]
        boxes = torch.cat((relative_positions, past_boxes[..., 2:]), dim=-1)
        relative_times = past_times - past_times[:, -1:]
        features = torch.cat((boxes, relative_times.unsqueeze(-1)), dim=-1).flatten(1)
        return self.encoder(features)


class ResidualRolloutStage(nn.Module):
    def __init__(self, model: KalmanResidualForecaster) -> None:
        super().__init__()
        self.residual_head = model.residual_head
        self.residual_scale = float(model.residual_scale)

    def forward(
        self,
        image_feat: torch.Tensor,
        history_feat: torch.Tensor,
        initial_state: torch.Tensor,
        step_dts: torch.Tensor,
    ) -> torch.Tensor:
        state = initial_state
        predictions = []
        for step in range(FUTURE_STEPS):
            dt = step_dts[:, step]
            step_feat = torch.cat((image_feat, history_feat, state, dt.unsqueeze(-1)), dim=-1)
            accel = self.residual_head(step_feat) * self.residual_scale
            pos, vel = state[:, :4], state[:, 4:]
            next_pos = (pos + vel * dt.unsqueeze(-1) + 0.5 * accel * dt.square().unsqueeze(-1)).clamp(0.0, 1.0)
            next_vel = vel + accel * dt.unsqueeze(-1)
            state = torch.cat((next_pos, next_vel), dim=-1)
            predictions.append(next_pos)
        return torch.stack(predictions, dim=1)


class ZeroResidualRolloutStage(nn.Module):
    def forward(self, initial_state: torch.Tensor, step_dts: torch.Tensor) -> torch.Tensor:
        state = initial_state
        predictions = []
        for step in range(FUTURE_STEPS):
            dt = step_dts[:, step]
            pos, vel = state[:, :4], state[:, 4:]
            next_pos = (pos + vel * dt.unsqueeze(-1)).clamp(0.0, 1.0)
            state = torch.cat((next_pos, vel), dim=-1)
            predictions.append(next_pos)
        return torch.stack(predictions, dim=1)


def _build_model(image_width: int, image_height: int, device: torch.device) -> KalmanResidualForecaster:
    return KalmanResidualForecaster(
        representations=["cstr3"],
        image_sizes={"cstr3": (image_width, image_height)},
        backbone_cfg={"type": "resnet18", "in_channels": 3, "out_dim": 128},
        history_steps=HISTORY_STEPS,
        fusion_hidden_dim=256,
        fusion_layers=1,
        history_feature_mode="relative",
        state_hidden_dim=128,
        state_layers=2,
        residual_hidden_dim=256,
        residual_layers=2,
        residual_scale=1.0,
        predict_size_residuals=True,
        use_filter_state_features=False,
        filter_state_feature_mode="full",
        filter_state_center_position_normalization="frame_centered",
        filter_covariance_features="none",
        initial_state_source="kalman_filter",
        kalman_params={"motion_model": "constant_velocity"},
    ).to(device).eval()


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))]


def _measure(operation: Callable[[], torch.Tensor]) -> dict[str, float]:
    with torch.inference_mode():
        for _ in range(WARMUP):
            operation()
        torch.cuda.synchronize()
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(ITERATIONS)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(ITERATIONS)]
        for start, end in zip(starts, ends):
            start.record()
            operation()
            end.record()
        torch.cuda.synchronize()
    values = [start.elapsed_time(end) for start, end in zip(starts, ends)]
    return {
        "mean_ms": statistics.fmean(values),
        "median_ms": statistics.median(values),
        "stdev_ms": statistics.stdev(values) if len(values) > 1 else 0.0,
        "p05_ms": _percentile(values, 0.05),
        "p95_ms": _percentile(values, 0.95),
    }


def _run_mode(
    image_stage: nn.Module,
    history_stage: nn.Module,
    residual_stage: nn.Module,
    zero_stage: nn.Module,
    image: torch.Tensor,
    past_boxes: torch.Tensor,
    past_times: torch.Tensor,
    image_feat: torch.Tensor,
    history_feat: torch.Tensor,
    initial_state: torch.Tensor,
    step_dts: torch.Tensor,
) -> dict[str, object]:
    image_stats = _measure(lambda: image_stage(image))
    history_stats = _measure(lambda: history_stage(past_boxes, past_times))
    residual_stats = _measure(lambda: residual_stage(image_feat, history_feat, initial_state, step_dts))
    zero_stats = _measure(lambda: zero_stage(initial_state, step_dts))
    incremental = residual_stats["median_ms"] - zero_stats["median_ms"]
    return {
        "image_encoding": image_stats,
        "history_encoding": history_stats,
        "residual_rollout": residual_stats,
        "zero_residual_rollout": zero_stats,
        "derived_median_incremental_rollout_ms": incremental,
        "derived_median_total_ml_overhead_ms": (
            image_stats["median_ms"] + history_stats["median_ms"] + incremental
        ),
    }


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("This focused benchmark requires CUDA.")
    device = torch.device(DEVICE)
    torch.manual_seed(SEED)
    torch.set_float32_matmul_precision("high")
    results: dict[str, object] = {
        "protocol": {
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device),
            "batch_size": BATCH_SIZE,
            "history_steps": HISTORY_STEPS,
            "future_steps": FUTURE_STEPS,
            "step_s": STEP_S,
            "forecast_ms": FUTURE_STEPS * STEP_S * 1000.0,
            "warmup": WARMUP,
            "iterations": ITERATIONS,
            "scope": "post-Kalman-history CUDA overhead only",
        },
        "sizes": {},
    }

    generator = torch.Generator(device="cpu").manual_seed(SEED)
    centers = torch.rand((BATCH_SIZE, HISTORY_STEPS, 2), generator=generator) * 0.6 + 0.2
    sizes = torch.rand((BATCH_SIZE, HISTORY_STEPS, 2), generator=generator) * 0.1 + 0.05
    past_boxes = torch.cat((centers, sizes), dim=-1).to(device)
    past_times = torch.arange(HISTORY_STEPS, dtype=torch.float32, device=device).mul(STEP_S).expand(BATCH_SIZE, -1)
    step_dts = torch.full((BATCH_SIZE, FUTURE_STEPS), STEP_S, device=device)
    kalman_params = kalman_std_tensors_from_config(None, device=device, dtype=past_boxes.dtype)
    # Intentionally outside every timed region.
    filter_state, _ = kalman_filter_history_tensor_params(past_boxes, past_times, kalman_params)
    initial_state = filter_state[:, :8]

    for image_width, image_height in IMAGE_SIZES:
        model = _build_model(image_width, image_height, device)
        image_stage = ImageEncodingStage(model).eval()
        history_stage = HistoryEncodingStage(model).eval()
        residual_stage = ResidualRolloutStage(model).eval()
        zero_stage = ZeroResidualRolloutStage().to(device).eval()
        image = torch.randn((BATCH_SIZE, 3, image_height, image_width), device=device)
        with torch.inference_mode():
            image_feat = image_stage(image)
            history_feat = history_stage(past_boxes, past_times)
        size_result: dict[str, object] = {
            "image_size_px": [image_width, image_height],
            "eager": _run_mode(
                image_stage, history_stage, residual_stage, zero_stage, image,
                past_boxes, past_times, image_feat,
                history_feat, initial_state, step_dts,
            ),
        }
        try:
            compiled_image = torch.compile(image_stage, mode="reduce-overhead", fullgraph=True)
            compiled_history = torch.compile(history_stage, mode="reduce-overhead", fullgraph=True)
            compiled_residual = torch.compile(residual_stage, mode="reduce-overhead", fullgraph=True)
            compiled_zero = torch.compile(zero_stage, mode="reduce-overhead", fullgraph=True)
            # Trigger compilation before the benchmark's own warm-up.
            with torch.inference_mode():
                compiled_image(image)
                compiled_history(past_boxes, past_times)
                compiled_residual(image_feat, history_feat, initial_state, step_dts)
                compiled_zero(initial_state, step_dts)
            torch.cuda.synchronize()
            size_result["compiled"] = _run_mode(
                compiled_image, compiled_history, compiled_residual, compiled_zero, image,
                past_boxes, past_times, image_feat,
                history_feat, initial_state, step_dts,
            )
        except Exception as exc:
            size_result["compiled"] = {"available": False, "error": f"{type(exc).__name__}: {exc}"}
        results["sizes"][f"{image_width}x{image_height}"] = size_result

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
