"""Recursive transfers for image tensors and nested hierarchy views."""
import torch


def to_device(value, device, non_blocking=False):
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=non_blocking)
    if isinstance(value, dict):
        return {k: to_device(v, device, non_blocking) for k, v in value.items()}
    if isinstance(value, list):
        return [to_device(v, device, non_blocking) for v in value]
    if isinstance(value, tuple):
        return tuple(to_device(v, device, non_blocking) for v in value)
    return value
