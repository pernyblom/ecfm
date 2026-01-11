from __future__ import annotations

import torch


def random_mask(num_tokens: int, mask_ratio: float, generator: torch.Generator | None = None) -> torch.Tensor:
    if not 0.0 < mask_ratio < 1.0:
        raise ValueError("mask_ratio must be in (0, 1)")
    num_mask = int(num_tokens * mask_ratio)
    perm = torch.randperm(num_tokens, generator=generator)
    mask = torch.zeros(num_tokens, dtype=torch.bool)
    mask[perm[:num_mask]] = True
    return mask

