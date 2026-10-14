"""Positive definite, optionally coordinate-separated covariance parameters."""
from __future__ import annotations

import math

import torch
from torch import nn

from .kalman_filter import channel_mask


class CovarianceParameters(nn.Module):
    """Learn a bounded triangular factor; forbidden entries have no parameters.

    Factor off-diagonals are relative to the row's positive diagonal. This keeps
    tiny position-noise scales and large velocity-noise scales trainable together.
    """

    def __init__(self, covariance: torch.Tensor, *, structure: str, optimize: bool,
                 min_std: float, max_std: float) -> None:
        super().__init__()
        self.min_std, self.max_std = float(min_std), float(max_std)
        dim = covariance.shape[0]
        factor = torch.linalg.cholesky(covariance)
        diagonal = factor.diagonal()
        if (diagonal < min_std).any() or (diagonal > max_std).any():
            raise ValueError("Initial covariance Cholesky scales must be inside min_std/max_std.")
        mask = torch.tril(torch.ones_like(covariance, dtype=torch.bool), diagonal=-1)
        if structure == "per_channel":
            mask &= channel_mask(dim, device=covariance.device)
        elif structure == "diagonal":
            mask.zero_()
        rows, cols = torch.where(mask)
        self.register_buffer("rows", rows)
        self.register_buffer("cols", cols)
        self.log_diagonal = nn.Parameter(diagonal.log(), requires_grad=optimize)
        relative = factor[rows, cols] / diagonal[rows]
        # The smooth bound prevents unbounded correlations while retaining SPD.
        if (relative.abs() >= 10).any():
            raise ValueError("Initial covariance factor off-diagonal/diagonal ratio must be < 10.")
        self.raw_off_diagonal = nn.Parameter(torch.atanh(relative / 10), requires_grad=optimize)

    def forward(self) -> torch.Tensor:
        diagonal = self.log_diagonal.exp()
        factor = torch.diag(diagonal)
        factor[self.rows, self.cols] = diagonal[self.rows] * 10 * self.raw_off_diagonal.tanh()
        return factor @ factor.T

    @torch.no_grad()
    def clamp_(self) -> None:
        if self.log_diagonal.requires_grad:
            self.log_diagonal.clamp_(math.log(self.min_std), math.log(self.max_std))
        if self.raw_off_diagonal.requires_grad:
            self.raw_off_diagonal.clamp_(-5, 5)

    def correlation_penalty(self) -> torch.Tensor:
        covariance = self()
        scale = covariance.diagonal().sqrt()
        correlation = covariance / (scale[:, None] * scale[None, :])
        return (correlation - torch.eye(len(scale), device=scale.device, dtype=scale.dtype)).square().mean()
