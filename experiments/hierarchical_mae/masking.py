"""Selection and reconstruction targets are independent, explicit token sets."""
from dataclasses import dataclass
import math

import torch
from .information_selection import STRATEGIES as INFORMATION_STRATEGIES, selection_scores


@dataclass
class TokenPlan:
    visible: torch.Tensor  # [B,N] bool; physically gathered before self-attention
    target: torch.Tensor   # [B,N] bool; only these tokens contribute reconstruction loss
    gates: torch.Tensor | None = None  # [B,N]; optional differentiable selection weights

    def validate(self, valid):
        for value in (self.visible, self.target):
            if value.shape != valid.shape or value.dtype != torch.bool:
                raise ValueError('Token masks must be boolean [batch,tokens]')
            if (value & ~valid).any():
                raise ValueError('Cannot select invalid tokens')
        if (self.visible & self.target).any() or not self.visible.any(1).all():
            raise ValueError('Need >=1 visible token per example, disjoint from targets')
        if self.gates is not None:
            if (self.gates.shape != valid.shape or not torch.isfinite(self.gates).all()
                    or (self.gates < 0).any()):
                raise ValueError('Gates must be finite, nonnegative [batch,tokens]')
            if not (self.gates * self.visible).sum(1).gt(0).all():
                raise ValueError('Visible gates must have positive total weight')


def overlaps(boxes):
    lo, hi = boxes[:, :3], boxes[:, :3] + boxes[:, 3:]
    return ((lo[:, None] < hi[None]) & (hi[:, None] > lo[None])).all(-1)


def select(eligible, layout, options, activity, scores=None, straight_through=False, temperature=1.):
    """External scores may come from a coarse-token policy; no patch access required.

    Hard top-k reduces attention length. Optional straight-through sigmoid gates
    give a surrogate gradient through selected scores (not through top-k indices).
    A future policy can alternatively construct TokenPlan with soft gates directly.
    """
    strategy, budget = options.get('strategy', 'all'), options.get('budget', 0)
    if scores is None:
        if strategy == 'activity':
            scores = activity
        elif strategy == 'activity_random':
            power = options.get('activity_power', 1.)
            if type(power) not in (int, float) or not math.isfinite(power) or power < 0:
                raise ValueError('activity_power must be finite and nonnegative')
            if not torch.isfinite(activity).all() or (activity < 0).any():
                raise ValueError('Activity must be finite, nonnegative log1p(event_count)')
            # Gumbel top-k samples without replacement with successive draw weights
            # (1 + log1p(event_count))**power. The +1 gives empty voxels a chance
            # and makes all-empty recordings uniform, without special-case filling.
            uniform = torch.rand_like(activity).clamp(min=torch.finfo(activity.dtype).tiny)
            scores = power * torch.log1p(activity) - torch.log(-torch.log(uniform))
        elif strategy == 'coarse':
            scores = -layout.level_ids.to(activity.device).expand_as(activity).float()
        elif strategy == 'random':
            scores = torch.rand_like(activity)
        elif strategy == 'all':
            scores = torch.zeros_like(activity)
        else:
            raise ValueError(f'Unknown selection strategy: {strategy}')
    if scores.shape != eligible.shape or not torch.isfinite(scores).all() or temperature <= 0:
        raise ValueError('Selection scores must be finite [B,N]; temperature must be positive')
    ranks = scores.masked_fill(~eligible, -torch.inf).argsort(dim=1, descending=True, stable=True).argsort(dim=1)
    visible = eligible & (ranks < budget) if budget else eligible.clone()
    gates = None
    if straight_through:
        soft = torch.sigmoid(scores / temperature)
        gates = torch.ones_like(soft) + soft - soft.detach()
    return visible, gates


def make_plan(view, layout, masking=None, selection=None):
    valid, activity = view['valid_mask'], view['log_counts']
    if masking is not None and masking['strategy'] == 'mixed':
        overlapping = make_plan(view, layout, dict(masking, strategy='token'), selection)
        strict = make_plan(view, layout, dict(masking, strategy='subtree'), selection)
        choose = torch.rand((len(valid), 1), device=valid.device) < masking.get('overlap_probability', .5)
        return TokenPlan(torch.where(choose, overlapping.visible, strict.visible),
                         torch.where(choose, overlapping.target, strict.target))
    target = torch.zeros_like(valid)
    eligible = valid.clone()
    if masking is not None:
        ratio, strategy = masking['ratio'], masking['strategy']
        boxes = layout.boxes.to(valid.device)
        if strategy == 'token':
            for b in range(len(valid)):
                ids = valid[b].nonzero().flatten()
                if len(ids) < 2:
                    raise ValueError('MAE needs at least two valid tokens')
                n = min(len(ids)-1, max(1, round(len(ids)*ratio)))
                target[b, ids[torch.randperm(len(ids), device=valid.device)[:n]]] = True
            eligible &= ~target
        else:
            lid = masking.get('level', int(layout.level_ids.max()))
            cells = boxes[layout.level_ids.to(valid.device) == lid].unique(dim=0)
            if len(cells) < 2:
                raise ValueError('Spatial masking needs at least two cells')
            n = min(len(cells)-1, max(1, round(len(cells)*ratio)))
            for b in range(len(valid)):
                chosen = cells[torch.randperm(len(cells), device=valid.device)[:n]]
                contained = ((boxes[:, None, :3] >= chosen[None, :, :3]) &
                             (boxes[:, None, :3]+boxes[:, None, 3:] <= chosen[None, :, :3]+chosen[None, :, 3:])).all(-1).any(-1)
                if strategy == 'voxel':
                    contained &= layout.level_ids.to(valid.device) == lid
                target[b] = contained & valid[b]
            # Mask all representations, parents and children intersecting targets.
            eligible &= ~(target.float() @ overlaps(boxes).float()).bool()
    options = selection or {}
    scores = selection_scores(view, eligible, options) if options.get('strategy') in INFORMATION_STRATEGIES else None
    visible, gates = select(eligible, layout, options, activity, scores=scores)
    result = TokenPlan(visible, target, gates)
    result.validate(valid)
    return result
