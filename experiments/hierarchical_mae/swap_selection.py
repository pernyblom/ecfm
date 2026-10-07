"""Conservative activity selection: predict the benefit of at most one swap.

The policy observes root patch features and candidate descriptors, never labels
or transformer outputs for alternative selections. Evaluated alternatives are
used only to supervise training and to cache features for controlled probes.
"""
import torch
from torch import nn
from torch.nn import functional as F


def activity_coarse_overlap(activity, valid, layout, budget=27):
    from .masking import select
    activity_mask, _ = select(valid, layout, dict(strategy='activity', budget=budget), activity)
    coarse_mask, _ = select(valid, layout, dict(strategy='coarse', budget=budget), activity)
    overlap = (activity_mask & coarse_mask).sum(1).float()/budget
    identical = (activity_mask == coarse_mask).all(1).float()
    return overlap, identical


def proposals(activity, valid, budget, drop_count=4, add_count=6):
    """Deterministic boundary swaps, preserving activity's canonical tie order.

    Returns [B,1+D*A,K] sorted token IDs. Alternative zero is exact activity.
    No root-retention override: even the no-op matches the existing heuristic.
    """
    if not 1 <= drop_count <= budget or add_count < 1:
        raise ValueError('Need 1 <= drop_count <= budget and add_count >= 1')
    if activity.shape != valid.shape or not (valid.sum(1) >= budget+add_count).all():
        raise ValueError('Not enough valid tokens for the requested alternatives')
    order = activity.masked_fill(~valid, -torch.inf).argsort(dim=1, descending=True, stable=True)
    baseline = order[:, :budget]
    drops = baseline[:, -drop_count:].flip(1)
    adds = order[:, budget:budget+add_count]
    removed = drops.repeat_interleave(add_count, 1)
    inserted = adds.repeat(1, drop_count)
    changed = baseline[:, None].expand(-1, drop_count*add_count, -1).clone()
    changed = torch.where(changed == removed[..., None], inserted[..., None], changed)
    return torch.cat([baseline[:, None], changed], 1).sort(-1).values, removed, inserted


def policy_inputs(backbone, view, removed, inserted):
    """Cheap patch projections (no transformer) plus counts, rank and geometry."""
    roots = (backbone.layout.level_ids == 0).nonzero().flatten().to(removed.device)
    # Encode only roots and proposed tokens. Geometry is already in descriptors.
    wanted = torch.zeros_like(view['valid_mask'])
    wanted[:, roots] = True
    wanted.scatter_(1, removed, True).scatter_(1, inserted, True)
    dim = backbone.cfg['model']['embed_dim']
    content = view['metadata'].new_zeros((*wanted.shape, dim))
    for group in backbone.layout.groups:
        local = wanted[:, group.start:group.stop]
        if local.any():
            b, i = local.nonzero(as_tuple=True)
            content[b, i+group.start] = backbone.patch_encoders[group.key](view['patches'][group.key][local])
    count = view['log_counts']
    rank = count.masked_fill(~view['valid_mask'], -torch.inf).argsort(dim=1, descending=True, stable=True).argsort(1)
    levels = F.one_hot(backbone.layout.level_ids.to(count.device)).float()[None].expand(len(count), -1, -1)
    planes = F.one_hot(backbone.layout.plane_ids.to(count.device)).float()[None].expand(len(count), -1, -1)
    descriptors = torch.cat([view['metadata'], count[..., None],
        (count/count.amax(1, keepdim=True).clamp_min(1))[..., None],
        (rank.float()/count.shape[1])[..., None], levels, planes], -1)
    batch = torch.arange(len(count), device=count.device)[:, None]
    context = content[:, roots].flatten(1)[:, None].expand(-1, removed.shape[1], -1)
    coarse = torch.cat([context, descriptors[batch, removed], descriptors[batch, inserted]], -1)
    local = torch.cat([coarse, content[batch, removed], content[batch, inserted]], -1)
    return coarse, local


class SwapPolicy(nn.Module):
    def __init__(self, input_dim, hidden=64):
        super().__init__()
        self.register_buffer('mean', torch.zeros(input_dim))
        self.register_buffer('scale', torch.ones(input_dim))
        self.net = nn.Sequential(nn.Linear(input_dim, hidden), nn.GELU(), nn.Linear(hidden, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, inputs):
        return self.net(((inputs-self.mean)/self.scale).clamp(-10, 10)).squeeze(-1)

    @torch.no_grad()
    def fit_normalization(self, inputs):
        flat = inputs.flatten(0, 1)
        self.mean.copy_(flat.mean(0))
        self.scale.copy_(flat.std(0).clamp_min(.05))


def choose(scores, threshold):
    """Index zero is no-op; ties and scores below threshold retain activity."""
    best, index = scores.max(-1)
    return torch.where(best > threshold, index+1, torch.zeros_like(index))


@torch.no_grad()
def selected_features(backbone, view, policy, threshold, budget=54,
                      drop_count=4, add_count=6, inputs='local'):
    """Deployment path: one 54-token transformer pass, no alternative evaluation."""
    from .masking import TokenPlan
    ids, removed, inserted = proposals(view['log_counts'], view['valid_mask'], budget, drop_count, add_count)
    coarse, local = policy_inputs(backbone, view, removed, inserted)
    choices = choose(policy(local if inputs == 'local' else coarse), threshold)
    selected = ids[torch.arange(len(ids), device=ids.device), choices]
    visible = torch.zeros_like(view['valid_mask']).scatter_(1, selected, True)
    return backbone.features(view, TokenPlan(visible, torch.zeros_like(visible))), choices
