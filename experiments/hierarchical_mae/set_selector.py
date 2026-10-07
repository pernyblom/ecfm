"""Coarse-conditioned, hard-budget selection of complete replacement sets."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from .masking import TokenPlan
from .swap_oracle import search_space, token_sets


def coarse_inputs(backbone, view, budget, drops, adds):
    """Root patch projections + candidate metadata; no fine patch content/labels."""
    valid, count = view['valid_mask'], view['log_counts']
    if budget < drops or not (valid.sum(1) >= budget+adds).all():
        raise ValueError('Insufficient valid tokens for replacement pool')
    order = count.masked_fill(~valid, -torch.inf).argsort(dim=1, descending=True, stable=True)
    pool = torch.cat([order[:, budget-drops:budget].flip(1), order[:, budget:budget+adds]], 1)
    roots = []
    for group in backbone.layout.groups:
        if group.level == 0:
            if not valid[:, group.start:group.stop].all():
                raise ValueError('Root tokens must be valid')
            values = view['patches'][group.key]
            roots.append(backbone.patch_encoders[group.key](values.flatten(0, 1)).reshape(len(count), -1))
    rank = order.argsort(dim=1).float()/count.shape[1]
    levels = F.one_hot(backbone.layout.level_ids.to(count.device)).float()[None].expand(len(count), -1, -1)
    planes = F.one_hot(backbone.layout.plane_ids.to(count.device)).float()[None].expand(len(count), -1, -1)
    descriptors = torch.cat([view['metadata'], count[..., None],
        (count/count.amax(1, keepdim=True).clamp_min(1))[..., None], rank[..., None], levels, planes], -1)
    descriptors = descriptors.gather(1, pool[..., None].expand(-1, -1, descriptors.shape[-1]))
    return torch.cat(roots, -1), descriptors


class SetPolicy(nn.Module):
    def __init__(self, root_dim, descriptor_dim, drops=4, adds=6, max_swaps=2, hidden=64, dropout=.1):
        super().__init__()
        if sum(math.comb(drops, k)*math.comb(adds, k) for k in range(max_swaps+1)) > 10000:
            raise ValueError('Too many replacement sets')
        self.settings = dict(root_dim=root_dim, descriptor_dim=descriptor_dim, drops=drops,
                             adds=adds, max_swaps=max_swaps, hidden=hidden, dropout=dropout)
        self.states, _ = search_space(drops, adds, max_swaps)
        removed, inserted = torch.zeros(len(self.states), drops), torch.zeros(len(self.states), adds)
        for i, (d, a) in enumerate(self.states):
            removed[i, list(d)] = 1
            inserted[i, list(a)] = 1
        self.register_buffer('removed', removed)
        self.register_buffer('inserted', inserted)
        self.register_buffer('depths', removed.sum(1))
        self.register_buffer('root_mean', torch.zeros(root_dim))
        self.register_buffer('root_scale', torch.ones(root_dim))
        self.register_buffer('desc_mean', torch.zeros(descriptor_dim))
        self.register_buffer('desc_scale', torch.ones(descriptor_dim))
        self.context = nn.Sequential(nn.Linear(root_dim, 32), nn.GELU())
        self.candidate = nn.Sequential(nn.Linear(descriptor_dim, 16), nn.GELU())
        self.scorer = nn.Sequential(nn.Linear(32+16*3+1, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, 1))
        nn.init.zeros_(self.scorer[-1].weight)
        nn.init.zeros_(self.scorer[-1].bias)

    @torch.no_grad()
    def fit_normalization(self, roots, descriptors):
        self.root_mean.copy_(roots.mean(0))
        self.root_scale.copy_(roots.std(0).clamp_min(.05))
        flat = descriptors.flatten(0, 1)
        self.desc_mean.copy_(flat.mean(0))
        self.desc_scale.copy_(flat.std(0).clamp_min(.05))

    def forward(self, roots, descriptors):
        context = self.context(((roots-self.root_mean)/self.root_scale).clamp(-10, 10))
        candidates = self.candidate(((descriptors-self.desc_mean)/self.desc_scale).clamp(-10, 10))
        removed = torch.einsum('sd,bdh->bsh', self.removed, candidates[:, :self.settings['drops']])
        inserted = torch.einsum('sa,bah->bsh', self.inserted, candidates[:, self.settings['drops']:])
        context = context[:, None].expand(-1, len(self.states), -1)
        depth = self.depths[None, :, None].expand(len(context), -1, -1)/self.settings['max_swaps']
        scores = self.scorer(torch.cat([context, removed, inserted, inserted-removed, depth], -1)).squeeze(-1)
        # No-change has known improvement zero, independent of network parameters.
        return torch.cat([torch.zeros_like(scores[:, :1]), scores[:, 1:]], 1)

    def choose(self, scores, threshold=0.):
        if not math.isfinite(threshold) or threshold < 0:
            raise ValueError('Threshold must be finite and nonnegative')
        gain, ids = scores.max(-1)
        return torch.where(gain > threshold, ids, torch.zeros_like(ids))


@torch.no_grad()
def selected_features(backbone, view, policy, threshold, budget=54):
    """No labels required; one transformer pass on exactly budget tokens."""
    options = policy.settings
    roots, descriptors = coarse_inputs(backbone, view, budget, options['drops'], options['adds'])
    choices = policy.choose(policy(roots, descriptors), threshold)
    ids, _, _, _ = token_sets(view['log_counts'], view['valid_mask'], budget,
                              options['drops'], options['adds'], policy.states)
    chosen = ids[torch.arange(len(ids), device=ids.device), choices]
    visible = torch.zeros_like(view['valid_mask']).scatter_(1, chosen, True)
    return backbone.features(view, TokenPlan(visible, torch.zeros_like(visible))), choices


class SetClassifier(nn.Module):
    def __init__(self, classifier, policy, threshold, budget):
        super().__init__()
        self.classifier, self.policy, self.threshold, self.budget = classifier, policy, threshold, budget

    def forward(self, view):
        features, _ = selected_features(self.classifier.backbone, view, self.policy, self.threshold, self.budget)
        return self.classifier.head(features)


def load_policy(path, probe=None, baseline=None, device='cpu'):
    """Load a trained selector and optional matching fresh head for inference."""
    from .downstream import Classifier
    from .feature_cache import file_digest
    from .model import HierarchicalMAE
    saved = torch.load(path, map_location='cpu', weights_only=True)
    baseline = baseline or saved['baseline']
    if file_digest(baseline) != saved['baseline_digest']:
        raise ValueError('Baseline checkpoint digest differs')
    source = torch.load(baseline, map_location='cpu', weights_only=True)
    classifier = Classifier(HierarchicalMAE(source['config']), source['config']['downstream']['num_classes'], True)
    classifier.load_state_dict(source['model'])
    policy = SetPolicy(**saved['architecture'])
    policy.load_state_dict(saved['policy'])
    if probe:
        head = torch.load(probe, map_location='cpu', weights_only=True)
        if head['policy_digest'] != file_digest(path):
            raise ValueError('Probe belongs to a different policy')
        classifier.head.load_state_dict(head['head'])
    return SetClassifier(classifier, policy, saved['threshold'], saved['budget']).to(device).eval().requires_grad_(False)
