"""Downstream-only coarse-conditioned, straight-through relaxed top-k selection."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from .masking import TokenPlan


def relaxed_topk(scores, eligible, k, temperature, noise_scale=0.):
    """Return [B,k,N] hard-forward/soft-backward rows and distinct hard IDs.

    Sequential softmax relaxation with log(1-p) suppression (Xie & Ermon,
    IJCAI 2019). One Gumbel perturbation per candidate is shared by hard ranking
    and the relaxed rows. This is a biased straight-through surrogate, not a
    derivative of the discrete top-k indices. Matmul with candidate embeddings
    provides gradients through unselected candidates as well as selected ones.
    """
    if (scores.ndim != 2 or eligible.shape != scores.shape or eligible.dtype != torch.bool
            or not torch.isfinite(scores).all() or type(k) is not int or k < 1
            or (eligible.sum(1) < k).any()):
        raise ValueError('Need finite [B,N] scores and at least k eligible candidates per example')
    if not math.isfinite(temperature) or temperature <= 0 or not math.isfinite(noise_scale) or noise_scale < 0:
        raise ValueError('Temperature must be positive and noise scale nonnegative, both finite')
    perturbed = scores.float()
    if noise_scale:
        uniform = torch.rand_like(perturbed).clamp_min(torch.finfo(perturbed.dtype).tiny)
        perturbed = perturbed - noise_scale * torch.log(-torch.log(uniform))
    logits = perturbed.masked_fill(~eligible, -torch.inf)
    ids = logits.argsort(dim=1, descending=True, stable=True)[:, :k]
    hard = F.one_hot(ids, scores.shape[1]).to(scores.dtype)
    rows = []
    remaining = logits
    for _ in range(k):
        probabilities = F.softmax(remaining / temperature, dim=-1)
        rows.append(probabilities)
        remaining = remaining + torch.log((1-probabilities).clamp_min(1e-6))
    soft = torch.stack(rows, dim=1).to(scores.dtype)
    return hard + (soft-soft.detach()), ids


def schedule(options, epoch):
    fraction = min(max(epoch, 0)/max(options.get('anneal_epochs', 80)-1, 1), 1.)
    def interpolate(key, start, end):
        a, b = options.get(key, [start, end])
        return a + (b-a)*fraction
    return interpolate('temperature', 1., .25), interpolate('noise_scale', 1., 0.)


class CoarseSelector(nn.Module):
    def __init__(self, cfg, layout):
        super().__init__()
        self.options = cfg['downstream']['learned_selector']
        dim, hidden = cfg['model']['embed_dim'], self.options.get('hidden_dim', 128)
        roots = (layout.level_ids == 0).nonzero().flatten()
        self.register_buffer('roots', roots)
        self.register_buffer('candidates', (layout.level_ids != 0).nonzero().flatten())
        self.register_buffer('levels', layout.level_ids)
        self.register_buffer('planes', layout.plane_ids)
        self.level_embedding = nn.Embedding(int(layout.level_ids.max())+1, 8)
        self.plane_embedding = nn.Embedding(len(layout.representations), 8)
        self.context_projection = nn.Sequential(nn.Linear(len(roots)*dim, hidden), nn.GELU(), nn.LayerNorm(hidden))
        # Geometry (9), raw log-count and fraction of root log-count (2), level/plane (16).
        self.scorer = nn.Sequential(nn.Linear(hidden+27, hidden), nn.GELU(), nn.Linear(hidden, 1))
        nn.init.zeros_(self.scorer[-1].weight)
        nn.init.zeros_(self.scorer[-1].bias)

    def forward(self, root_features, view):
        global_features = self.context_projection(root_features.flatten(1))
        count = view['log_counts']
        relative_count = count / count[:, self.roots].amax(1, keepdim=True).clamp_min(1.)
        levels = self.level_embedding(self.levels)[None].expand(len(count), -1, -1)
        planes = self.plane_embedding(self.planes)[None].expand(len(count), -1, -1)
        features = torch.cat([global_features[:, None].expand(-1, count.shape[1], -1),
                              view['metadata'], count[..., None], relative_count[..., None], levels, planes], dim=-1)
        correction = self.scorer(features).squeeze(-1)
        return self.options.get('activity_prior_weight', 1.) * count + correction


class SelectionEncoder(nn.Module):
    """Wrap a pretrained MAE encoder without changing its pretraining state format."""
    is_learned_selector = True

    def __init__(self, backbone, mode):
        super().__init__()
        self.backbone, self.cfg, self.layout = backbone, backbone.cfg, backbone.layout
        self.selector = CoarseSelector(self.cfg, self.layout)
        self.options = self.cfg['downstream']['learned_selector']
        self.budget = self.options['budget']
        self.finetune = mode == 'selector_finetune'
        self.train_selector = mode != 'selector_probe'
        self.epoch = 0
        self.last_diagnostics = None

    def train(self, mode=True):
        super().train(mode)
        if not self.finetune:
            self.backbone.eval()
        if not self.train_selector:
            self.selector.eval()
        return self

    def encoder_parameters(self):
        if self.train_selector:
            yield from self.selector.parameters()
        if self.finetune:
            yield from self.backbone.encoder_parameters()

    def root_features(self, view):
        if self.options.get('context', 'patch') == 'transformer':
            visible = torch.zeros_like(view['valid_mask'])
            visible[:, self.selector.roots] = True
            encoded, _, _ = self.backbone.encode(view, TokenPlan(visible, torch.zeros_like(visible)))
            return encoded
        patches = []
        for group in self.layout.groups:
            if group.level == 0:
                values = view['patches'][group.key]
                patches.append(self.backbone.patch_encoders[group.key](values.flatten(0, 1)).reshape(len(values), -1,
                                                                                                 self.cfg['model']['embed_dim']))
        return torch.cat(patches, dim=1)

    def features(self, view):
        roots, candidates = self.selector.roots, self.selector.candidates
        valid = view['valid_mask']
        if not valid[:, roots].all() or (valid.sum(1) < self.budget).any():
            raise ValueError('Learned selection requires valid root tokens and enough tokens for its budget')
        scores = self.selector(self.root_features(view), view)
        k = self.budget-len(roots)
        eligible = valid[:, candidates]
        training = self.training and self.train_selector
        if training:
            temperature, noise = schedule(self.options, self.epoch)
            matrix, local_ids = relaxed_topk(scores[:, candidates], eligible, k, temperature, noise)
        else:
            local_ids = scores[:, candidates].masked_fill(~eligible, -torch.inf).argsort(
                dim=1, descending=True, stable=True)[:, :k]
        ids = candidates[local_ids]
        visible = torch.zeros_like(valid)
        visible[:, roots] = True
        visible.scatter_(1, ids, True)
        if training:
            # Frozen backbone PARAMETERS still allow autograd through the selector's
            # matrix. Do not wrap the transformer below in torch.no_grad().
            embeddings = self.backbone.token_embeddings(view)
            chosen = matrix @ embeddings[:, candidates]
            packed = torch.cat([embeddings[:, roots], chosen], dim=1)
            features = self.backbone.encoder(packed).mean(1)
        else:
            features = self.backbone.features(view, TokenPlan(visible, torch.zeros_like(visible)))
        with torch.no_grad():
            activity_ids = view['log_counts'][:, candidates].masked_fill(~eligible, -torch.inf).argsort(
                dim=1, descending=True, stable=True)[:, :k]
            baseline = torch.zeros_like(valid)
            baseline[:, roots] = True
            baseline.scatter_(1, candidates[activity_ids], True)
            self.last_diagnostics = dict(
                token_frequency=visible.float().sum(0),
                tokens_by_level=torch.stack([visible[:, self.selector.levels == i].sum() for i in range(len(self.cfg['hierarchy']['levels']))]),
                tokens_by_representation=torch.stack([visible[:, self.selector.planes == i].sum() for i in range(len(self.layout.representations))]),
                activity_overlap=(visible & baseline).sum().float()/self.budget)
        return features
