from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from ecfm.models.mae import EventMAE


def pool(tokens, valid):
    weights = valid.to(tokens.dtype).unsqueeze(-1)
    return (tokens * weights).sum(1) / weights.sum(1).clamp_min(1)


class RegionWorldModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        m, d = cfg['model'], cfg['data']
        dim = m['embed_dim']
        self.encoder = EventMAE(
            patch_size=m['patch_size'], embed_dim=dim,
            num_heads=m['num_heads'], num_layers=m['num_layers'],
            decoder_embed_dim=m.get('decoder_embed_dim', dim),
            decoder_num_heads=m['num_heads'], decoder_num_layers=1,
            mlp_ratio=4, plane_embed_dim=32, metadata_dim=9,
            num_tokens=max(d['num_regions_choices']), num_planes=len(d['plane_types']),
            use_pos_embedding=False, use_relative_bias=True)
        self.predictor = nn.Sequential(nn.Linear(dim + 15, dim * 2), nn.GELU(),
                                       nn.Linear(dim * 2, dim))

    def encode(self, view, mask=None):
        return self.encoder.encode(**view, mask=mask)

    def features(self, view):
        return pool(self.encode(view), view['valid_mask'])

    def predict(self, tokens, action):
        action = action[:, None, :].expand(-1, tokens.shape[1], -1)
        return tokens + self.predictor(torch.cat([tokens, action], dim=-1))

    def forward(self, batch, mask=None):
        source = self.encode(batch['source'], mask)
        target = self.encode(batch['target'])
        return source, target, self.predict(source, batch['action'])


def sigreg(z, projections=64, integration_steps=17):
    """Sketched Gaussian characteristic-function matching, on batch samples.

    Kept local so this experiment has no dependency on the FRED experiment.
    This is a SIGReg-style numerical integral, not an exact paper reproduction.
    """
    directions = F.normalize(torch.randn(z.shape[-1], projections, device=z.device), dim=0)
    t = torch.linspace(0, 3, integration_steps, device=z.device)
    h = (z.float() @ directions).unsqueeze(-1) * t
    gaussian = torch.exp(-t.square() / 2)
    error = (h.cos().mean(0) - gaussian).square() + h.sin().mean(0).square()
    return torch.trapezoid(error * gaussian, t, dim=-1).mean()


def objective(model, batch, cfg, mask=None):
    source, target, pred = model(batch, mask)
    valid = batch['source']['valid_mask'] & batch['target']['valid_mask']
    prediction = (pred - target).square().mean(-1)[valid].mean()
    reg = cfg['loss']
    # Regularize both views across independent recordings, not tokens treated
    # as independent examples. Neither target nor source is detached.
    regularizer = (sigreg(pool(source, valid), reg['projections'], reg['integration_steps']) +
                   sigreg(pool(target, valid), reg['projections'], reg['integration_steps'])) / 2
    reconstruction = prediction.new_zeros(())
    if reg.get('reconstruction_weight', 0) > 0 and mask is not None and mask.any():
        reconstructed, _, _ = model.encoder(**batch['source'], mask=mask)
        reconstruction = (reconstructed - batch['source']['patches']).square().mean((2, 3, 4))[mask].mean()
    loss = prediction + reg['regularizer_weight'] * regularizer + reg.get('reconstruction_weight', 0) * reconstruction
    return loss, dict(prediction=prediction, regularizer=regularizer,
                      reconstruction=reconstruction), (source, target, pred)
