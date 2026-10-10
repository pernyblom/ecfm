"""Expose hierarchical MAE token features through the existing task backbone API."""
from copy import deepcopy

import torch
from torch import nn

from .config import load_config
from .masking import make_plan
from .model import HierarchicalMAE


def backbone_config(options):
    if options.get('mae_config'):
        return load_config(options['mae_config'])
    if options.get('mae_cfg'):
        return deepcopy(options['mae_cfg'])
    if options.get('checkpoint'):
        return torch.load(options['checkpoint'], map_location='cpu', weights_only=True)['config']
    raise ValueError('hierarchical_mae backbone requires mae_config, mae_cfg or a pretraining checkpoint')


class HierarchicalBackbone(nn.Module):
    def __init__(self, options):
        super().__init__()
        cfg = backbone_config(options)
        self.model = HierarchicalMAE(cfg)
        if options.get('checkpoint'):
            state = torch.load(options['checkpoint'], map_location='cpu', weights_only=True)
            if state['config']['hierarchy'] != cfg['hierarchy'] or state['config']['model'] != cfg['model']:
                raise ValueError('MAE checkpoint architecture differs from configured hierarchy/model')
            self.model.load_state_dict(state['model'], strict=True)
        self.fmap_dim = cfg['model']['embed_dim']
        self.out_dim = int(options.get('out_dim', 128))
        self.projection = nn.Linear(self.fmap_dim, self.out_dim) if self.out_dim != self.fmap_dim else nn.Identity()
        self.selection = dict(options.get('selection', {'strategy': 'all'}))
        self.frozen = bool(options.get('freeze', False))
        # Reconstruction modules are loaded for checkpoint compatibility but
        # are never optimized by downstream heads.
        self.model.requires_grad_(False)
        if not self.frozen:
            for parameter in self.model.encoder_parameters():
                parameter.requires_grad_(True)
        if self.frozen:
            self.model.eval()
        self.level = int(options.get('fmap_level', len(cfg['hierarchy']['levels'])-1))
        if not 0 <= self.level < len(cfg['hierarchy']['levels']):
            raise ValueError('Invalid fmap_level')
        self.nx, self.ny, self.nt = cfg['hierarchy']['levels'][self.level]['splits']

    def train(self, mode=True):
        super().train(mode)
        if self.frozen:
            self.model.eval()
        return self

    def forward(self, view):
        from experiments.object_detection.models.backbones import EncoderOutput
        if not isinstance(view, dict) or 'patches' not in view:
            raise ValueError('HierarchicalBackbone expects a batched hierarchy view, not an image tensor')
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.frozen):
            plan = make_plan(view, self.model.layout, selection=self.selection)
            encoded, ids, padding = self.model.encode(view, plan)
            weights = (~padding).to(encoded.dtype)
            if plan.gates is not None:
                weights = weights*plan.gates.gather(1, ids)
            pooled = (encoded*weights[..., None]).sum(1)/weights.sum(1, keepdim=True).clamp_min(1e-6)
            batch, _, dim = encoded.shape
            canonical = encoded.new_zeros(batch, self.model.layout.count, dim).scatter_add(
                1, ids[..., None].expand(-1, -1, dim), encoded*weights[..., None])
            coverage = weights.new_zeros(batch, self.model.layout.count).scatter_add(1, ids, weights)
            # Local spatial cells are in t,y,x order. Average time and alternate
            # projections, preserving the per-cell transformer content for CenterNet.
            total = encoded.new_zeros(batch, self.ny, self.nx, dim)
            counts = weights.new_zeros(batch, self.ny, self.nx)
            for group in self.model.layout.groups:
                if group.level != self.level:
                    continue
                total = total+canonical[:, group.start:group.stop].reshape(batch, self.nt, self.ny, self.nx, dim).sum(1)
                counts = counts+coverage[:, group.start:group.stop].reshape(batch, self.nt, self.ny, self.nx).sum(1)
            fmap = (total/counts[..., None].clamp_min(1)).permute(0, 3, 1, 2)
        return EncoderOutput(fmap=fmap, pooled=self.projection(pooled))
