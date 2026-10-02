from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .data import Layout
from .masking import make_plan


def transformer(dim, heads, layers):
    return nn.TransformerEncoder(nn.TransformerEncoderLayer(dim, heads, dim*4, dropout=0.,
        activation='gelu', batch_first=True, norm_first=True), layers,
        norm=nn.LayerNorm(dim), enable_nested_tensor=False)


class GeometryEncoding(nn.Module):
    def __init__(self, layout, dim):
        super().__init__()
        self.origin = nn.ModuleList([nn.Embedding(n+1, dim) for n in layout.maximum])
        self.extent = nn.ModuleList([nn.Embedding(n+1, dim) for n in layout.maximum])
        self.level = nn.Embedding(int(layout.level_ids.max())+1, dim)
        self.plane = nn.Embedding(len(layout.representations), dim)
        self.time_geometry = nn.Sequential(nn.Linear(9, dim), nn.GELU(), nn.Linear(dim, dim))
        self.register_buffer('boxes', layout.boxes)
        self.register_buffer('levels', layout.level_ids)
        self.register_buffer('planes', layout.plane_ids)

    def forward(self, metadata):
        out = self.level(self.levels) + self.plane(self.planes)
        for axis in range(3):
            out = out + self.origin[axis](self.boxes[:, axis]) + self.extent[axis](self.boxes[:, axis+3])
        return out[None] + self.time_geometry(metadata)


class HierarchicalMAE(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg, self.layout = cfg, Layout(cfg)
        m = cfg['model']
        dim, dec = m['embed_dim'], m['decoder_dim']
        self.geometry = GeometryEncoding(self.layout, dim)
        self.patch_encoders = nn.ModuleDict({g.key: nn.Sequential(
            nn.Flatten(1), nn.Linear(g.channels*g.size*g.size, dim), nn.LayerNorm(dim))
            for g in self.layout.groups})
        self.count_encoder = nn.Linear(1, dim)
        self.encoder = transformer(dim, m['num_heads'], m['num_layers'])
        self.decoder_geometry = GeometryEncoding(self.layout, dec)
        self.decoder_projection = nn.Linear(dim, dec)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dec))
        nn.init.normal_(self.mask_token, std=.02)
        self.decoder = transformer(dec, m['num_heads'], m['decoder_layers'])
        self.patch_decoders = nn.ModuleDict({g.key: nn.Linear(dec, g.channels*g.size*g.size)
                                           for g in self.layout.groups})
        self.count_decoder = nn.Linear(dec, 1)

    def encode(self, view, plan):
        plan.validate(view['valid_mask'])
        batch, _ = plan.visible.shape
        geometry = self.geometry(view['metadata'])
        # Encode only visible patches: neither masked pixels nor counts enter the encoder.
        content = torch.zeros_like(geometry)
        for g in self.layout.groups:
            local = plan.visible[:, g.start:g.stop]
            if local.any():
                b, i = local.nonzero(as_tuple=True)
                content[b, i+g.start] = self.patch_encoders[g.key](view['patches'][g.key][local])
        b, i = plan.visible.nonzero(as_tuple=True)
        content[b, i] = content[b, i] + self.count_encoder(view['log_counts'][b, i, None])
        values = content + geometry
        if plan.gates is not None:
            values = values * plan.gates[..., None]
        lengths = plan.visible.sum(1)
        capacity = int(lengths.max())
        # Stable gather preserves canonical token IDs, independent of selection rank.
        ids = (~plan.visible).int().argsort(dim=1, stable=True)[:, :capacity]
        padding = torch.arange(capacity, device=ids.device)[None] >= lengths[:, None]
        packed = values.gather(1, ids[..., None].expand(-1, -1, values.shape[-1]))
        encoded = self.encoder(packed, src_key_padding_mask=padding)
        return encoded, ids, padding

    def features(self, view, plan=None, selection=None):
        if plan is None:
            plan = make_plan(view, self.layout, selection=selection or self.cfg['downstream'].get('selection', {}))
        encoded, ids, padding = self.encode(view, plan)
        weights = (~padding).to(encoded.dtype)
        if plan.gates is not None:
            weights = weights * plan.gates.gather(1, ids)
        return (encoded * weights[..., None]).sum(1) / weights.sum(1, keepdim=True).clamp_min(1e-6)

    def forward(self, view, plan=None):
        if plan is None:
            plan = make_plan(view, self.layout, self.cfg['masking'], self.cfg.get('selection'))
        if not plan.target.any(1).all():
            raise ValueError('MAE requires reconstruction targets in every example')
        encoded, ids, padding = self.encode(view, plan)
        dec = self.mask_token.expand(len(encoded), self.layout.count, -1).clone()
        batch = torch.arange(len(encoded), device=encoded.device)[:, None].expand_as(ids)
        dec[batch[~padding], ids[~padding]] = self.decoder_projection(encoded)[~padding]
        dec = self.decoder(dec + self.decoder_geometry(view['metadata']),
                           src_key_padding_mask=~view['valid_mask'])
        predictions = {}
        # Normalize pixels per token, then balance the loss across active level/representation groups.
        group_losses = []
        for g in self.layout.groups:
            pred = self.patch_decoders[g.key](dec[:, g.start:g.stop]).reshape(
                len(dec), g.stop-g.start, g.channels, g.size, g.size)
            predictions[g.key] = pred
            error = (pred - view['patches'][g.key]).square().mean((2, 3, 4))
            target = plan.target[:, g.start:g.stop]
            if target.any():
                group_losses.append(error[target].mean())
        patch_loss = torch.stack(group_losses).mean()
        count_pred = self.count_decoder(dec).squeeze(-1)
        count_loss = F.mse_loss(count_pred[plan.target], view['log_counts'][plan.target])
        loss = patch_loss + self.cfg['loss']['count_weight'] * count_loss
        return dict(loss=loss, patch_loss=patch_loss, count_loss=count_loss,
                    predictions=predictions, count_predictions=count_pred, plan=plan,
                    visible_tokens=plan.visible.sum(1).float().mean())

    def encoder_parameters(self):
        for module in (self.geometry, self.patch_encoders, self.count_encoder, self.encoder):
            yield from module.parameters()
