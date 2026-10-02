"""Save a bounded patch contact sheet and the complete first-example tensors."""
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch


def save_inspection(output, epoch, view, result, layout, per_group=4):
    directory = Path(output) / 'patches'
    directory.mkdir(parents=True, exist_ok=True)
    # Full precision artifacts preserve negative predictions, all tokens, metadata and masks.
    payload = dict(patches={k: v[0].detach().cpu() for k, v in view['patches'].items()},
        predictions={k: v[0].detach().cpu() for k, v in result['predictions'].items()},
        metadata=view['metadata'][0].cpu(), log_counts=view['log_counts'][0].cpu(),
        count_predictions=result['count_predictions'][0].detach().cpu(),
        visible=result['plan'].visible[0].cpu(), target=result['plan'].target[0].cpu(),
        boxes=layout.boxes, groups=[vars(g) for g in layout.groups])
    torch.save(payload, directory / f'epoch_{epoch:04d}.pt')
    rows = []
    for g in layout.groups:
        # Put reconstruction targets first, but also show context if space remains.
        order = (~payload['target'][g.start:g.stop]).int().argsort(stable=True)[:per_group]
        rows.extend((g, int(i)) for i in order)
    canvas = Image.new('RGB', (650, 110*len(rows)), 'white')
    draw = ImageDraw.Draw(canvas)
    for row, (g, i) in enumerate(rows):
        idx = g.start+i
        status = 'target' if payload['target'][idx] else 'visible' if payload['visible'][idx] else 'excluded'
        seconds = float(torch.expm1(payload['metadata'][idx, 7]))
        draw.text((5, row*110+5), f'{g.key} #{i} {status}\nsize={g.size}, duration={seconds:.4g}s\nlog1p(count)={payload["log_counts"][idx]:.3f}\nGT / prediction', fill='black')
        for col, key in enumerate(('patches', 'predictions')):
            patch = payload[key][g.key][i].clamp(0, 1)
            if patch.shape[0] == 2:
                patch = torch.stack((patch[1], torch.zeros_like(patch[0]), patch[0]))
            image = Image.fromarray((patch.permute(1, 2, 0).numpy()*255).astype(np.uint8))
            canvas.paste(image.resize((96, 96), Image.Resampling.NEAREST), (420+col*110, row*110+5))
    canvas.save(directory / f'epoch_{epoch:04d}.png')
