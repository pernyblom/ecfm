"""Paired content-ablation metrics; low error ratios suggest metadata shortcuts."""
import torch


def feature_diagnostic(full, blank):
    full, blank = full.double(), blank.double()
    variance = full.var(0, unbiased=False).mean()
    mse = (full - blank).square().mean()
    return dict(feature_mse=float(mse), feature_variance=float(variance),
                mse_over_variance=float(mse / variance) if variance > 1e-12 else None,
                full_std=float(full.std(0, unbiased=False).mean()),
                blank_std=float(blank.std(0, unbiased=False).mean()))


@torch.no_grad()
def probe_blank_diagnostic(model, loader, device):
    from .train import to_device
    model.eval()
    full, blank = [], []
    correct = blank_correct = agreement = count = 0
    for raw in loader:
        batch = to_device(raw, device)
        if 'features' in batch:
            z, b = batch['features'], batch['blank_features']
        else:
            view = batch['source']
            z = model.backbone.features(view)
            b = model.backbone.features(dict(view, patches=torch.zeros_like(view['patches'])))
        prediction, blank_prediction = model.head(z).argmax(-1), model.head(b).argmax(-1)
        correct += int((prediction == batch['label']).sum())
        blank_correct += int((blank_prediction == batch['label']).sum())
        agreement += int((prediction == blank_prediction).sum())
        count += len(z)
        full.append(z.cpu())
        blank.append(b.cpu())
    return dict(**feature_diagnostic(torch.cat(full), torch.cat(blank)), samples=count,
                accuracy=correct / count, blank_accuracy=blank_correct / count,
                prediction_agreement=agreement / count)
