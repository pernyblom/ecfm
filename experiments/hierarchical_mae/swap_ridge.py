"""Low-capacity one-swap control, reusing a completed swap_experiment bank."""
import argparse
from copy import deepcopy
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .downstream import save_checkpoint
from .feature_cache import file_digest
from .loading import to_device
from .swap_selection import choose
from .swap_experiment import selected, metrics, fit_probe


class RidgePolicy(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.register_buffer('mean', torch.zeros(dim))
        self.register_buffer('scale', torch.ones(dim))
        self.register_buffer('weight', torch.zeros(dim))
        self.register_buffer('bias', torch.zeros(()))

    def forward(self, x):
        return ((x-self.mean)/self.scale).clamp(-10, 10) @ self.weight + self.bias


def load_bank(source, split, digest):
    # Select by provenance, not filename ordering; an interrupted run can leave
    # more than one implementation version in the same directory.
    matches = []
    for path in source.glob(f'{split}_*.pt'):
        payload = torch.load(path, map_location='cpu', weights_only=True)
        if payload['metadata']['base']['checkpoint'] == digest:
            matches.append((path.stat().st_mtime_ns, payload['bank']))
    if not matches:
        raise ValueError(f'Missing matching {split} bank')
    return max(matches, key=lambda x: x[0])[1]


def run(args):
    torch.set_num_threads(4)
    source, output = Path(args.source), Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if (output/'results.json').exists():
        raise ValueError('Completed output exists')
    original = json.loads((source/'results.json').read_text())
    digest = original['baseline_checkpoint_digest']
    baseline_path = original['arguments']['baseline']
    if file_digest(baseline_path) != digest:
        raise ValueError('Baseline checkpoint changed')
    checkpoint = torch.load(baseline_path, map_location='cpu', weights_only=True)
    cfg = checkpoint['config']
    device = torch.device(args.device)
    train = to_device(load_bank(source, 'train', digest), device)
    val = to_device(load_bank(source, 'validation', digest), device)
    head = nn.Linear(cfg['model']['embed_dim'], cfg['downstream']['num_classes']).to(device)
    head.load_state_dict({k[5:]: v for k, v in checkpoint['model'].items() if k.startswith('head.')})
    head.eval().requires_grad_(False)
    with torch.no_grad():
        logits = head(train['features'])
        losses = F.cross_entropy(logits.flatten(0, 1), train['labels'][:, None].expand(-1, logits.shape[1]).flatten(),
                                 reduction='none').reshape(logits.shape[:2])
        target = (losses[:, :1]-losses[:, 1:]).flatten()
    baseline_val = metrics(head(val['features'][:, 0]), val['labels'])
    best = dict(validation=dict(baseline_val, swap_rate=0.), inputs='coarse', alpha=None, threshold=0.)
    winner = RidgePolicy(train['coarse'].shape[-1]).to(device)
    trials = []
    with torch.no_grad():
        for kind in ('coarse', 'local'):
            flat = train[kind].flatten(0, 1)
            policy = RidgePolicy(flat.shape[-1]).to(device)
            policy.mean.copy_(flat.mean(0))
            policy.scale.copy_(flat.std(0).clamp_min(.05))
            x = ((flat-policy.mean)/policy.scale).clamp(-10, 10)
            xmean, ymean = x.mean(0), target.mean()
            centered = x-xmean
            gram = centered.T @ centered/len(x)
            rhs = centered.T @ (target-ymean)/len(x)
            identity = torch.eye(x.shape[1], device=device)
            for alpha in (.001, .01, .1, 1., 10.):
                policy.weight.copy_(torch.linalg.solve(gram+alpha*identity, rhs))
                policy.bias.copy_(ymean-xmean @ policy.weight)
                scores = policy(val[kind])
                for threshold in (0., .005, .01, .025, .05, .1, .2):
                    choices = choose(scores, threshold)
                    result = metrics(head(selected(val, choices)), val['labels'])
                    result['swap_rate'] = choices.ne(0).float().mean().item()
                    trial = dict(inputs=kind, alpha=alpha, threshold=threshold, validation=result)
                    trials.append(trial)
                    if result['loss'] < best['validation']['loss']:
                        best, winner = trial, deepcopy(policy)
    report = dict(source=str(source), baseline_checkpoint_digest=digest, selection=best, trials=trials)
    (output/'validation_selection.json').write_text(json.dumps(report, indent=2))
    save_checkpoint(output/'policy.pt', dict(policy=winner.state_dict(), input_dim=winner.mean.numel(),
        kind='ridge', **best, baseline=str(Path(baseline_path).resolve()), baseline_digest=digest,
        budget=original['arguments']['budget'], drops=original['arguments']['drops'],
        adds=original['arguments']['adds'], config=cfg))
    with torch.no_grad():
        kind, threshold = best['inputs'], best['threshold']
        train_x = selected(train, choose(winner(train[kind]), threshold))
        val_x = selected(val, choose(winner(val[kind]), threshold))
    runs = []
    for seed in (7, 17, 27):
        probe, info = fit_probe(train_x, train['labels'], val_x, val['labels'], cfg, seed, device)
        save_checkpoint(output/f'probe_{seed}.pt', dict(head=probe.state_dict(), seed=seed, **info))
        runs.append((probe, dict(seed=seed, **info)))
        print(f'ridge probe {seed}: {info}', flush=True)
    # Selection above uses no test data, even though the preceding experiment
    # already reported its own results on this shared exploratory test set.
    test = to_device(load_bank(source, 'test', digest), device)
    with torch.no_grad():
        choices = choose(winner(test[kind]), threshold)
        test_x = selected(test, choices)
        report['test_fixed_head'] = dict(metrics(head(test_x), test['labels']),
                                       swap_rate=choices.ne(0).float().mean().item())
        report['probes'] = [dict(info, test=metrics(probe(test_x), test['labels'])) for probe, info in runs]
    (output/'results.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != 'trials'}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--device', default='cuda')
    run(parser.parse_args())
