"""Train and evaluate conservative one-swap selection from an activity probe.

Example: python -m experiments.hierarchical_mae.swap_experiment --baseline
outputs/hierarchical_mae/linear_probe_activity54/best.pt --output-dir
outputs/hierarchical_mae/swap54
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time

import torch
from torch import nn
from torch.nn import functional as F

from experiments.region_worldmodel.train import seed_all
from .data import HierarchyDataset, splits
from .downstream import Classifier, save_checkpoint
from .feature_cache import cache_metadata, file_digest
from .loading import make_loader, to_device
from .model import HierarchicalMAE
from . import swap_selection
from .swap_selection import SwapPolicy, choose, proposals, policy_inputs, activity_coarse_overlap


def metrics(logits, labels):
    return dict(loss=F.cross_entropy(logits, labels).item(),
                accuracy=(logits.argmax(-1) == labels).float().mean().item(), samples=len(labels))


@torch.no_grad()
def bank(model, entries, cfg, args, split, device):
    """Cache exact alternative pooled features, not pixels or relaxed mixtures."""
    dataset = HierarchyDataset(cfg, entries)
    metadata = dict(base=cache_metadata(dataset, args.baseline), drops=args.drops, adds=args.adds,
                    implementation=[file_digest(__file__), file_digest(swap_selection.__file__)])
    digest = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
    path = Path(args.output_dir)/f'{split}_{digest[:16]}.pt'
    if path.exists():
        saved = torch.load(path, weights_only=True, map_location='cpu')
        if saved['metadata'] != metadata:
            raise ValueError('Swap bank metadata mismatch')
        print(f'{split}: cached {path}', flush=True)
        return saved['bank']
    loader = make_loader(dataset, args.extraction_batch, cfg['train']['num_workers'])
    chunks = {key: [] for key in ('features', 'coarse', 'local', 'labels', 'activity27_overlap', 'activity27_equal')}
    start = time.monotonic()
    for step, raw in enumerate(loader):
        view = to_device(raw['source'], device)
        ids, removed, inserted = proposals(view['log_counts'], view['valid_mask'], args.budget, args.drops, args.adds)
        coarse, local = policy_inputs(model.backbone, view, removed, inserted)
        embeddings = model.backbone.token_embeddings(view)
        b, alternatives, k = ids.shape
        packed = embeddings.gather(1, ids.flatten(1)[..., None].expand(-1, -1, embeddings.shape[-1]))
        packed = packed.reshape(b*alternatives, k, -1)
        # Match MAE.features: canonical IDs and an explicit padding mask.
        encoded = []
        for part in packed.split(args.encoder_batch):
            padding = torch.zeros(part.shape[:2], dtype=torch.bool, device=device)
            encoded.append(model.backbone.encoder(part, src_key_padding_mask=padding).mean(1))
        features = torch.cat(encoded).reshape(b, alternatives, -1)
        overlap, identical = activity_coarse_overlap(view['log_counts'], view['valid_mask'], model.backbone.layout)
        values = [features, coarse, local, raw['label'], overlap, identical]
        for key, value in zip(chunks, values):
            chunks[key].append(value.cpu())
        if step % 10 == 0 or step+1 == len(loader):
            print(f'{split}: {min((step+1)*args.extraction_batch, len(dataset))}/{len(dataset)} '
                  f'({time.monotonic()-start:.1f}s)', flush=True)
    result = {k: torch.cat(v) for k, v in chunks.items()}
    if any(not torch.isfinite(v).all() for v in result.values()):
        raise FloatingPointError('Nonfinite swap bank')
    save_checkpoint(path, dict(metadata=metadata, bank=result))
    return result


def selected(bank, choices):
    return bank['features'][torch.arange(len(choices), device=choices.device), choices]


@torch.no_grad()
def evaluate_policy(policy, bank, head, inputs, threshold):
    scores = policy(bank[inputs])
    choices = choose(scores, threshold)
    result = metrics(head(selected(bank, choices)), bank['labels'])
    result['swap_rate'] = choices.ne(0).float().mean().item()
    return result, choices


def train_policy(train, val, head, args, inputs, device):
    seed_all(args.seed)
    policy = SwapPolicy(train[inputs].shape[-1], args.hidden).to(device)
    policy.fit_normalization(train[inputs])
    with torch.no_grad():
        logits = head(train['features'])
        losses = F.cross_entropy(logits.flatten(0, 1),
            train['labels'][:, None].expand(-1, logits.shape[1]).flatten(), reduction='none').reshape(logits.shape[:2])
        targets = losses[:, :1]-losses[:, 1:]
    # Unmodified activity is explicitly eligible to win validation selection.
    baseline = metrics(head(val['features'][:, 0]), val['labels'])
    best = dict(state=deepcopy(policy.state_dict()), threshold=0., epoch=-1, validation=dict(baseline, swap_rate=0.))
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.policy_lr, weight_decay=.01)
    thresholds = [0., .005, .01, .025, .05, .1, .2]
    history = []
    for epoch in range(args.policy_epochs):
        policy.train()
        permutation = torch.randperm(len(targets), device=device)
        total = 0.
        for idx in permutation.split(128):
            loss = F.mse_loss(policy(train[inputs][idx]), targets[idx])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 1.)
            optimizer.step()
            total += loss.item()*len(idx)
        policy.eval()
        with torch.no_grad():
            scores = policy(val[inputs])
            candidates = []
            for threshold in thresholds:
                choices = choose(scores, threshold)
                result = metrics(head(selected(val, choices)), val['labels'])
                result['swap_rate'] = choices.ne(0).float().mean().item()
                candidates.append((result['loss'], threshold, result))
            _, threshold, result = min(candidates, key=lambda x: x[0])
        if result['loss'] < best['validation']['loss']:
            best = dict(state=deepcopy(policy.state_dict()), threshold=threshold, epoch=epoch, validation=result)
        history.append(dict(epoch=epoch, mse=total/len(targets), threshold=threshold, validation=result))
        if epoch % 20 == 0:
            print(f'policy {inputs} {epoch}: {history[-1]}', flush=True)
    policy.load_state_dict(best['state'])
    return policy, best, history


def fit_probe(train_features, labels, val_features, val_labels, cfg, seed, device):
    """Same initialization, AdamW and shuffled batches for each compared policy."""
    seed_all(seed)
    # Preserve downstream.Classifier's head initialization after backbone construction.
    holder = Classifier(HierarchicalMAE(cfg), cfg['downstream']['num_classes'], True)
    head = holder.head.to(device)
    settings = cfg['downstream']
    optimizer = torch.optim.AdamW(head.parameters(), lr=settings['lr'], weight_decay=settings['weight_decay'])
    from torch.utils.data import TensorDataset
    loader = make_loader(TensorDataset(train_features.cpu(), labels.cpu()), settings['batch_size'],
                         shuffle=True, seed=seed)
    best, best_epoch, state = float('inf'), -1, None
    for epoch in range(settings['epochs']):
        seed_all(seed+epoch)
        for x, y in loader:
            loss = F.cross_entropy(head(x.to(device)), y.to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            result = metrics(head(val_features), val_labels)
        if result['loss'] < best:
            best, best_epoch, state = result['loss'], epoch, deepcopy(head.state_dict())
    head.load_state_dict(state)
    return head, dict(epoch=best_epoch, validation=metrics(head(val_features), val_labels))


def run(args):
    torch.set_num_threads(4)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if (output/'results.json').exists():
        raise ValueError('Completed output exists; choose a new output directory')
    source = torch.load(args.baseline, map_location='cpu', weights_only=True)
    cfg = deepcopy(source['config'])
    if source['mode'] != 'linear_probe' or cfg['downstream']['selection'] != dict(strategy='activity', budget=args.budget):
        raise ValueError('Requires a trained activity probe at the requested budget')
    if cfg['downstream'].get('max_batches', 0) or cfg['downstream'].get('max_test_samples', 0):
        raise ValueError('Use a full-data baseline')
    cfg['train']['device'] = args.device
    device = torch.device(args.device)
    seed_all(args.seed)
    model = Classifier(HierarchicalMAE(cfg), cfg['downstream']['num_classes'], True).to(device)
    model.load_state_dict(source['model'])
    model.eval().requires_grad_(False)
    train_entries, val_entries, test_entries = splits(cfg, include_test=True)
    train = to_device(bank(model, train_entries, cfg, args, 'train', device), device)
    val = to_device(bank(model, val_entries, cfg, args, 'validation', device), device)
    baseline_val = metrics(model.head(val['features'][:, 0]), val['labels'])
    if abs(baseline_val['loss']-source['validation']['loss']) > 1e-4:
        raise ValueError(f'Baseline failed reproduction: {baseline_val} vs {source["validation"]}')
    report = dict(arguments=vars(args), baseline_checkpoint_digest=file_digest(args.baseline),
                  pretrained_digest=source.get('pretrained_digest'), baseline_validation=baseline_val,
                  activity27_coarse27={s: dict(mean_overlap=b['activity27_overlap'].mean().item(),
                                              identical_fraction=b['activity27_equal'].mean().item())
                                      for s, b in [('train', train), ('validation', val)]}, policies={})
    policies = {}
    for inputs in ('coarse', 'local'):
        policy, best, history = train_policy(train, val, model.head, args, inputs, device)
        report['policies'][inputs] = {k: v for k, v in best.items() if k != 'state'}
        save_checkpoint(output/f'policy_{inputs}.pt', dict(policy=policy.state_dict(), input_dim=train[inputs].shape[-1],
            hidden=args.hidden, inputs=inputs, threshold=best['threshold'], budget=args.budget, drops=args.drops,
            adds=args.adds, baseline=str(Path(args.baseline).resolve()), baseline_digest=file_digest(args.baseline),
            config=cfg, epoch=best['epoch'], validation=best['validation']))
        (output/f'policy_{inputs}_history.json').write_text(json.dumps(history, indent=2))
        policies[inputs] = (policy, best['threshold'])
    # Lock policy/threshold decisions before accessing test examples or labels.
    (output/'validation_selection.json').write_text(json.dumps(report, indent=2))
    probe_runs = {}
    for name in ('activity', 'coarse', 'local'):
        if name == 'activity':
            train_x, val_x = train['features'][:, 0], val['features'][:, 0]
        else:
            policy, threshold = policies[name]
            _, train_choices = evaluate_policy(policy, train, model.head, name, threshold)
            _, val_choices = evaluate_policy(policy, val, model.head, name, threshold)
            train_x, val_x = selected(train, train_choices), selected(val, val_choices)
        probe_runs[name] = []
        for seed in args.probe_seeds:
            head, info = fit_probe(train_x, train['labels'], val_x, val['labels'], cfg, seed, device)
            save_checkpoint(output/f'probe_{name}_{seed}.pt', dict(head=head.state_dict(), seed=seed, **info))
            probe_runs[name].append((head, dict(seed=seed, **info)))
            print(f'probe {name} seed={seed}: {info}', flush=True)
    test = to_device(bank(model, test_entries, cfg, args, 'test', device), device)
    report['baseline_test'] = metrics(model.head(test['features'][:, 0]), test['labels'])
    report['probes'] = {}
    for name, runs in probe_runs.items():
        if name == 'activity':
            test_x = test['features'][:, 0]
        else:
            policy, threshold = policies[name]
            result, choices = evaluate_policy(policy, test, model.head, name, threshold)
            report['policies'][name]['test_fixed_head'] = result
            test_x = selected(test, choices)
        report['probes'][name] = [dict(info, test=metrics(head(test_x), test['labels'])) for head, info in runs]
    # Oracle is a labelled diagnostic bound within this proposal bank, never a deployable policy.
    report['oracle_diagnostic'] = {}
    with torch.no_grad():
        for split, values in [('train', train), ('validation', val), ('test', test)]:
            logits = model.head(values['features'])
            losses = F.cross_entropy(logits.flatten(0, 1), values['labels'][:, None].expand(-1, logits.shape[1]).flatten(),
                                     reduction='none').reshape(logits.shape[:2])
            ids = losses.argmin(1)
            report['oracle_diagnostic'][split] = dict(
                metrics(logits[torch.arange(len(ids), device=device), ids], values['labels']),
                mean_loss_gain=(losses[:, 0]-losses.min(1).values).mean().item(),
                beneficial_fraction=(losses[:, 1:].min(1).values < losses[:, 0]).float().mean().item())
    (output/'results.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--budget', type=int, default=54)
    parser.add_argument('--drops', type=int, default=4)
    parser.add_argument('--adds', type=int, default=6)
    parser.add_argument('--extraction-batch', type=int, default=8)
    parser.add_argument('--encoder-batch', type=int, default=64)
    parser.add_argument('--policy-epochs', type=int, default=100)
    parser.add_argument('--policy-lr', type=float, default=.001)
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--probe-seeds', type=int, nargs='+', default=[7, 17, 27])
    args = parser.parse_args()
    if args.policy_epochs < 1 or args.extraction_batch < 1 or args.encoder_batch < 1:
        parser.error('Epochs and batch sizes must be positive')
    run(args)


if __name__ == '__main__':
    main()
