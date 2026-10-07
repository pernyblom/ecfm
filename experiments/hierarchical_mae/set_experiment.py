"""Cached supervised set selection at hard swap limits, followed by fresh probes."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from experiments.region_worldmodel.train import seed_all
from .data import HierarchyDataset, splits
from .downstream import Classifier, save_checkpoint
from .feature_cache import cache_metadata, file_digest
from .loading import make_loader, to_device
from .model import HierarchicalMAE
from . import set_selector, crop_selection
from .set_selector import SetPolicy, coarse_inputs, selected_features
from .swap_experiment import fit_probe, metrics
from .crop_selection import augmented_inputs, epoch_indices


@torch.no_grad()
def input_bank(model, entries, cfg, oracle, args, split):
    dataset = HierarchyDataset(cfg, entries)
    metadata = cache_metadata(dataset, args.baseline)
    oracle_file = oracle['splits'][split]['bank']
    saved = torch.load(oracle_file, map_location='cpu', weights_only=True)
    # Same checkpoint, observations, entries and implementation as target generation.
    if saved['metadata']['base'] != metadata:
        raise ValueError(f'{split}: oracle targets do not match current data/encoder provenance')
    options = oracle['arguments']
    fingerprint = dict(metadata=metadata, oracle_digest=file_digest(oracle_file),
                       implementation=file_digest(set_selector.__file__))
    key = hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()[:16]
    path = Path(args.output_dir)/f'inputs_{split}_{key}.pt'
    if path.exists():
        result = torch.load(path, weights_only=True, map_location='cpu')
        if result['fingerprint'] != fingerprint:
            raise ValueError('Input cache mismatch')
        return result['data']
    roots, descriptors, labels = [], [], []
    loader = make_loader(dataset, 32, cfg['train']['num_workers'])
    for raw in loader:
        view = to_device(raw['source'], args.device)
        root, desc = coarse_inputs(model.backbone, view, options['budget'], options['drops'], options['adds'])
        roots.append(root.cpu())
        descriptors.append(desc.cpu())
        labels.append(raw['label'])
    data = dict(roots=torch.cat(roots), descriptors=torch.cat(descriptors), labels=torch.cat(labels),
                losses=saved['bank']['losses'], predictions=saved['bank']['predictions'])
    if not torch.equal(data['labels'], saved['bank']['labels']):
        raise ValueError('Oracle label order differs')
    if any(not torch.isfinite(value).all() for value in data.values()):
        raise ValueError('Nonfinite cached input')
    save_checkpoint(path, dict(fingerprint=fingerprint, data=data))
    print(f'Inputs {split}: {len(data["labels"])} recordings', flush=True)
    return data


def scores_metrics(data, choices, policy):
    rows = torch.arange(len(choices), device=choices.device)
    return dict(loss=data['losses'][rows, choices].mean().item(),
        accuracy=(data['predictions'][rows, choices] == data['labels']).float().mean().item(),
        samples=len(choices), swap_rate=choices.ne(0).float().mean().item(),
        mean_swaps=policy.depths[choices].mean().item())


def train_policy(train, val, maximum, args, oracle):
    seed_all(args.seed)
    options = oracle['arguments']
    policy = SetPolicy(train['roots'].shape[-1], train['descriptors'].shape[-1], options['drops'],
                       options['adds'], maximum, args.hidden, args.dropout).to(args.device)
    policy.fit_normalization(train['roots'], train['descriptors'])
    n = len(policy.states)
    targets = train['losses'][:, :1]-train['losses'][:, 1:n]
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    zero = torch.zeros(len(val['labels']), dtype=torch.long, device=args.device)
    best = dict(policy=deepcopy(policy.state_dict()), threshold=0., epoch=-1, validation=scores_metrics(val, zero, policy))
    start = 0
    last_path = Path(args.output_dir)/f'last_swaps{maximum}.pt'
    # Deterministic epoch seeds allow exact continuation of this cached training.
    signature = dict(architecture=policy.settings, lr=args.lr, weight_decay=args.weight_decay,
                     seed=args.seed, baseline_digest=oracle['baseline_digest'],
                     augmentation=getattr(args, 'augmentation', None),
                     oracle_train_digest=file_digest(oracle['splits']['train']['bank']),
                     oracle_val_digest=file_digest(oracle['splits']['validation']['bank']),
                     implementation=[file_digest(__file__), file_digest(set_selector.__file__),
                                     file_digest(crop_selection.__file__)])
    if last_path.exists():
        saved = torch.load(last_path, map_location=args.device, weights_only=True)
        if saved['signature'] != signature:
            raise ValueError('Resume settings differ; use a new output directory')
        policy.load_state_dict(saved['policy'])
        optimizer.load_state_dict(saved['optimizer'])
        best, start = saved['best'], saved['epoch']+1
    history_path = Path(args.output_dir)/f'history_swaps{maximum}.jsonl'
    for epoch in range(start, args.epochs):
        seed_all(args.seed+epoch)
        policy.train()
        total = 0.
        indices = epoch_indices(train.get('records', len(targets)), train.get('views', 1), epoch,
                                args.seed, args.device)
        for ids in indices.split(128):
            predicted = policy(train['roots'][ids], train['descriptors'][ids])[:, 1:]
            loss = F.mse_loss(predicted, targets[ids])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.)
            optimizer.step()
            total += loss.item()*len(ids)
        policy.eval()
        with torch.no_grad():
            scores = policy(val['roots'], val['descriptors'])
            trials = [(threshold, scores_metrics(val, policy.choose(scores, threshold), policy))
                      for threshold in (0., .005, .01, .025, .05, .1, .2)]
            threshold, result = min(trials, key=lambda pair: pair[1]['loss'])
        if result['loss'] < best['validation']['loss']:
            best = dict(policy=deepcopy(policy.state_dict()), threshold=threshold, epoch=epoch, validation=result)
        record = dict(epoch=epoch, training_mse=total/len(indices), threshold=threshold, validation=result)
        with history_path.open('a') as stream:
            stream.write(json.dumps(record)+'\n')
        if epoch % 25 == 0 or epoch+1 == args.epochs:
            print(f'Swaps {maximum}: {record}', flush=True)
        save_checkpoint(last_path, dict(policy=policy.state_dict(), optimizer=optimizer.state_dict(), best=best,
                                        signature=signature, epoch=epoch))
    policy.load_state_dict(best['policy'])
    policy.eval()
    path = Path(args.output_dir)/f'policy_swaps{maximum}.pt'
    save_checkpoint(path, dict(**best, architecture=policy.settings, budget=options['budget'],
        baseline=str(Path(args.baseline).resolve()), baseline_digest=oracle['baseline_digest'],
        augmentation=getattr(args, 'augmentation', None),
        config=torch.load(args.baseline, weights_only=True, map_location='cpu')['config']))
    return policy, path, {k: v for k, v in best.items() if k != 'policy'}


@torch.no_grad()
def extract_features(model, entries, policy, threshold, cfg, args, split, name, policy_path):
    dataset = HierarchyDataset(cfg, entries)
    metadata = dict(base=cache_metadata(dataset, args.baseline), policy_digest=file_digest(policy_path) if policy else None,
                    implementation=file_digest(set_selector.__file__))
    key = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()[:16]
    path = Path(args.output_dir)/f'features_{name}_{split}_{key}.pt'
    if path.exists():
        saved = torch.load(path, map_location='cpu', weights_only=True)
        if saved['metadata'] != metadata:
            raise ValueError('Feature cache mismatch')
        return saved['features'], saved['labels']
    features, labels = [], []
    for raw in make_loader(dataset, 32, cfg['train']['num_workers']):
        view = to_device(raw['source'], args.device)
        if policy:
            result, _ = selected_features(model.backbone, view, policy, threshold, cfg['downstream']['selection']['budget'])
        else:
            result = model.backbone.features(view)
        features.append(result.cpu())
        labels.append(raw['label'])
    x, y = torch.cat(features), torch.cat(labels)
    save_checkpoint(path, dict(metadata=metadata, features=x, labels=y))
    return x, y


def run(args):
    torch.set_num_threads(4)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if (output/'results.json').exists():
        raise ValueError('Completed output exists; choose a new directory')
    oracle = json.loads(Path(args.oracle).read_text())
    if file_digest(args.baseline) != oracle['baseline_digest']:
        raise ValueError('Baseline differs from oracle')
    if any(m < 1 or m > oracle['arguments']['max_swaps'] for m in args.limits):
        raise ValueError('Swap limits exceed oracle coverage')
    source = torch.load(args.baseline, map_location='cpu', weights_only=True)
    cfg = deepcopy(source['config'])
    cfg['train']['device'] = args.device
    model = Classifier(HierarchicalMAE(cfg), cfg['downstream']['num_classes'], True).to(args.device)
    model.load_state_dict(source['model'])
    model.eval().requires_grad_(False)
    entries = dict(zip(('train', 'validation', 'test'), splits(cfg, include_test=True)))
    train = input_bank(model, entries['train'], cfg, oracle, args, 'train')
    if args.crop_views:
        train = augmented_inputs(model, entries['train'], cfg, oracle, args, train)
    train = to_device(train, args.device)
    val = to_device(input_bank(model, entries['validation'], cfg, oracle, args, 'validation'), args.device)
    report = dict(arguments=vars(args), baseline_digest=oracle['baseline_digest'], policies={}, probes={})
    policies = {'activity': (None, None, dict(threshold=0.))}
    for maximum in args.limits:
        policy, path, info = train_policy(train, val, maximum, args, oracle)
        name = f'swaps{maximum}'
        policies[name] = policy, path, info
        report['policies'][name] = info
    # Freeze all policies and thresholds before accessing test views or labels.
    (output/'validation_selection.json').write_text(json.dumps(report, indent=2))
    runs = {}
    for name, (policy, path, info) in policies.items():
        train_x, train_y = extract_features(model, entries['train'], policy, info['threshold'], cfg, args, 'train', name, path)
        val_x, val_y = extract_features(model, entries['validation'], policy, info['threshold'], cfg, args, 'validation', name, path)
        if policy:
            measured = metrics(model.head(val_x.to(args.device)), val_y.to(args.device))
            if abs(measured['loss']-info['validation']['loss']) > 1e-4:
                raise AssertionError('Live policy differs from cached oracle lookup')
        runs[name] = []
        for seed in args.probe_seeds:
            probe, probe_info = fit_probe(train_x.to(args.device), train_y.to(args.device),
                                          val_x.to(args.device), val_y.to(args.device), cfg, seed, args.device)
            save_checkpoint(output/f'probe_{name}_{seed}.pt', dict(head=probe.state_dict(), seed=seed, **probe_info,
                policy_digest=file_digest(path) if path else None, baseline_digest=oracle['baseline_digest']))
            runs[name].append((probe, dict(seed=seed, **probe_info)))
            print(f'Probe {name} seed {seed}: {probe_info}', flush=True)
    test = to_device(input_bank(model, entries['test'], cfg, oracle, args, 'test'), args.device)
    for name, (policy, path, info) in policies.items():
        x, y = extract_features(model, entries['test'], policy, info['threshold'], cfg, args, 'test', name, path)
        x, y = x.to(args.device), y.to(args.device)
        with torch.no_grad():
            report['probes'][name] = [dict(probe_info, test=metrics(probe(x), y)) for probe, probe_info in runs[name]]
            if policy:
                choices = policy.choose(policy(test['roots'], test['descriptors']), info['threshold'])
                fixed = scores_metrics(test, choices, policy)
                actual = metrics(model.head(x), y)
                if abs(actual['loss']-fixed['loss']) > 1e-4:
                    raise AssertionError('Live test policy differs from oracle bank')
                report['policies'][name]['test_fixed_head'] = fixed
            else:
                report['baseline_test'] = metrics(model.head(x), y)
    (output/'results.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--oracle', required=True, help='Completed swap_oracle results.json')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--limits', type=int, nargs='+', default=[1, 2, 4])
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=.1)
    parser.add_argument('--lr', type=float, default=.001)
    parser.add_argument('--weight-decay', type=float, default=.01)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--probe-seeds', type=int, nargs='+', default=[7, 17, 27])
    parser.add_argument('--crop-views', type=int, default=0,
                        help='Random training crops per recording, in addition to the original full view')
    parser.add_argument('--crop-cache-dir', default='outputs/hierarchical_mae_selector_crops')
    args = parser.parse_args()
    if args.epochs < 1 or len(set(args.limits)) != len(args.limits) or args.crop_views < 0:
        parser.error('Require positive epochs, unique swap limits, and nonnegative crop-views')
    run(args)
