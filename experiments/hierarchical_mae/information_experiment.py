"""Compare raw-voxel information selectors using independent frozen-encoder probes."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from experiments.region_worldmodel.feature_cache import file_digest
from .config import load_config, validate
from .downstream import run as downstream_run
from .data import Layout, splits
from .information_selection import METRICS


def variants(budget, metrics=METRICS, weights=(.25, .5, .75)):
    result = {'all': dict(strategy='all', budget=0),
              'activity': dict(strategy='activity', budget=budget),
              'coarse': dict(strategy='coarse', budget=budget)}
    for metric in metrics:
        result[metric] = dict(strategy='information', metric=metric, budget=budget)
        for weight in weights:
            result[f'{metric}_blend{weight}'] = dict(strategy='activity_information', metric=metric,
                                                      budget=budget, combination='blend', activity_weight=weight)
        result[f'{metric}_product'] = dict(strategy='activity_information', metric=metric,
                                          budget=budget, combination='product')
    return result


def summarize(records):
    summary = {}
    for name in sorted({r['variant'] for r in records}):
        group = [r for r in records if r['variant'] == name]
        summary[name] = {'seeds': [r['seed'] for r in group], 'selection': group[0]['result']['selection']}
        for split in ('validation', 'test'):
            summary[name][split] = {}
            for metric in ('loss', 'accuracy'):
                values = [r['result'][split][metric] for r in group]
                summary[name][split][metric] = dict(mean=float(np.mean(values)),
                                                   std=float(np.std(values)), values=values)
    return summary


def write_json(path, value):
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, suffix='.tmp')
    try:
        with os.fdopen(descriptor, 'w') as stream:
            json.dump(value, stream, indent=2)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def run_experiment(cfg, checkpoint, output_dir, seeds=(7, 17, 27), budget=54,
                   metrics=METRICS, weights=(.25, .5, .75), dry_run=False):
    if type(budget) is not int or budget < 1:
        raise ValueError('Comparison budget must be positive')
    if not seeds or len(set(seeds)) != len(seeds) or any(type(s) is not int or s < 0 for s in seeds):
        raise ValueError('Seeds must be distinct nonnegative integers')
    if not metrics or len(set(metrics)) != len(metrics) or set(metrics) - set(METRICS):
        raise ValueError('Choose distinct supported information metrics')
    if len(set(weights)) != len(weights) or any(not np.isfinite(w) or not 0 < w < 1 for w in weights):
        raise ValueError('Blend weights must be distinct and strictly between 0 and 1')
    cfg = deepcopy(cfg)
    # Compute once for every variant so the shared patch cache is reusable.
    cfg['data'].setdefault('information_selection', {'bins': [8, 8, 8], 'support_saturation': 3.})
    validate(cfg)
    if budget > Layout(cfg).count:
        raise ValueError('Comparison budget exceeds the number of candidate tokens')
    selections = variants(budget, metrics, weights)
    plan = dict(config=cfg, checkpoint=str(Path(checkpoint).resolve()),
                seeds=list(seeds), selections=selections)
    if dry_run:
        print(json.dumps(plan, indent=2))
        return plan
    output = Path(output_dir)
    plan['checkpoint_digest'] = file_digest(checkpoint)
    plan['implementation'] = {p.name: file_digest(p) for p in sorted(Path(__file__).parent.glob('*.py'))}
    plan['entries'] = {
        name: [[str(p.resolve()), label, p.stat().st_size, p.stat().st_mtime_ns, p.stat().st_ctime_ns]
               for p, label in entries]
        for name, entries in zip(('train', 'validation', 'test'), splits(cfg, include_test=True))
    }
    manifest = output / 'experiment.json'
    if manifest.exists():
        if json.loads(manifest.read_text()) != plan:
            raise ValueError('Existing experiment configuration/provenance differs; use a new output directory')
    else:
        if output.exists() and any(output.iterdir()):
            raise ValueError('Experiment output directory is nonempty and has no manifest')
        output.mkdir(parents=True, exist_ok=True)
        write_json(manifest, plan)
    records = []
    for name, selection in selections.items():
        for seed in seeds:
            current = deepcopy(cfg)
            current['train']['seed'] = seed
            current['downstream']['selection'] = selection
            validate(current)
            directory = output / f'{name}_seed{seed}'
            result_path = directory / 'results.json'
            if result_path.exists():
                result = json.loads(result_path.read_text())
            else:
                last = directory / 'last.pt'
                result = downstream_run(current, checkpoint, 'linear_probe', directory,
                                        resume=last if last.exists() else None)
            records.append(dict(variant=name, seed=seed, result=result))
            # Save progress after each completed probe, allowing interrupted suites to continue.
            write_json(output / 'results.json', dict(runs=records, summary=summarize(records)))
    return dict(runs=records, summary=summarize(records))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='experiments/hierarchical_mae/configs/thu_linear_probe.yaml')
    parser.add_argument('--checkpoint', required=True, help='Pretrained MAE checkpoint')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--budget', type=int, default=54)
    parser.add_argument('--seeds', nargs='+', type=int, default=[7, 17, 27])
    parser.add_argument('--metrics', nargs='+', choices=METRICS, default=list(METRICS))
    parser.add_argument('--weights', nargs='*', type=float, default=[.25, .5, .75])
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--device')
    parser.add_argument('--workers', type=int)
    parser.add_argument('--max-batches', type=int, help='Bound train and validation batches for pipeline checks')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg['downstream']['epochs'] = args.epochs
    if args.device is not None:
        cfg['train']['device'] = args.device
    if args.workers is not None:
        cfg['train']['num_workers'] = args.workers
    if args.max_batches is not None:
        if args.max_batches < 0:
            parser.error('--max-batches must be nonnegative')
        cfg['downstream'].update(max_batches=args.max_batches, max_val_batches=args.max_batches)
    run_experiment(cfg, args.checkpoint, args.output_dir, args.seeds, args.budget,
                   args.metrics, args.weights, args.dry_run)


if __name__ == '__main__':
    main()
