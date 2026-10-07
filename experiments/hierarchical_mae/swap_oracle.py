"""Label-assisted multi-swap diagnostics with a fixed classifier and candidate pool.

Evaluates every distinct token set once, then simulates greedy and beam searches
on the cached losses. Exhaustive search is practical for the default 4x6 pool:
there are just 210 distinct sets including the unchanged activity selection.
"""
import argparse
from copy import deepcopy
import hashlib
from itertools import combinations
import json
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from .data import HierarchyDataset, splits
from .downstream import Classifier, save_checkpoint
from .feature_cache import cache_metadata, file_digest
from .loading import make_loader, to_device
from .model import HierarchicalMAE


def search_space(drops, adds, maximum):
    if min(drops, adds, maximum) < 1 or maximum > min(drops, adds):
        raise ValueError('Require 1 <= maximum <= min(drops, adds)')
    states = [((), ())]
    for depth in range(1, maximum+1):
        states.extend((d, a) for d in combinations(range(drops), depth)
                      for a in combinations(range(adds), depth))
    lookup = {state: i for i, state in enumerate(states)}
    children = []
    for removed, inserted in states:
        following = []
        if len(removed) < maximum:
            for d in range(drops):
                for a in range(adds):
                    if d not in removed and a not in inserted:
                        child = (tuple(sorted((*removed, d))), tuple(sorted((*inserted, a))))
                        following.append(lookup[child])
        children.append(sorted(set(following)))
    return states, children


def token_sets(activity, valid, budget, drops, adds, states):
    if budget < drops or not (valid.sum(1) >= budget+adds).all():
        raise ValueError('Insufficient tokens for the candidate pool')
    order = activity.masked_fill(~valid, -torch.inf).argsort(dim=1, descending=True, stable=True)
    baseline = order[:, :budget]
    # Same removal ordering and tie handling as the original one-swap experiment.
    removal_ids = baseline[:, -drops:].flip(1)
    addition_ids = order[:, budget:budget+adds]
    candidates = []
    for removed, inserted in states:
        values = baseline.clone()
        for d, a in zip(removed, inserted):
            values = torch.where(values == removal_ids[:, d, None], addition_ids[:, a, None], values)
        candidates.append(values.sort(1).values)
    return torch.stack(candidates, 1), baseline, removal_ids, addition_ids


def search(losses, states, children, maximum, beam_width=4, method='beam'):
    """Return best-so-far IDs and cumulative unique evaluations at each depth.

    Greedy stops unless a child strictly improves the loss. Beam expands exact
    depth frontiers even when they worsen the current best, allowing interacting
    swaps to overcome a local minimum. Outputs always retain earlier solutions.
    """
    values = losses.tolist()
    if method not in ('greedy', 'beam', 'exhaustive') or beam_width < 1:
        raise ValueError('Invalid search method or beam width')
    frontier, best, seen = [0], 0, {0}
    choices, counts = [0], [1]
    for depth in range(1, maximum+1):
        if method == 'exhaustive':
            candidates = [i for i, state in enumerate(states) if len(state[0]) == depth]
        else:
            candidates = sorted({j for i in frontier for j in children[i]})
        seen.update(candidates)
        candidates.sort(key=lambda i: (values[i], i))
        if candidates and values[candidates[0]] < values[best]:
            best = candidates[0]
            improved = True
        else:
            improved = False
        if method == 'greedy':
            frontier = [best] if improved else []
        else:
            frontier = candidates[:beam_width]
        choices.append(best)
        counts.append(len(seen))
    return choices, counts


@torch.no_grad()
def evaluate_bank(model, entries, args, states, split, device):
    dataset = HierarchyDataset(model.backbone.cfg, entries)
    metadata = dict(base=cache_metadata(dataset, args.baseline), drops=args.drops, adds=args.adds,
                    maximum=args.max_swaps, implementation=file_digest(__file__))
    key = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()[:16]
    path = Path(args.output_dir)/f'{split}_{key}.pt'
    if path.exists():
        payload = torch.load(path, map_location='cpu', weights_only=True)
        if payload['metadata'] != metadata:
            raise ValueError('Oracle cache metadata mismatch')
        print(f'{split}: cache hit {path}', flush=True)
        return payload['bank'], path
    loader = make_loader(dataset, args.batch_size, model.backbone.cfg['train']['num_workers'])
    chunks = {k: [] for k in ('losses', 'predictions', 'labels', 'baseline_ids', 'removal_ids', 'addition_ids')}
    start = time.monotonic()
    for step, raw in enumerate(loader):
        view = to_device(raw['source'], device)
        ids, baseline, removal, addition = token_sets(view['log_counts'], view['valid_mask'],
            args.budget, args.drops, args.adds, states)
        embeddings = model.backbone.token_embeddings(view)
        logits = []
        # Avoid materializing a whole batch x alternatives x tokens x dim tensor.
        flat_ids = ids.flatten(0, 1)
        sample_ids = torch.arange(len(ids), device=device).repeat_interleave(len(states))
        for offset in range(0, len(flat_ids), args.encoder_batch):
            sub_ids = flat_ids[offset:offset+args.encoder_batch]
            sub_samples = sample_ids[offset:offset+args.encoder_batch]
            packed = embeddings[sub_samples[:, None], sub_ids]
            padding = torch.zeros(packed.shape[:2], dtype=torch.bool, device=device)
            logits.append(model.head(model.backbone.encoder(packed, src_key_padding_mask=padding).mean(1)))
        logits = torch.cat(logits).reshape(len(ids), len(states), -1)
        labels = raw['label'].to(device)
        losses = F.cross_entropy(logits.flatten(0, 1), labels[:, None].expand(-1, len(states)).flatten(),
                                 reduction='none').reshape(len(ids), len(states))
        for k, v in zip(chunks, [losses, logits.argmax(-1), labels, baseline, removal, addition]):
            chunks[k].append(v.cpu())
        if step % 10 == 0 or step+1 == len(loader):
            print(f'{split}: {min((step+1)*args.batch_size, len(dataset))}/{len(dataset)} '
                  f'in {time.monotonic()-start:.1f}s', flush=True)
    bank = {k: torch.cat(v) for k, v in chunks.items()}
    if not torch.isfinite(bank['losses']).all():
        raise FloatingPointError('Nonfinite oracle losses')
    bank['extraction_seconds'] = time.monotonic()-start
    save_checkpoint(path, dict(metadata=metadata, bank=bank))
    return bank, path


def summarize(bank, states, children, maximum, beam_width):
    result, chosen = {}, {}
    for method in ('greedy', 'beam', 'exhaustive'):
        pairs = [search(row, states, children, maximum, beam_width, method) for row in bank['losses']]
        choices = torch.tensor([pair[0] for pair in pairs])
        evaluations = torch.tensor([pair[1] for pair in pairs])
        chosen[method] = choices
        depths = torch.tensor([len(s[0]) for s in states])
        rows = torch.arange(len(choices))
        result[method] = {}
        previous = bank['losses'][:, 0]
        for budget in range(maximum+1):
            ids = choices[:, budget]
            loss = bank['losses'][rows, ids]
            if (loss > previous+1e-6).any():
                raise AssertionError('Best-so-far loss increased')
            previous = loss
            result[method][str(budget)] = dict(loss=loss.mean().item(),
                accuracy=(bank['predictions'][rows, ids] == bank['labels']).float().mean().item(),
                samples=len(ids), mean_actual_swaps=depths[ids].float().mean().item(),
                mean_unique_evaluations=evaluations[:, budget].float().mean().item(),
                min_unique_evaluations=int(evaluations[:, budget].min()),
                max_unique_evaluations=int(evaluations[:, budget].max()))
    return result, chosen


def run(args):
    torch.set_num_threads(4)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if (output/'results.json').exists():
        raise ValueError('Completed output exists; use a new directory')
    states, children = search_space(args.drops, args.adds, args.max_swaps)
    if len(states) > args.max_sets:
        raise ValueError(f'{len(states)} sets exceeds --max-sets={args.max_sets}')
    source = torch.load(args.baseline, weights_only=True, map_location='cpu')
    cfg = deepcopy(source['config'])
    if source['mode'] != 'linear_probe' or cfg['downstream']['selection'] != dict(strategy='activity', budget=args.budget):
        raise ValueError('Requires an activity probe checkpoint at the same budget')
    cfg['train']['device'] = args.device
    model = Classifier(HierarchicalMAE(cfg), cfg['downstream']['num_classes'], True).to(args.device)
    model.load_state_dict(source['model'])
    model.eval().requires_grad_(False)
    report = dict(arguments=vars(args), label_assisted=True, objective='minimum cross-entropy',
                  baseline_digest=file_digest(args.baseline), pretrained_digest=source.get('pretrained_digest'),
                  distinct_sets=len(states), sets_by_exact_swaps={str(d): sum(len(s[0]) == d for s in states)
                                                                  for d in range(args.max_swaps+1)}, splits={})
    entries = dict(zip(('train', 'validation', 'test'), splits(cfg, include_test=True)))
    # Save complete search settings before any test evaluation.
    (output/'settings.json').write_text(json.dumps(report, indent=2))
    reference = json.loads(Path(args.reference).read_text()) if args.reference else None
    if reference and reference['baseline_checkpoint_digest'] != report['baseline_digest']:
        raise ValueError('Reference experiment uses a different checkpoint')
    for split in args.splits:
        bank, path = evaluate_bank(model, entries[split], args, states, split, args.device)
        summary, choices = summarize(bank, states, children, args.max_swaps, args.beam_width)
        if split == 'validation' and abs(summary['greedy']['0']['loss']-source['validation']['loss']) > 1e-4:
            raise AssertionError('Activity baseline failed reproduction')
        if reference and args.drops == 4 and args.adds == 6:
            old = reference['oracle_diagnostic'][split]
            current = summary['greedy']['1']
            if abs(current['loss']-old['loss']) > 1e-4 or abs(current['accuracy']-old['accuracy']) > 1e-6:
                raise AssertionError('Single-swap oracle failed reproduction')
        report['splits'][split] = dict(searches=summary, bank=str(path),
            extraction_seconds=bank['extraction_seconds'], actual_evaluations=len(bank['labels'])*len(states))
        save_checkpoint(output/f'{split}_choices.pt', dict(states=states, choices=choices,
            entries=[str(p) for p, _ in entries[split]], bank=str(path)))
        print(json.dumps({split: report['splits'][split]}, indent=2), flush=True)
    (output/'results.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--reference', help='Original swap_experiment results.json, for reproduction checks')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--budget', type=int, default=54)
    parser.add_argument('--drops', type=int, default=4)
    parser.add_argument('--adds', type=int, default=6)
    parser.add_argument('--max-swaps', type=int, default=4)
    parser.add_argument('--beam-width', type=int, default=4)
    parser.add_argument('--max-sets', type=int, default=10000)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--encoder-batch', type=int, default=64)
    parser.add_argument('--splits', nargs='+', choices=['train', 'validation', 'test'], default=['train', 'validation', 'test'])
    args = parser.parse_args()
    if min(args.batch_size, args.encoder_batch, args.beam_width) < 1:
        parser.error('Batch sizes and beam width must be positive')
    run(args)
