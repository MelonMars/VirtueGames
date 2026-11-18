"""Inspect what selected dossier directions predict and their training-set stability."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
import re

import torch
from safetensors.torch import load_file
from tqdm import tqdm
from analyze_directions import load_run, jsonl, weights, corr


def weighted_corr(x, y, rows):
    return corr(torch.tensor(x, dtype=torch.float64), torch.tensor(y, dtype=torch.float64), weights(rows))


def item_fields(path, reviewed):
    groups = defaultdict(list)
    for r in jsonl(reviewed or path / 'samples.jsonl'):
        if r.get('decision_confidence') is not None and r.get('decision_correct') is not None:
            groups[r['id']].append(r)
    result = {}
    for key, rs in groups.items():
        result[key] = {'confidence': sum(r['decision_confidence']/100 for r in rs)/len(rs),
                       'correctness': sum(int(r['decision_correct']) for r in rs)/len(rs),
                       'abstention': sum((r.get('answer') or '').strip().casefold() == 'cannot determine' for r in rs)/len(rs)}
    return result


def fit_axis(rows, hook):
    x = torch.stack([r['x'][hook] for r in rows]).double()
    y = torch.tensor([r['error'] for r in rows], dtype=torch.float64)
    w = weights(rows)
    axis = (w[:, None]*(x-(w[:,None]*x).sum(0))*(y-(w*y).sum())[:,None]).sum(0)
    return axis/axis.norm() if axis.norm() > 1e-15 else None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('train_run', type=Path)
    p.add_argument('validation_run', type=Path)
    p.add_argument('--analysis-dir', type=Path, required=True)
    p.add_argument('--manifest', type=Path, default=Path(__file__).parent/'challenge_bank/partitions.json')
    p.add_argument('--train-reviewed', type=Path)
    p.add_argument('--validation-reviewed', type=Path)
    p.add_argument('--repeats', type=int, default=100)
    p.add_argument('--permutations', type=int, default=0, help='Group-preserving shuffled-label trials across all captured layers')
    p.add_argument('--seed', type=int, default=41)
    p.add_argument('--out', type=Path, required=True, help='New JSON output file')
    a = p.parse_args()
    if a.repeats < 1 or a.permutations < 0:
        p.error('--repeats must be positive and --permutations nonnegative')
    report = json.loads((a.analysis_dir/'report.json').read_text())
    vectors = load_file(str(a.analysis_dir/'directions.safetensors'), device='cpu')
    if set(vectors) != set(report['selected']) or not vectors:
        raise ValueError('Selected direction keys do not match vector file')
    if hashlib.sha256(a.manifest.read_bytes()).hexdigest() != report['partition_sha256']:
        raise ValueError('Manifest checksum does not match the analysis')
    manifest = json.loads(a.manifest.read_text())['cases']
    trconfig, train, _, _ = load_run(a.train_run, 'train', a.train_reviewed)
    vconfig, val, _, _ = load_run(a.validation_run, 'validation', a.validation_reviewed)
    for c in (trconfig, vconfig):
        if c.get('bank_sha256') != report['bank_sha256'] or c.get('partition_sha256') != report['partition_sha256'] or c.get('model') != report['model'] or c.get('revision') != report['revision']:
            raise ValueError('Run provenance differs from analysis')
    # Detect accidental use of different samples from those used to fit the directions.
    for field, supplied in [('train_reviewed', a.train_reviewed), ('validation_reviewed', a.validation_reviewed)]:
        if bool(report.get(field)) != bool(supplied):
            raise ValueError(f'{field} must match whether the analysis used reviewed samples')
    for rows, run, reviewed in ((train, a.train_run, a.train_reviewed), (val, a.validation_run, a.validation_reviewed)):
        details = item_fields(run, reviewed)
        for r in rows:
            r.update(details[r['id']])
            r['component'] = manifest[r['id']]['source_component']
    if {r['component'] for r in train} & {r['component'] for r in val}:
        raise ValueError('Source components overlap train and validation')
    rng = random.Random(a.seed)
    result = {}
    for key in report['selected']:
        layer = int(re.fullmatch(r'decision\.layer_(\d+)\.hook_resid_post', key)[1])
        hook = f'blocks.{layer}.hook_resid_post'
        vector = vectors[key].double()
        fitted = fit_axis(train, hook)
        if fitted is None or float(fitted @ vector) < 0.99999:
            raise ValueError(f'{key}: training activations or labels do not reproduce the saved direction')
        projections = [float(r['x'][hook].double() @ vector) for r in val]
        descriptive = {name: weighted_corr(projections, [r[name] for r in val], val)
                       for name in ('error','confidence','correctness','abstention')}
        original = next(r for r in report['layers'] if r.get('direction_key') == key)
        if descriptive['error'] is None or abs(descriptive['error'] - original['validation_correlation']) > 1e-5:
            raise ValueError(f'{key}: validation activations or labels differ from the saved analysis')
        per_family = {}
        for family in sorted({r['family'] for r in val}):
            indices = [i for i,r in enumerate(val) if r['family'] == family]
            subset = [val[i] for i in indices]
            per_family[family] = {'cases': len(indices), 'correlation': weighted_corr([projections[i] for i in indices], [val[i]['error'] for i in indices], subset) if len(indices)>=3 else None}
        groups = sorted({r['component'] for r in train})
        subsamples = []
        for _ in tqdm(range(a.repeats), desc=f'Layer {layer} source-group resamples', unit='resample'):
            kept = set(rng.sample(groups, max(1, int(len(groups)*0.8))))
            sample = [r for r in train if r['component'] in kept]
            if len(sample)<4:
                continue
            axis = fit_axis(sample, hook)
            if axis is None:
                continue
            pp = [float(r['x'][hook].double() @ axis) for r in val]
            subsamples.append({'cosine': float(axis @ vector), 'validation_correlation': weighted_corr(pp, [r['error'] for r in val], val)})
        result[key] = {'validation_correlations': descriptive, 'validation_by_family': per_family,
                       'training_source_groups': len(groups), 'resamples': subsamples}
    null = None
    if a.permutations:
        # Exchange whole source groups only between groups with identical family counts.
        # Order items within each family so labels move in family-matched slots.
        group_indices = defaultdict(list)
        for i, row in enumerate(train):
            group_indices[row['component']].append(i)
        strata = defaultdict(list)
        for component, indices in group_indices.items():
            indices.sort(key=lambda i: (train[i]['family'], train[i]['id']))
            signature = tuple(sorted((family, sum(train[i]['family'] == family for i in indices))
                                     for family in {train[i]['family'] for i in indices}))
            strata[signature].append(component)
        exchangeable = sum(len(v) for v in strata.values() if len(v) > 1)
        if not exchangeable:
            raise ValueError('No exchangeable source groups for the shuffled-label control')
        n = len(train)
        permuted = []
        for _ in range(a.permutations):
            mapping = list(range(n))
            for components in strata.values():
                sources = components[:]
                rng.shuffle(sources)
                for dest, source in zip(components, sources):
                    for i, j in zip(group_indices[dest], group_indices[source]):
                        mapping[i] = j
            permuted.append(mapping)
        yt = torch.tensor([r['error'] for r in train], dtype=torch.float32)
        labels = yt[torch.tensor(permuted, dtype=torch.long).T]
        wt, wv = weights(train).float(), weights(val).float()
        labels -= (wt[:, None] * labels).sum(0)
        yv = torch.tensor([r['error'] for r in val], dtype=torch.float32)
        yv -= (wv*yv).sum()
        yv_norm = torch.sqrt((wv*yv.square()).sum())
        maxima = torch.full((a.permutations,), -float('inf'))
        hooks = sorted(set(train[0]['x']) & set(val[0]['x']))
        for hook in tqdm(hooks, desc='Full-layer shuffled-label control', unit='layer'):
            xt = torch.stack([r['x'][hook] for r in train]).float()
            xv = torch.stack([r['x'][hook] for r in val]).float()
            xt -= (wt[:, None]*xt).sum(0)
            xv -= (wv[:, None]*xv).sum(0)
            axes = xt.T @ (wt[:, None]*labels)
            projection = xv @ axes
            denominator = torch.sqrt((wv[:, None]*projection.square()).sum(0))*yv_norm
            correlations = torch.where(denominator > 1e-15, (wv[:, None]*projection*yv[:, None]).sum(0)/denominator, -torch.inf)
            maxima = torch.maximum(maxima, correlations)
        finite = [float(x) for x in maxima if torch.isfinite(x)]
        observed = max(r['validation_correlation'] for r in report['layers'] if r['validation_correlation'] is not None)
        null = {'trials': len(finite), 'exchangeable_source_groups': exchangeable,
                'observed_best_validation_correlation': observed,
                'max_validation_correlations': finite,
                'empirical_upper_tail_fraction': (1+sum(x >= observed for x in finite))/(1+len(finite)),
                'method': 'Shuffle whole training source groups within matching family-count strata; refit and select the maximum validation correlation across every captured layer.'}
    if a.out.exists():
        raise FileExistsError(a.out)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({'directions': result, 'seed': a.seed, 'shuffled_labels': null,
        'limitations': ['Validation correlations after layer selection are exploratory.',
                        'Resamples retain 80% of training source groups without replacement; validation is reused.',
                        'Abstention detection uses exact Cannot determine text.',
                        'Correlations do not isolate the contribution of a variable or establish causality.',
                        'Repeated outputs from a case are averaged, not treated as independent cases.']}, indent=2, allow_nan=False)+'\n')
    print(f'Saved {a.out}')


if __name__ == '__main__':
    main()
