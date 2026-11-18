"""Sweep dossier confidence steering on validation, or evaluate a frozen choice on test."""
import argparse
from collections import defaultdict
from dataclasses import fields
import hashlib
import json
import math
from pathlib import Path
import random
import re
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch
from safetensors.torch import load_file
from tqdm import tqdm
from config import RunConfig
from llm import generate, load_model
from original.bet_on_yourself.steer_calibration import ResidualAddition
from original.bet_on_yourself.authored_dossier.run import parse_number
from runs import model_runtime


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def jsonl(path):
    with Path(path).open(encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def samples(run, reviewed):
    original = jsonl(run / 'samples.jsonl')
    rows = jsonl(reviewed) if reviewed else original
    def index(rs):
        d = {}
        for r in rs:
            k = (r['id'], r['sample'])
            if k in d:
                raise ValueError(f'Duplicate sample {k}')
            d[k] = r
        return d
    old, new = index(original), index(rows)
    if old.keys() != new.keys():
        raise ValueError('Reviewed sample keys do not match original run')
    for k in old:
        for field in ('id', 'sample', 'family', 'answer', 'answer_response', 'answer_prompt', 'decision_confidence'):
            if old[k].get(field) != new[k].get(field):
                raise ValueError(f'Review altered {field} for {k}')
        if new[k].get('decision_correct') is not None and type(new[k]['decision_correct']) is not bool:
            raise ValueError(f'Invalid decision_correct for {k}')
    return rows


def weighted_mean(rows, value):
    # First average repetitions in a case, then cases within families, then families.
    cases = defaultdict(list)
    for r in rows:
        v = value(r)
        if v is not None:
            cases[(r['family'], r['id'])].append(v)
    fam = defaultdict(list)
    for (family, _), vals in cases.items():
        fam[family].append(sum(vals) / len(vals))
    return sum(sum(v)/len(v) for v in fam.values())/len(fam) if fam else None


def summarize(rows):
    valid = [r for r in rows if r['confidence'] is not None and r['correct'] is not None]
    out = {'samples': len(rows), 'cases': len({r['id'] for r in rows}),
           'valid_samples': len(valid), 'parse_rate': sum(r['confidence'] is not None for r in rows)/len(rows) if rows else None,
           'brier_equal_family': weighted_mean(valid, lambda r: (r['confidence']/100-int(r['correct']))**2),
           'confidence_equal_family': weighted_mean(valid, lambda r: r['confidence']/100),
           'accuracy_equal_family': weighted_mean(valid, lambda r: int(r['correct']))}
    bins = []
    for low in range(0, 100, 20):
        group = [r for r in valid if low <= r['confidence'] < low+20 or (low == 80 and r['confidence'] == 100)]
        bins.append({'range': f'{low}-{low+20}', 'samples': len(group),
                     'confidence': sum(r['confidence']/100 for r in group)/len(group) if group else None,
                     'accuracy': sum(r['correct'] for r in group)/len(group) if group else None})
    out['calibration_bins_sample_weighted'] = bins
    return out


def pair_change(rows, baseline):
    base = {(r['id'], r['sample']): r for r in baseline}
    paired = []
    for r in rows:
        b = base[(r['id'], r['sample'])]
        if r['confidence'] is not None and b['confidence'] is not None and r['correct'] is not None:
            paired.append(dict(r, delta=(r['confidence']/100-int(r['correct']))**2 -
                               (b['confidence']/100-int(b['correct']))**2))
    return {'n_pairs': len(paired), 'complete_coverage': len(paired) == len(baseline),
            'brier_change_equal_family': weighted_mean(paired, lambda r: r['delta'])}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run_dir', type=Path, help='Completed validation or test dossier run')
    p.add_argument('--analysis-dir', type=Path, required=True)
    p.add_argument('--train-run', type=Path, required=True, help='Training run used for the simple offset baseline')
    p.add_argument('--out', type=Path, required=True, help='New output directory')
    p.add_argument('--reviewed', type=Path, help='Reviewed samples.jsonl for this run')
    p.add_argument('--train-reviewed', type=Path, help='Reviewed training samples.jsonl')
    p.add_argument('--selection', type=Path, help='Frozen selection.json from a validation sweep; required for test')
    p.add_argument('--alphas', default='-4,-2,-1,1,2,4')
    p.add_argument('--seed', type=int, default=2701)
    p.add_argument('--device', choices=('auto', 'cuda', 'cpu'), default='auto')
    p.add_argument('--runtime', choices=('local', 'runpod'), default='local')
    p.add_argument('--model', help='Local path to the same model weights')
    p.add_argument('--allow-unverified-checkpoint', action='store_true', help='Acknowledge that the recorded checkpoint revision is unavailable')
    args = p.parse_args()
    run, train, analysis = args.run_dir.resolve(), args.train_run.resolve(), args.analysis_dir.resolve()
    source, training, report = read(run/'config.json'), read(train/'config.json'), read(analysis/'report.json')
    if bool(report.get('train_reviewed')) != bool(args.train_reviewed):
        raise ValueError('Training review must match the analysis; rerun direction analysis after changing labels')
    if read(run/'status.json').get('status') != 'complete' or read(train/'status.json').get('status') != 'complete':
        raise ValueError('Both runs must be complete')
    is_test = source.get('split') == 'test' and source.get('partition') is None
    is_validation = source.get('split') == 'train' and source.get('partition') == 'validation'
    if not is_test and not is_validation:
        raise ValueError('Input run must be the validation partition or held-out test split')
    if is_test != bool(args.selection):
        raise ValueError('Supply --selection for test only; do not select using test data')
    if is_validation and bool(report.get('validation_reviewed')) != bool(args.reviewed):
        raise ValueError('Validation review must match the analysis; rerun direction analysis after changing labels')
    if training.get('split') != 'train' or training.get('partition') != 'train':
        raise ValueError('Training run must use the train partition')
    if report.get('bank_sha256') != source.get('bank_sha256') or report.get('bank_sha256') != training.get('bank_sha256'):
        raise ValueError('Bank checksum differs from analysis')
    if report.get('partition_sha256') != training.get('partition_sha256') or (is_validation and source.get('partition_sha256') != report.get('partition_sha256')):
        raise ValueError('Partition checksum differs from analysis')
    for key in ('model', 'revision', 'backend', 'thinking', 'answer_system', 'wager_system', 'confidence_system', 'temperature'):
        if training.get(key) != source.get(key):
            raise ValueError(f'Run settings differ: {key}')
    if report.get('model') != training.get('model') or report.get('revision') != training.get('revision'):
        raise ValueError('Training model differs from analysis')
    for key in ('resolved_revision', 'dtype'):
        if training.get('model_runtime', {}).get(key) != source.get('model_runtime', {}).get(key):
            raise ValueError(f'Run model runtime differs: {key}')
    revision = training.get('model_runtime', {}).get('resolved_revision')
    if not revision and not args.allow_unverified_checkpoint:
        raise ValueError('No resolved checkpoint revision recorded; verify weights manually and pass --allow-unverified-checkpoint')
    vectors_path = analysis/'directions.safetensors'
    vectors = load_file(str(vectors_path), device='cpu')
    if set(vectors) != set(report['selected']) or not vectors:
        raise ValueError('Vectors do not match the analysis selection')
    for key, vector in vectors.items():
        if not re.fullmatch(r'decision\.layer_\d+\.hook_resid_post', key) or vector.ndim != 1 or not torch.isfinite(vector).all() or abs(float(vector.norm())-1) > 1e-3:
            raise ValueError(f'Invalid vector: {key}')
    if args.selection:
        selection = read(args.selection)
        if selection.get('vectors_sha256') != digest(vectors_path) or selection.get('report_sha256') != digest(analysis/'report.json'):
            raise ValueError('Frozen choice was made for different analysis files')
        if selection.get('training_samples_sha256') != digest(args.train_reviewed or train/'samples.jsonl'):
            raise ValueError('Frozen choice used different training labels')
        seed = selection['seed']
        choice = selection['choice']
        if choice and (choice['key'] not in vectors or not math.isfinite(choice['alpha']) or choice['alpha'] == 0):
            raise ValueError('Invalid frozen choice')
        alphas = [choice['alpha']] if choice else []
        keys = [choice['key']] if choice else []
    else:
        seed = args.seed
        alphas = sorted(set(float(x) for x in args.alphas.split(',')) - {0.0})
        if not alphas or not all(math.isfinite(a) for a in alphas) or not min(alphas) < 0 < max(alphas):
            raise ValueError('Supply finite alphas of both signs')
        keys = list(report['selected'])
    train_rows = samples(train, args.train_reviewed)
    valid_train = [r for r in train_rows if r.get('decision_confidence') is not None and r.get('decision_correct') is not None]
    if not valid_train:
        raise ValueError('No usable training confidences')
    offset = weighted_mean(valid_train, lambda r: int(r['decision_correct']) - r['decision_confidence']/100)
    rows = [r for r in samples(run, args.reviewed) if r.get('decision_correct') is not None]
    if not rows or any(not r.get('answer_prompt') for r in rows):
        raise ValueError('No scored fixed-answer prompts')
    cases = {r['id'] for r in rows}
    if cases & {r['id'] for r in train_rows}:
        raise ValueError('Evaluation cases overlap training')
    if any(r.get('analysis_split') != ('test' if is_test else 'train') or
           r.get('partition') != (None if is_test else 'validation') for r in rows):
        raise ValueError('Evaluation sample split disagrees with run configuration')
    if is_test and {r['family'] for r in rows} & {r['family'] for r in train_rows}:
        raise ValueError('Test families overlap training families')
    values = {f.name: source[f.name] for f in fields(RunConfig) if f.name in source}
    values.update(questions=Path(source.get('questions_file', run/'cases.json')), difficulty=tuple(source.get('difficulty_filter') or ()),
                  backend='transformers', extract_activations=False, device=args.device, runtime=args.runtime, sample_batch_size=1)
    if args.model:
        values['model'] = args.model
    if revision:
        values['revision'] = revision
    config = RunConfig(**values)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    write(out/'status.json', {'status': 'running'})
    conditions = [('baseline', None, 0), ('zero_hook', keys[0] if keys else next(iter(vectors)), 0)]
    for key in keys:
        for alpha in alphas:
            conditions.extend([('vector', key, alpha), ('random', key, alpha)])
    controls = {}
    for key in keys:
        gen = torch.Generator(device='cpu').manual_seed(seed + int(hashlib.sha256(key.encode()).hexdigest()[:8], 16))
        v = torch.randn(vectors[key].shape, generator=gen, dtype=torch.float32)
        v -= (v @ vectors[key].float()) * vectors[key].float()
        controls[key] = v / v.norm()
    generated = defaultdict(list)
    try:
        from transformer_lens.model_bridge import TransformerBridge
        with load_model(config) as llm:
            loaded = model_runtime(llm)
            if revision and loaded.get('resolved_revision') != revision:
                raise ValueError('Loaded checkpoint revision differs or cannot be verified')
            expected_dtype = source.get('model_runtime', {}).get('dtype')
            if expected_dtype and loaded.get('dtype') != expected_dtype:
                raise ValueError('Loaded model dtype differs from source')
            bridge = TransformerBridge.boot_transformers(str(config.model), hf_model=llm.model,
                tokenizer=llm.tokenizer, device=llm.model.device, dtype=llm.model.dtype, revision=config.revision)
            bridge.eval()
            with (out/'samples.jsonl').open('w', encoding='utf-8') as stream, tqdm(total=len(rows)*len(conditions), desc='Confidence steering', unit='generation') as bar:
                for row in rows:
                    fixed = row['answer_prompt'] + '\n\nFixed answer: ' + (row.get('answer') or '[invalid answer format]')
                    paired_seed = (seed + int.from_bytes(hashlib.sha256(f"{row['id']}:{row['sample']}".encode()).digest()[:4], 'big')) % (2**31-1)
                    baseline_raw = None
                    for kind, key, alpha in conditions:
                        kwargs = dict(temperature=config.temperature, max_tokens=8, seed=paired_seed)
                        if kind == 'baseline':
                            raw = generate(llm, source['confidence_system'], fixed, **kwargs)
                            baseline_raw = raw
                        else:
                            layer = int(re.fullmatch(r'decision\.layer_(\d+)\.hook_resid_post', key)[1])
                            hook = f'blocks.{layer}.hook_resid_post'
                            if hook not in bridge.hook_dict:
                                raise ValueError(f'Missing model hook {hook}')
                            vector = controls[key] if kind == 'random' else vectors[key]
                            addition = ResidualAddition(vector, alpha, site='prompt-and-decode')
                            with bridge.hooks(fwd_hooks=[(hook, addition)]):
                                raw = generate(llm, source['confidence_system'], fixed, **kwargs)
                            if not addition.applications:
                                raise RuntimeError(f'Steering hook did not fire: {hook}')
                            if kind == 'zero_hook' and raw != baseline_raw:
                                raise RuntimeError(f'Zero-strength hook changed paired output for {row["id"]}/{row["sample"]}')
                        record = {'id': row['id'], 'sample': row['sample'], 'family': row['family'],
                                  'correct': row['decision_correct'], 'kind': kind, 'key': key, 'alpha': alpha,
                                  'raw': raw, 'confidence': parse_number(raw, 100), 'seed': paired_seed}
                        generated[(kind,key,alpha)].append(record)
                        stream.write(json.dumps(record)+'\n')
                        bar.update(1)
        baseline = generated[('baseline',None,0)]
        offset_rows = [dict(r, confidence=max(0, min(100, r['confidence']+100*offset)) if r['confidence'] is not None else None) for r in baseline]
        summaries = []
        for (kind,key,alpha), group in generated.items():
            summaries.append({'kind': kind, 'key': key, 'alpha': alpha, 'metrics': summarize(group),
                              'paired_vs_baseline': pair_change(group, baseline),
                              'by_family': {f: summarize([r for r in group if r['family']==f]) for f in sorted({r['family'] for r in group})}})
        scores = [r for r in summaries if r['kind']=='vector' and r['paired_vs_baseline']['complete_coverage'] and r['paired_vs_baseline']['brier_change_equal_family'] is not None]
        if not is_test:
            best = min(scores, key=lambda r:r['paired_vs_baseline']['brier_change_equal_family']) if scores else None
            choice = {'key': best['key'], 'alpha': best['alpha']} if best and best['paired_vs_baseline']['brier_change_equal_family'] < 0 else None
            write(out/'selection.json', {'choice': choice, 'seed': seed, 'vectors_sha256': digest(vectors_path),
                  'report_sha256': digest(analysis/'report.json'), 'training_samples_sha256': digest(args.train_reviewed or train/'samples.jsonl'),
                  'validation_samples_sha256': digest(args.reviewed or run/'samples.jsonl'),
                  'selection_rule': 'lowest complete-coverage paired equal-family Brier change if negative; otherwise no steering',
                  'notes': 'Selected on validation; freeze this file before opening held-out results.'})
        write(out/'report.json', {'split': 'test' if is_test else 'validation', 'selection': choice,
              'model_runtime': loaded, 'training_offset': offset, 'simple_offset_baseline': summarize(offset_rows),
              'simple_offset_paired_vs_baseline': pair_change(offset_rows, baseline), 'conditions': summaries,
              'excluded_unscored_answers': len(samples(run, args.reviewed))-len(rows),
              'notes': ['Answer is fixed; intervention acts only during confidence generation.',
                        'Paired seeds do not guarantee identical random draws after logits change.',
                        'Validation selection is exploratory; test must use its frozen selection.',
                        'Simple offset is fit using original training confidences and clipped to [0, 100].',
                        'The data bank and provisional answer scoring require independent review.']})
        write(out/'status.json', {'status': 'complete'})
    except BaseException as exc:
        write(out/'status.json', {'status': 'failed', 'error': str(exc)})
        raise
    print(f'Saved {out}; selection: {choice}')


if __name__ == '__main__':
    main()
