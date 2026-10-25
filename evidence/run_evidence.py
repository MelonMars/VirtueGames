"""Deterministic measurement and comprehension gating for evidence responsiveness."""
import sys
from pathlib import Path
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
from collections import Counter, defaultdict
from dataclasses import replace
from contextlib import nullcontext
from tqdm import tqdm
from config import parse_config
from llm import load_model, without_activation_capture
from runs import jsonl_writer, next_run_dir, write_json, utc_now, model_runtime
from evidence.protocol import ANSWERS, PROTOCOL, SYSTEM, parse_commitment, targets, messages, control_messages, load_bank, digest
from evidence.score_evidence import SCORING_VERSION, SUPPORT_COORDINATES, control_passes, score_sample, summarize


def complete(llm, conversation, config, *, temperature, seed, context, capture=True):
    # Use the same backend entry point as llm.generate_messages, retaining finish metadata.
    before = getattr(llm, 'performance', {}).get('token_limit_hits', 0)
    kwargs = dict(messages=conversation, temperature=temperature, seed=seed, max_tokens=config.max_tokens)
    with nullcontext() if capture else without_activation_capture(llm):
        if getattr(llm, 'activation_recorder', None) is not None:
            kwargs['activation_context'] = context
        response = llm.create_chat_completion(**kwargs)
    choice = response['choices'][0]
    raw = choice['message'].get('content') or ''
    hits = getattr(llm, 'performance', {}).get('token_limit_hits', 0) - before
    hits = max(hits, int(choice.get('finish_reason') == 'length'))
    # Reject backends that cannot report truncation rather than silently assuming success.
    if 'finish_reason' not in choice and 'token_limit_hits' not in getattr(llm, 'performance', {}):
        raise RuntimeError('Backend must expose finish_reason or token_limit_hits')
    return dict(messages=conversation, raw=raw, parsed=parse_commitment(raw), seed=seed,
                temperature=temperature, token_limit_hits=hits, finish_reason=choice.get('finish_reason'),
                usage=response.get('usage'))


def run(llm, questions, config, run_dir, *, variance=False, condition='neutral'):
    rows, items, calls = [], [], []
    with jsonl_writer(run_dir / 'samples.jsonl') as write, jsonl_writer(run_dir / 'transcripts.jsonl') as transcript:
        for q in tqdm(questions, desc='Evidence variance' if variance else 'Evidence measurement'):
            def call(msg, phase, sample=0, temperature=0., capture=True):
                value = complete(llm, msg, config, temperature=temperature, seed=config.seed+sample,
                    context=dict(id=q['id'], condition=phase, sample=sample, social_condition=condition, answer_protocol=PROTOCOL, structured_answer_choices=ANSWERS), capture=capture)
                record = dict(id=q['id'], intervention=q['intervention'], difficulty=q['difficulty'], phase=phase, sample=sample, **value)
                calls.append(record)
                transcript(record)
                return value
            control = call(control_messages(q), 'control', capture=False)
            usable = control_passes(q, control)
            prior = call(messages(q), 'prior')
            treatment = []
            for sample in range(config.n_samples if variance else 1):
                posterior = call(messages(q, prior['raw'], condition), 'posterior', sample,
                                 config.temperature if variance else 0.)
                hits = prior['token_limit_hits'] + posterior['token_limit_hits']
                row = dict(id=q['id'], split_group=q['split_group'], split=q['split'],
                    intervention=q['intervention'], difficulty=q['difficulty'], condition=condition,
                    sample=sample, prior=prior, posterior=posterior, control=control, usable=usable,
                    parse_ok=prior['parsed']['parse_ok'] and posterior['parsed']['parse_ok'], token_limit_hits=hits)
                row.update(score_sample(q, prior['parsed'], posterior['parsed'], usable=usable, truncated=bool(hits)))
                write(row)
                rows.append(row)
                treatment.append(row)
            counts = Counter(r['label'] for r in treatment if r['scored'])
            negatives = [label for label in ('under_update', 'over_update') if counts['correct_update'] and counts[label]]
            items.append(dict(id=q['id'], split_group=q['split_group'], split=q['split'], intervention=q['intervention'],
                difficulty=q['difficulty'], condition=condition, usable=usable, n_samples=len(treatment),
                update_distribution=dict(counts), unscored=sum(not r['scored'] for r in treatment),
                contrastive=bool(negatives) and variance and condition == 'neutral', negative_sets=negatives,
                discard_reason=('control_failed' if not usable else
                    'prior_unparsed' if not prior['parsed']['parse_ok'] else
                    'prior_truncated' if prior['token_limit_hits'] else
                    'prior_incorrect' if prior['parsed']['answer'] != targets(q)[0] else
                    'social_condition' if condition != 'neutral' else
                    'measurement_only' if not variance else
                    'no_valid_contrast' if not negatives else None)))
    with jsonl_writer(run_dir / 'variance.jsonl') as write:
        for item in items:
            write(item)
    write_json(run_dir / 'contrastive_items.json', [i for i in items if i['contrastive']])
    summary = summarize(rows)
    # Call-level counts avoid counting the shared prior/control N times.
    summary['generation_counts'] = {}
    for field in ('intervention', 'difficulty', 'phase'):
        buckets = defaultdict(list)
        for c in calls:
            buckets[c[field]].append(c)
        summary['generation_counts']['by_'+field] = {k: dict(n=len(v), unparsed=sum(not c['parsed']['parse_ok'] for c in v),
            token_limit_hits=sum(c['token_limit_hits'] for c in v)) for k,v in buckets.items()}
    summary['token_limit_hits'] = sum(c['token_limit_hits'] for c in calls)
    summary['contrastive_count'] = sum(i['contrastive'] for i in items)
    for field in ('difficulty', 'intervention'):
        summary['discard_by_'+field] = {key: dict(Counter(i['discard_reason'] or 'retained' for i in items if i[field] == key))
            for key in sorted({i[field] for i in items})}
    summary['contrastive_counts'] = {field: {key: {negative: sum(i['contrastive'] and negative in i['negative_sets'] for i in items if i[field] == key)
        for negative in ('under_update', 'over_update')} for key in sorted({i[field] for i in items})}
        for field in ('intervention', 'difficulty', 'split')}
    summary['phase1_ready'] = variance and all(any(i['contrastive'] and i['split']=='train' and neg in i['negative_sets']
        for i in items) for neg in ('under_update', 'over_update'))
    write_json(run_dir / 'summary.json', summary)
    return summary


def main(argv=None, *, variance=False):
    extra = argparse.ArgumentParser(add_help=False)
    extra.add_argument('--condition', choices=('neutral', 'social'), default='neutral')
    extra.add_argument('--ids', default='',
                       help='Comma-separated item IDs; select complete scenario families only')
    raw_args = sys.argv[1:] if argv is None else argv
    if '--help' in raw_args or '-h' in raw_args:
        print('Evidence options: --condition {neutral,social}.')
    options, remaining = extra.parse_known_args(raw_args)
    if '--questions' not in remaining and not any(a.startswith('--questions=') for a in remaining):
        remaining += ['--questions', str(Path(__file__).with_name('questions.json'))]
    config, output = parse_config(remaining, max_tokens=2048,
        out='runs_evidence_variance' if variance else 'runs_evidence', variance=True)
    if config.max_tokens < 1024:
        raise ValueError('Evidence runs require max_tokens >= 1024')
    if variance and (config.temperature <= 0 or config.n_samples < 2):
        raise ValueError('Variance requires positive temperature and at least two samples')
    if config.sample_batch_size != 1:
        raise ValueError('Evidence uses individual conversation samples; sample-batch-size must be 1')
    if not variance:
        config = replace(config, n_samples=1, temperature=0.)
    config = replace(config, activation_layers='all', activation_types='resid_post',
                     activation_positions='answer-tokens-1,answer-tokens')
    bank = load_bank(config.questions)
    requested_ids = {item.strip() for item in options.ids.split(',') if item.strip()}
    available_ids = {q['id'] for q in bank}
    missing_ids = requested_ids - available_ids
    if missing_ids:
        raise ValueError(f'Unknown item IDs: {sorted(missing_ids)}')
    selected = [q for q in bank
                if (not config.difficulty or q['difficulty'] in config.difficulty)
                and (not requested_ids or q['id'] in requested_ids)]
    if not selected:
        raise ValueError('No items selected')
    if requested_ids:
        selected_groups = {q['split_group'] for q in selected}
        for group in selected_groups:
            family = [q for q in bank if q['split_group'] == group]
            if {q['id'] for q in family} != {q['id'] for q in selected if q['split_group'] == group}:
                raise ValueError(f'--ids must include the complete scenario family: {group}')
    run_dir = next_run_dir(output)
    metadata = config.metadata('evidence-variance' if variance else 'evidence', len(selected))
    metadata.update(condition=options.condition, protocol=PROTOCOL, system_prompt=SYSTEM,
        seed_policy='control and prior: base_seed; posterior: base_seed + sample_index; reset conversation per sample',
        bank_hash=digest(bank), split_hash=digest({q['id']: [q['split_group'],q['split']] for q in bank}),
        selected_ids=[q['id'] for q in selected], split_unit='split_group',
        activation_slots=['answer-token-1', 'answer-token'], scoring_version=SCORING_VERSION, support_coordinates=SUPPORT_COORDINATES)
    metadata['config_hash'] = digest({k:v for k,v in metadata.items() if k != 'started'})
    write_json(run_dir / 'questions.json', dict(protocol=PROTOCOL, questions=bank))
    write_json(run_dir / 'splits.json', {q['id']: dict(split_group=q['split_group'], split=q['split']) for q in bank})
    write_json(run_dir / 'config.json', metadata)
    write_json(run_dir / 'status.json', dict(status='running', started=utc_now()))
    try:
        with load_model(config) as llm:
            write_json(run_dir / 'runtime.json', model_runtime(llm))
            if config.extract_activations:
                llm.begin_run(run_dir)
            summary = run(llm, selected, config, run_dir, variance=variance, condition=options.condition)
    except BaseException as exc:
        write_json(run_dir / 'status.json', dict(status='failed', error=str(exc), finished=utc_now()))
        raise
    write_json(run_dir / 'status.json', dict(status='complete', finished=utc_now()))
    print(run_dir)
    return summary


if __name__ == '__main__':
    main()
