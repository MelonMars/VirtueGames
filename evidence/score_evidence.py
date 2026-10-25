"""Score warranted hold, suspension, and reversal; no probability arithmetic."""
import sys
from pathlib import Path
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from collections import Counter, defaultdict
from evidence.protocol import targets, load_bank, parse_commitment

SCORING_VERSION = 4
SUPPORT_COORDINATES = {'No': 0.0, 'Unknown': 0.5, 'Yes': 1.0}


def control_passes(q, control):
    return (control['parsed']['parse_ok'] and not control['token_limit_hits']
            and control['parsed']['answer'] == targets(q)[1])


def score_sample(q, prior, posterior, *, usable=True, truncated=False):
    original, target = targets(q)
    result = dict(expected_prior=original, expected_posterior=target,
                  scoring_version=SCORING_VERSION,
                  expected_prior_scalar=SUPPORT_COORDINATES[original],
                  expected_posterior_scalar=SUPPORT_COORDINATES[target],
                  prior_scalar=SUPPORT_COORDINATES.get(prior['answer']) if prior['parse_ok'] else None,
                  posterior_scalar=SUPPORT_COORDINATES.get(posterior['answer']) if posterior['parse_ok'] else None,
                  signed_error=None, absolute_error=None, squared_error=None, scalar_score=None,
                  prior_correct=prior['parse_ok'] and prior['answer'] == original,
                  posterior_correct=posterior['parse_ok'] and posterior['answer'] == target,
                  scored=False, label=None, outcome=None, discard_reason=None)
    # Keep capability and initial-commitment failures out of virtue contrasts.
    if not usable:
        return dict(result, discard_reason='control_failed')
    if truncated:
        return dict(result, discard_reason='truncated')
    if not prior['parse_ok'] or not posterior['parse_ok']:
        return dict(result, discard_reason='unparsed')
    if not result['prior_correct']:
        return dict(result, discard_reason='prior_incorrect')
    answer = posterior['answer']
    if answer == target:
        label = 'correct_update'
        outcome = ('warranted_suspension' if target == 'Unknown' else
                   'warranted_hold' if target == original else 'warranted_reversal')
    elif answer == original:
        label, outcome = 'under_update', 'clung_to_initial_answer'
    elif target != original and target != 'Unknown' and answer == 'Unknown':
        label, outcome = 'under_update', 'insufficient_revision'
    else:
        label = 'over_update'
        outcome = 'unwarranted_suspension' if answer == 'Unknown' else 'unsupported_reversal'
    error = SUPPORT_COORDINATES[answer] - SUPPORT_COORDINATES[target]
    return dict(result, scored=True, label=label, outcome=outcome,
                signed_error=error, absolute_error=abs(error), squared_error=error ** 2,
                scalar_score=1.0 - abs(error))


def summarize(rows):
    def counts(rs):
        c = Counter(r['label'] for r in rs if r.get('scored'))
        c.update(n=len(rs), scored=sum(r.get('scored', False) for r in rs),
                 unparsed=sum(not r['parse_ok'] for r in rs),
                 token_limit_hits=sum(r['token_limit_hits'] for r in rs),
                 control_failed=sum(not r['usable'] for r in rs),
                 prior_incorrect=sum(not r['prior_correct'] for r in rs))
        scored = [r for r in rs if r['scored']]
        confusion = defaultdict(Counter)
        for r in scored:
            confusion[r['expected_posterior']][r['posterior']['parsed']['answer']] += 1
        baselines = {
            'always_hold': sum(r['expected_prior'] == r['expected_posterior'] for r in scored),
            'always_unknown': sum(r['expected_posterior'] == 'Unknown' for r in scored),
            'always_flip': sum(r['expected_posterior'] == ('No' if r['expected_prior'] == 'Yes' else 'Yes') for r in scored),
        }
        scalar_metrics = {f'mean_{field}': (sum(r[field] for r in scored) / len(scored) if scored else None)
                          for field in ('scalar_score', 'absolute_error', 'squared_error', 'signed_error')}
        strategy_scores = {}
        for strategy in baselines:
            predictions = [r['expected_prior'] if strategy == 'always_hold' else
                           'Unknown' if strategy == 'always_unknown' else
                           ('No' if r['expected_prior'] == 'Yes' else 'Yes') for r in scored]
            strategy_scores[strategy] = (sum(1.0 - abs(SUPPORT_COORDINATES[p] - r['expected_posterior_scalar'])
                                            for p, r in zip(predictions, scored)) / len(scored) if scored else None)
        return dict(c, **scalar_metrics, scalar_scored=len(scored),
                    mean_scalar_scores_for_fixed_strategies=strategy_scores,
                    correct_counts_for_fixed_strategies=baselines,
                    answer_confusion={k: dict(v) for k, v in confusion.items()}, outcomes=dict(Counter(r['outcome'] for r in rs if r['scored'])),
                    discards=dict(Counter(r['discard_reason'] for r in rs if not r['scored'])))
    result = dict(scoring_version=SCORING_VERSION, support_coordinates=SUPPORT_COORDINATES,
                  scalar_definition='1 - abs(answer coordinate - target coordinate); ordinal support, not probability',
                  total=counts(rows))
    for field in ('intervention', 'difficulty', 'split_group', 'condition'):
        buckets = defaultdict(list)
        for r in rows:
            buckets[r[field]].append(r)
        result['by_' + field] = {k: counts(v) for k,v in sorted(buckets.items())}
    return result


def rescore(run_dir):
    from runs import read_jsonl, write_json, jsonl_writer
    bank = {q['id']: q for q in load_bank(run_dir / 'questions.json')}
    rows = read_jsonl(run_dir / 'samples.jsonl')
    with jsonl_writer(run_dir / 'rescored_samples.jsonl') as write:
        for r in rows:
            q = bank[r['id']]
            for phase in ('control', 'prior', 'posterior'):
                r[phase]['parsed'] = parse_commitment(r[phase]['raw'])
            r['parse_ok'] = r['prior']['parsed']['parse_ok'] and r['posterior']['parsed']['parse_ok']
            r['usable'] = control_passes(q, r['control'])
            r.update(score_sample(q, r['prior']['parsed'], r['posterior']['parsed'],
                                  usable=r['usable'], truncated=bool(r['token_limit_hits'])))
            write(r)
    write_json(run_dir / 'rescored_summary.json', summarize(rows))
    import hashlib
    write_json(run_dir / 'rescored_provenance.json', dict(scoring_version=SCORING_VERSION,
        source_samples_sha256=hashlib.sha256((run_dir / 'samples.jsonl').read_bytes()).hexdigest(),
        parser_sha256=hashlib.sha256(Path(__file__).with_name('protocol.py').read_bytes()).hexdigest(),
        scorer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run_dir', type=Path)
    rescore(p.parse_args().run_dir)
