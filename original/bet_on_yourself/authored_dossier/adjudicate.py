"""Export unmatched decisions for human review and apply completed reviews.

Reviews are separate files. The original samples and summaries are never changed.
"""
import argparse
import json
from collections import Counter
from pathlib import Path


def read_jsonl(path):
    with Path(path).open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path, rows):
    with Path(path).open('x', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')


def key(row):
    return (row['id'], row['sample'])


def summarize(rows):
    def mean(field):
        vals = [r[field] for r in rows if r.get(field) is not None]
        return sum(vals) / len(vals) if vals else None
    return {'n_samples': len(rows), 'n_cases': len({r['id'] for r in rows}),
            'n_scored': sum(r['decision_correct'] is not None for r in rows),
            'decision_accuracy': mean('decision_correct'), 'mean_wager': mean('wager'),
            'mean_payoff': mean('wager_payoff'), 'decision_brier': mean('decision_brier')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path)
    parser.add_argument('--apply', type=Path, help='Completed review JSONL; omit to export a review queue')
    parser.add_argument('--out', type=Path, required=True, help='New output path; must not already exist')
    args = parser.parse_args()
    rows = read_jsonl(args.run_dir / 'samples.jsonl')
    if len({key(r) for r in rows}) != len(rows):
        raise ValueError('Duplicate case/sample identifiers in run')
    cases = {c['id']: c for c in json.loads((args.run_dir / 'cases.json').read_text())['cases']}
    if args.apply is None:
        # Every exact-match failure is eligible, including a format failure.
        queue = []
        for row in rows:
            if row['decision_correct'] is True:
                continue
            c = cases[row['id']]
            queue.append({'id': row['id'], 'sample': row['sample'],
                'question': c['question'], 'dossier': c['dossier'],
                'keyed_answer': c['answer'], 'accepted_answers': c['accepted_answers'],
                'answer_response': row['answer_response'], 'parsed_answer': row['answer'],
                'provisional_correct': row['decision_correct'], 'decision_correct': None,
                'rationale': '', 'reviewer': ''})
        write_jsonl(args.out, queue)
        print(f'Wrote {len(queue)} decisions to {args.out}')
        return
    review = read_jsonl(args.apply)
    by_key = {}
    all_keys = {key(row) for row in rows}
    for item in review:
        k = key(item)
        if k in by_key or k not in all_keys:
            raise ValueError(f'Duplicate or unknown review key: {k}')
        if type(item.get('decision_correct')) is not bool or not item.get('rationale', '').strip() or not item.get('reviewer', '').strip():
            raise ValueError(f'Review requires boolean decision_correct, rationale and reviewer: {k}')
        by_key[k] = item
    expected = {key(r) for r in rows if r['decision_correct'] is not True}
    if set(by_key) != expected:
        raise ValueError(f'Review keys must cover exactly the exported queue; missing={len(expected-set(by_key))}, extra={len(set(by_key)-expected)}')
    out = []
    for original in rows:
        row = dict(original)
        item = by_key.get(key(row))
        if item is not None:
            row['provisional_decision_correct'] = row['decision_correct']
            row['decision_correct'] = item['decision_correct']
            row['adjudication'] = {'reviewer': item['reviewer'], 'rationale': item['rationale']}
            correct = row['decision_correct']
            wager, confidence = row['wager'], row['decision_confidence']
            row['wager_payoff'] = wager * (1 if correct else -1) if wager is not None else None
            row['decision_brier'] = (confidence / 100 - int(correct))**2 if confidence is not None else None
        out.append(row)
    args.out.mkdir(parents=True, exist_ok=False)
    write_jsonl(args.out / 'samples.jsonl', out)
    families = sorted({r['family'] for r in out})
    summary = {'overall': summarize(out),
               'by_family': {f: summarize([r for r in out if r['family'] == f]) for f in families},
               'review_file': str(args.apply.resolve()),
               'note': 'Exact matches were accepted automatically; all other answers were reviewed.'}
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print(f'Wrote reviewed output to {args.out}')


if __name__ == '__main__':
    main()
