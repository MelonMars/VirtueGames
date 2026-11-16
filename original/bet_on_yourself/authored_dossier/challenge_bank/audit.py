"""Report dossier-bank coverage and mechanical problems; not a source review."""
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

FAMILIES = {
    'train': {'direct_support', 'indirect_evidence', 'incomplete_dossier',
              'conflicting_sources', 'source_quality', 'scope_boundary', 'false_premise'},
    'test': {'overturn_prior', 'ambiguous_identity', 'well_supported_abstention'},
}


def audit(path):
    bank = json.loads(Path(path).read_text(encoding='utf-8'))
    cases = bank.get('cases', [])
    issues = []
    if bank.get('protocol') != 'bet-on-yourself-dossier-v1' or not isinstance(cases, list):
        raise ValueError('Unexpected protocol or cases')
    ids, questions, splits_by_source = set(), {}, defaultdict(set)
    counts, outcomes, reviews, source_uses = Counter(), defaultdict(Counter), Counter(), Counter()
    for c in cases:
        cid = c.get('id', '<missing>')
        if cid in ids:
            issues.append(f'{cid}: duplicate ID')
        ids.add(cid)
        q = ' '.join(re.findall(r'\w+', c.get('question', '').casefold()))
        if not q or q in questions:
            issues.append(f'{cid}: missing or duplicate question ({questions.get(q)})')
        questions[q] = cid
        split, family = c.get('analysis_split'), c.get('family')
        if family not in FAMILIES.get(split, set()):
            issues.append(f'{cid}: family/split assignment is not in the design')
        counts[(split, family)] += 1
        outcomes[family][c.get('answer_type', '<missing>')] += 1
        reviews[c.get('review_status', '<missing>')] += 1
        for key in ('question', 'dossier', 'answer', 'rationale'):
            if not isinstance(c.get(key), str) or not c[key].strip():
                issues.append(f'{cid}: missing {key}')
        aliases = c.get('accepted_answers')
        if not isinstance(aliases, list) or not aliases or any(not isinstance(x, str) or not x.strip() for x in aliases):
            issues.append(f'{cid}: missing accepted answer forms')
        if c.get('answer_type') not in ('fact', 'unresolved'):
            issues.append(f'{cid}: invalid answer type')
        sources = c.get('sources')
        if not isinstance(sources, list) or not sources:
            issues.append(f'{cid}: missing sources')
            continue
        for source in sources:
            url = source.get('url', '')
            if not url.startswith(('https://', 'http://')) or not source.get('accessed'):
                issues.append(f'{cid}: incomplete source metadata')
            splits_by_source[url].add(split)
            source_uses[url] += 1
    for url, splits in splits_by_source.items():
        if len(splits) > 1:
            issues.append(f'source crosses splits: {url}')
    missing = {family: 30 - counts[(split, family)] for split, families in FAMILIES.items()
               for family in sorted(families) if counts[(split, family)] < 30}
    return {'n_cases': len(cases), 'family_counts': {f'{s}/{f}': counts[(s, f)]
            for s, fs in FAMILIES.items() for f in sorted(fs)}, 'remaining_to_30': missing,
            'outcomes': {f: dict(c) for f, c in sorted(outcomes.items())},
            'review_status': dict(reviews), 'n_distinct_source_urls': len(source_uses),
            'most_reused_sources': [{'url': url, 'case_uses': n} for url, n in source_uses.most_common(10)],
            'issues': issues,
            'note': 'Mechanical audit only: source fidelity, shared facts, answer completeness, independent review, '
                    'semantic duplication, and prior beliefs require separate checks.'}


if __name__ == '__main__':
    report = audit(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).with_name('cases.json'))
    print(json.dumps(report, indent=2))
    raise SystemExit(bool(report['issues']))
