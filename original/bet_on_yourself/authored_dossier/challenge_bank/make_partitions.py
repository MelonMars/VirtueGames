"""Assign training-source components to development or validation, before modeling.

Connected components link cases sharing a source URL. This is a conservative
mechanical proxy, not a substitute for reviewing shared facts and topics.
"""
import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


def build(path, seed, topic_links_path):
    raw = Path(path).read_bytes()
    cases = json.loads(raw)['cases']
    parents = {c['id']: c['id'] for c in cases}
    if len(parents) != len(cases):
        raise ValueError('Duplicate case IDs')
    def root(item):
        while parents[item] != item:
            parents[item] = parents[parents[item]]
            item = parents[item]
        return item
    prior = {}
    for case in cases:
        for source in case['sources']:
            url = source['url']
            if url in prior:
                parents[root(case['id'])] = root(prior[url])
            else:
                prior[url] = case['id']
    topic_bytes = Path(topic_links_path).read_bytes()
    topic_links = json.loads(topic_bytes)
    if not isinstance(topic_links, dict) or not isinstance(topic_links.get('groups'), list):
        raise ValueError('Expected topic group manifest')
    for group in topic_links['groups']:
        ids = group.get('case_ids', [])
        if len(ids) < 2 or any(cid not in parents for cid in ids):
            raise ValueError(f'Invalid topic group: {group.get("topic")}')
        for cid in ids[1:]:
            parents[root(cid)] = root(ids[0])
    components = defaultdict(list)
    for case in cases:
        components[root(case['id'])].append(case)
    assignment, counts = {}, Counter()
    for component in components.values():
        splits = {c['analysis_split'] for c in component}
        if len(splits) != 1:
            raise ValueError('Source component crosses train/test: ' + ', '.join(c['id'] for c in component))
        ids = sorted(c['id'] for c in component)
        signature = hashlib.sha256(('\n'.join(ids)).encode()).hexdigest()
        # 20% is predeclared; hash does not depend on labels, model outputs, or outcomes.
        draw = int(hashlib.sha256((str(seed) + ':' + signature).encode()).hexdigest(), 16)
        partition = ('test' if next(iter(splits)) == 'test' else
                     'validation' if draw % 5 == 0 else 'train')
        for case in component:
            assignment[case['id']] = {'partition': partition, 'source_component': signature}
            counts[(partition, case['family'])] += 1
    return {'bank_sha256': hashlib.sha256(raw).hexdigest(),
            'topic_links_sha256': hashlib.sha256(topic_bytes).hexdigest(), 'seed': seed,
            'policy': 'Train-source components hashed to train/validation (80/20 in expectation); test families remain test.',
            'limitations': 'Exact source URLs and explicitly linked topics are grouped; unlisted shared facts still require review.',
            'by_partition_family': {f'{p}/{f}': n for (p, f), n in sorted(counts.items())},
            'cases': assignment}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bank', type=Path, default=Path(__file__).with_name('cases.json'))
    parser.add_argument('--out', type=Path, default=Path(__file__).with_name('partitions.json'))
    parser.add_argument('--topic-links', type=Path, default=Path(__file__).with_name('topic_links.json'))
    parser.add_argument('--seed', type=int, default=47)
    args = parser.parse_args()
    manifest = build(args.bank, args.seed, args.topic_links)
    with args.out.open('x', encoding='utf-8') as stream:
        json.dump(manifest, stream, indent=2)
        stream.write('\n')
    print(json.dumps(manifest['by_partition_family'], indent=2))


if __name__ == '__main__':
    main()
