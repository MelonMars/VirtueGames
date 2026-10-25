"""Item-balanced, train-only contrasts at shared answer-token slots."""
import sys
from pathlib import Path
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import json
from collections import Counter, defaultdict
from runs import read_jsonl, write_json
from evidence.protocol import PROTOCOL, load_bank, digest


def analyze(run_dir, *, variant='posterior'):
    import torch
    from safetensors.torch import load_file, save_file
    run_dir = Path(run_dir)
    config = json.loads((run_dir / 'config.json').read_text())
    if json.loads((run_dir / 'status.json').read_text())['status'] != 'complete':
        raise ValueError('A completed run is required')
    if config.get('protocol') != PROTOCOL:
        raise ValueError('Incompatible source protocol')
    if config['mode'] != 'evidence-variance' or config['condition'] != 'neutral':
        raise ValueError('Only neutral variance runs may feed clean extraction')
    bank = load_bank(run_dir / 'questions.json')
    if digest(bank) != config['bank_hash']:
        raise ValueError('Question provenance mismatch')
    splits = {q['id']: [q['split_group'], q['split']] for q in bank}
    if digest(splits) != config['split_hash']:
        raise ValueError('Split provenance mismatch')
    eligible = {i['id'] for i in read_jsonl(run_dir / 'variance.jsonl') if i['contrastive'] and splits[i['id']][1]=='train'}
    rows = [r for r in read_jsonl(run_dir / 'samples.jsonl') if r['id'] in eligible and r['scored'] and r['condition']=='neutral']
    index = {(r['context']['id'],r['context']['condition'],r['context']['sample']):r
             for r in read_jsonl(run_dir / 'activations/index.jsonl')}
    def state(qid, phase, sample):
        rec = index[(qid,phase,sample)]
        if rec['context'].get('answer_protocol') != PROTOCOL:
            raise ValueError('Expected strict reason-withdrawal answer positions')
        if not rec['file'] or rec['position_selection'] != 'answer-tokens-1,answer-tokens':
            raise ValueError('Missing shared answer-token slots')
        values = load_file(str(run_dir / 'activations' / rec['file']))
        if any(t.ndim != 3 or t.shape[:2] != (1,2) or not torch.isfinite(t).all() for t in values.values()):
            raise ValueError('Invalid activation tensor')
        return {k:t.float() for k,t in values.items()}
    per_item = defaultdict(lambda: defaultdict(list))
    for r in rows:
        values = state(r['id'],'posterior',r['sample'])
        if variant == 'delta':
            prior = state(r['id'],'prior',0)
            values = {k:v-prior[k] for k,v in values.items()}
        per_item[r['id']][r['label']].append(values)
    out = run_dir / ('directions_' + variant)
    out.mkdir(exist_ok=True)
    vectors, provenance = {}, {}
    for negative in ('under_update','over_update'):
        differences = defaultdict(list)
        ids = []
        for qid, labels in per_item.items():
            if not labels['correct_update'] or not labels[negative]:
                continue
            ids.append(qid)
            for hook in labels['correct_update'][0]:
                good = torch.stack([v[hook] for v in labels['correct_update']]).mean(0)
                bad = torch.stack([v[hook] for v in labels[negative]]).mean(0)
                differences[hook].append(good-bad)
        if not ids:
            raise ValueError(f'No train contrastive items for {negative}; Phase 1 is not ready')
        vectors[negative] = {k:torch.stack(v).mean(0) for k,v in differences.items()}
        provenance[negative] = sorted(ids)
    cosine = {}
    for hook in vectors['under_update']:
        a,b = vectors['under_update'][hook],vectors['over_update'][hook]
        cosine[hook] = [None if x.norm()==0 or y.norm()==0 else float(torch.nn.functional.cosine_similarity(x,y,dim=0))
                        for x,y in zip(a[0],b[0])]
    for negative, values in vectors.items():
        save_file(values, str(out / (negative+'.safetensors')), metadata=dict(
            training_items=json.dumps(provenance[negative]), split_hash=config['split_hash'],
            config_hash=config['config_hash'], variant=variant, scaling='raw', slots='answer-token-1,answer-token'))
    answer_counts = {label: dict(Counter(r['posterior']['parsed']['answer'] for r in rows if r['label'] == label))
                     for label in ('correct_update', 'under_update', 'over_update')}
    write_json(out / 'report.json', dict(answer_counts_by_label=answer_counts, protocol=PROTOCOL, variant=variant, training_items=provenance,
        train_groups=sorted({splits[i][0] for ids in provenance.values() for i in ids}),
        config_hash=config['config_hash'], split_hash=config['split_hash'], cosine_by_hook_and_slot=cosine,
        interpretation='Near -1 supports one signed axis; near zero supports distinct axes. Answer vocabulary and suspension frequency are confounds; geometry alone does not establish a virtue.'))
    return out


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('run_dir',type=Path)
    p.add_argument('--variant',choices=('posterior','delta'),default='posterior')
    a=p.parse_args()
    print(analyze(a.run_dir,variant=a.variant))
