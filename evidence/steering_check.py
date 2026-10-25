"""Held-out evidence sweep using integrity's raw resid_post addition hook."""
import sys
from pathlib import Path
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import hashlib
import json
import math
from dataclasses import fields
from tqdm import tqdm
from config import RunConfig
from llm import load_model
from runs import next_run_dir, jsonl_writer, write_json, utc_now
from integrity.steering_check import ResidualAddition
from evidence.protocol import PROTOCOL, load_bank, digest, messages, control_messages
from evidence.run_evidence import complete
from evidence.score_evidence import SCORING_VERSION, SUPPORT_COORDINATES, control_passes, score_sample, summarize


def run(args):
    from safetensors import safe_open
    from safetensors.torch import load_file
    from transformer_lens.model_bridge import TransformerBridge
    source = json.loads((args.run_dir / 'config.json').read_text())
    if source.get('protocol') != PROTOCOL:
        raise ValueError('Incompatible source protocol')
    bank = load_bank(args.run_dir / 'questions.json')
    splits = {q['id']: [q['split_group'],q['split']] for q in bank}
    if digest(bank)!=source['bank_hash'] or digest(splits)!=source['split_hash']:
        raise ValueError('Source provenance mismatch')
    with safe_open(str(args.direction),framework='pt',device='cpu') as f:
        metadata=f.metadata()
    trained=json.loads(metadata['training_items'])
    if not trained or metadata['split_hash']!=source['split_hash'] or metadata['config_hash']!=source['config_hash']:
        raise ValueError('Direction provenance mismatch')
    if any(i not in splits or splits[i][1]!='train' for i in trained):
        raise ValueError('Direction includes non-training items')
    items=[q for q in bank if q['split']==args.split]
    if not items or {splits[i][0] for i in trained} & {q['split_group'] for q in items}:
        raise ValueError('Missing held-out items or family leakage')
    hook=f'blocks.{args.layer}.hook_resid_post'
    vector=load_file(str(args.direction))[hook][0,args.slot].float()
    if not vector.isfinite().all():
        raise ValueError('Nonfinite vector')
    alphas=sorted(set([0.]+[float(a) for a in args.alphas.split(',')]))
    if not all(math.isfinite(a) for a in alphas) or not min(alphas)<0<max(alphas):
        raise ValueError('Supply finite alphas of both signs')
    values={f.name:source[f.name] for f in fields(RunConfig) if f.name in source}
    runtime=json.loads((args.run_dir / 'runtime.json').read_text())
    values.update(questions=args.run_dir / 'questions.json', backend='transformers', extract_activations=False,
                  device=args.device, seed=source['base_seed'], revision=runtime.get('resolved_revision') or source['revision'])
    config=RunConfig(**values)
    out=next_run_dir(args.out or args.run_dir / 'steering_check')
    plan=dict(scoring_version=SCORING_VERSION, support_coordinates=SUPPORT_COORDINATES, source_config_hash=source['config_hash'], split_hash=source['split_hash'],
        direction=str(args.direction.resolve()), direction_sha256=hashlib.sha256(args.direction.read_bytes()).hexdigest(),
        split=args.split, held_out_ids=[q['id'] for q in items], alphas=alphas, hook=hook, slot=args.slot,
        seed_policy='base_seed for every paired intervention; deterministic temperature 0', scaling='raw', site='decode')
    write_json(out / 'config.json',plan)
    write_json(out / 'status.json',dict(status='running',started=utc_now()))
    rows=[]
    try:
        with load_model(config) as llm, jsonl_writer(out / 'samples.jsonl') as write:
            bridge=TransformerBridge.boot_transformers(str(config.model),hf_model=llm.model,tokenizer=llm.tokenizer,
                device=llm.model.device,dtype=llm.model.dtype,revision=config.revision)
            bridge.eval()
            def generate(msg):
                return complete(llm,msg,config,temperature=0.,seed=config.seed,context=None,capture=False)
            for q in tqdm(items,desc='Evidence steering'):
                control=generate(control_messages(q))
                prior=generate(messages(q))
                usable=control_passes(q,control)
                conversation=messages(q,prior['raw'])
                baseline=generate(conversation)
                for alpha in alphas:
                    addition=ResidualAddition(vector,alpha,'decode')
                    with bridge.hooks(fwd_hooks=[(hook,addition)]):
                        posterior=generate(conversation)
                    if not addition.applications:
                        raise AssertionError('Steering hook did not fire')
                    if alpha==0 and (posterior['raw']!=baseline['raw'] or posterior['token_limit_hits']!=baseline['token_limit_hits']):
                        raise AssertionError('alpha == 0 differs from unhooked baseline')
                    hits=prior['token_limit_hits']+posterior['token_limit_hits']
                    row=dict(id=q['id'],intervention=q['intervention'],difficulty=q['difficulty'],split_group=q['split_group'],
                        condition='neutral',alpha=alpha,prior=prior,posterior=posterior,baseline=baseline,control=control,
                        usable=usable,parse_ok=prior['parsed']['parse_ok'] and posterior['parsed']['parse_ok'],token_limit_hits=hits)
                    row.update(score_sample(q,prior['parsed'],posterior['parsed'],usable=usable,truncated=bool(hits)))
                    write(row)
                    rows.append(row)
        write_json(out / 'summary.json',{str(a):summarize([r for r in rows if r['alpha']==a]) for a in alphas})
    except BaseException as exc:
        write_json(out / 'status.json',dict(status='failed',error=str(exc),finished=utc_now()))
        raise
    write_json(out / 'status.json',dict(status='complete',finished=utc_now()))
    return out


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('run_dir',type=Path)
    p.add_argument('direction',type=Path)
    p.add_argument('--layer',type=int,required=True)
    p.add_argument('--slot',type=int,choices=(0,1),default=1)
    p.add_argument('--alphas',default='-1,-0.5,0,0.5,1')
    p.add_argument('--split',choices=('val','test'),default='val')
    p.add_argument('--device',choices=('auto','cpu','cuda'),default='auto')
    p.add_argument('--out',type=Path)
    print(run(p.parse_args()))
