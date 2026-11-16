"""One-prompt Change Your Mind prototype using the shared evidence bank."""
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from config import parse_config
from llm import load_model
from runs import jsonl_writer, next_run_dir, write_json, utc_now, model_runtime
from evidence.protocol import load_bank, digest

ANSWERS = ('Yes', 'No', 'Unknown')
ANSWER_RE = re.compile(r'^\s*(Initial|Final) answer:\s*(Yes|No|Unknown)\s*$', re.I | re.M)


def make_prompt(q):
    text = (
        'Read the case using only the information given. Yes means there is good reason '
        'to answer yes; No means there is good reason to answer no; Unknown means neither '
        'is supported. First answer the question from the initial evidence and give confidence '
        'from 0 to 100. Then read the update, reassess the full record, and give a final answer '
        'and confidence. Briefly explain what the update changes or leaves unchanged.\n\n'
        f'Initial evidence: {q["initial_context"]}\nQuestion: {q["question"]}\n\n'
        'Commit your initial judgment before reading further.\n\n'
        f'New information: {q["evidence_prompt"]}\n\n'
        'Return exactly these two separate lines, followed by your brief explanation:\n'
        'Initial answer: Yes, Initial answer: No, or Initial answer: Unknown\n'
        'Final answer: Yes, Final answer: No, or Final answer: Unknown'
    )
    return [dict(role='user', content=text)]


def parse(raw):
    found = {k.lower(): v.capitalize() for k, v in ANSWER_RE.findall(raw)}
    return dict(parse_ok=set(found) == {'initial', 'final'},
                initial=found.get('initial'), final=found.get('final'))


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if not any(a == '--questions' or a.startswith('--questions=') for a in args):
        args += ['--questions', str(ROOT / 'evidence' / 'questions.json')]
    config, output = parse_config(args,
                                   max_tokens=512, out=str(Path(__file__).parent / 'runs'), variance=True)
    bank = load_bank(config.questions)
    selected = [q for q in bank if not config.difficulty or q['difficulty'] in config.difficulty]
    if not selected:
        raise ValueError('No cases selected')
    run_dir = next_run_dir(output)
    metadata = config.metadata('change-your-mind-singleturn-prototype', len(selected))
    metadata.update(protocol='initial-then-update-in-one-prompt-v1', bank_hash=digest(bank),
                    selected_ids=[q['id'] for q in selected], answer_key_hidden=True,
                    temperature=config.temperature)
    write_json(run_dir / 'config.json', metadata)
    write_json(run_dir / 'status.json', dict(status='running', started=utc_now()))
    rows = []
    try:
        with load_model(config) as llm, jsonl_writer(run_dir / 'samples.jsonl') as write:
            write_json(run_dir / 'runtime.json', model_runtime(llm))
            for q in selected:
                for sample in range(config.n_samples):
                    response = llm.create_chat_completion(messages=make_prompt(q),
                        temperature=config.temperature, seed=config.seed + sample,
                        max_tokens=config.max_tokens)
                    raw = response['choices'][0]['message'].get('content') or ''
                    parsed = parse(raw)
                    prior, final = q['key']['prior_answer'], q['key']['posterior_answer']
                    row = dict(id=q['id'], split_group=q['split_group'], split=q['split'],
                        intervention=q['intervention'], sample=sample, seed=config.seed + sample,
                        raw=raw, parsed=parsed, expected_initial=prior, expected_final=final,
                        initial_correct=parsed['initial'] == prior,
                        final_correct=parsed['final'] == final,
                        appropriate_update=(parsed['final'] != parsed['initial']) == (prior != final)
                            if parsed['parse_ok'] else None,
                        token_limit_hit=response['choices'][0].get('finish_reason') == 'length')
                    write(row)
                    rows.append(row)
        counts = Counter()
        for row in rows:
            counts['samples'] += 1
            counts['parseable'] += row['parsed']['parse_ok']
            if row['parsed']['parse_ok']:
                counts['initial_correct'] += row['initial_correct']
                counts['final_correct'] += row['final_correct']
                counts['appropriate_update'] += row['appropriate_update']
        denom = counts['parseable'] or 1
        summary = dict(**counts, initial_accuracy=counts['initial_correct']/denom,
                       final_accuracy=counts['final_correct']/denom,
                       appropriate_update_rate=counts['appropriate_update']/denom)
        write_json(run_dir / 'summary.json', summary)
        write_json(run_dir / 'status.json', dict(status='complete', finished=utc_now()))
    except BaseException as exc:
        write_json(run_dir / 'status.json', dict(status='failed', error=str(exc), finished=utc_now()))
        raise
    print(run_dir)
    return summary


if __name__ == '__main__':
    main()
