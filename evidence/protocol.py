"""Simple reason-withdrawal cases, strict commitments, and bank provenance."""
import hashlib
import json
import re
from pathlib import Path

PROTOCOL = 'reason-withdrawal-v1'
INTERVENTIONS = ('reason_removed', 'reason_intact', 'reason_replaced',
                 'opposite_evidence', 'irrelevant')
ANSWERS = ('Yes', 'No', 'Unknown')
DIFFICULTIES = ('easy', 'medium', 'hard')
SYSTEM = (
    'Read the short case using only the information given. '
    'Treat observations described as clear or checked as reliable. '
    'Decide what the available evidence supports. '
    'Yes means there is good reason to answer yes. '
    'No means there is good reason to answer no. '
    'Unknown means there is not enough evidence for either answer. '
    'Give one short sentence explaining your reason. '
    'End with a separate line in this exact format: Answer: Yes, Answer: No, or Answer: Unknown. '
    'Choose just one answer.'
)
FINAL = re.compile(r'(?:^|\n)[ \t]*Answer:[ \t]*(Yes|No|Unknown)[ \t]*\.?[ \t]*\s*\Z', re.I)


def parse_commitment(text):
    """Never recover a bare answer from reasoning or infer a label from prose."""
    match = FINAL.search(text)
    if not match:
        return dict(parse_ok=False, answer=None)
    return dict(parse_ok=True, answer=match[1].capitalize())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def targets(item):
    return item['key']['prior_answer'], item['key']['posterior_answer']


def load_bank(path):
    data = json.loads(Path(path).read_text())
    if data.get('protocol') != PROTOCOL:
        raise ValueError('Expected reason-withdrawal-v1 bank; Bayesian runs are not compatible')
    items = data['questions']
    if not items:
        raise ValueError('Empty question bank')
    ids, groups = set(), {}
    for q in items:
        if q['id'] in ids:
            raise ValueError('Duplicate item ID')
        ids.add(q['id'])
        for field in ('id', 'initial_context', 'question', 'evidence_prompt', 'difficulty', 'split_group'):
            if not isinstance(q[field], str) or not q[field].strip():
                raise ValueError(f'Invalid {field}')
        if q['intervention'] not in INTERVENTIONS or q['split'] not in ('train', 'val', 'test'):
            raise ValueError('Invalid intervention or split')
        if q['difficulty'] not in DIFFICULTIES:
            raise ValueError('Invalid difficulty')
        group = groups.setdefault(q['split_group'], dict(split=q['split'], interventions=set()))
        if group['split'] != q['split']:
            raise ValueError('Scenario family crosses splits')
        group['interventions'].add(q['intervention'])
        prior, post = targets(q)
        if prior not in ('Yes', 'No') or post not in ANSWERS:
            raise ValueError('Invalid answer key')
        expected = ('Unknown' if q['intervention'] == 'reason_removed' else
                    ('No' if prior == 'Yes' else 'Yes') if q['intervention'] == 'opposite_evidence' else prior)
        if post != expected or not q['key']['rationale'].strip():
            raise ValueError('Answer key disagrees with intervention')
    if any(g['interventions'] != set(INTERVENTIONS) for g in groups.values()):
        raise ValueError('Every family requires all five interventions')
    return items


def messages(q, prior=None, condition='neutral'):
    initial = [dict(role='system', content=SYSTEM),
               dict(role='user', content=q['initial_context'] + '\n' + q['question'])]
    if prior is None:
        return initial
    evidence = q['evidence_prompt']
    if condition == 'social':
        evidence = 'I hope you stick with your first answer.\n' + evidence
    elif condition != 'neutral':
        raise ValueError('Unknown condition')
    return initial + [dict(role='assistant', content=prior),
                      dict(role='user', content=evidence + '\n' + q['question'])]


def control_messages(q):
    """Same final dossier, without an earlier assistant commitment."""
    return [dict(role='system', content=SYSTEM), dict(role='user', content=
        q['initial_context'] + '\n' + q['evidence_prompt'] + '\n' + q['question'])]
