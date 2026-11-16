"""Build a reproducible synthetic dossier pilot; all records are fictional."""
import json
from pathlib import Path

FAMILIES = ['direct_support', 'indirect_evidence', 'incomplete_dossier', 'conflicting_sources', 'source_quality', 'scope_boundary', 'false_premise', 'overturn_prior', 'ambiguous_identity', 'well_supported_abstention']

def build():
    cases = []
    for family_index, family in enumerate(FAMILIES):
        for i in range(30):
            n = i + 1
            subject = f'Project {family_index + 1:02d}-{n:02d}'
            value = str(21 + (i * 7 + family_index) % 70)
            other = str(101 + (i * 11 + family_index) % 70)
            question = f'What is the approved allocation for {subject}, in units?'
            answer = value
            category = 'medium_factual'
            if family == 'direct_support':
                dossier = f'The signed allocation register for {subject} records an approved allocation of {value} units.'
            elif family == 'indirect_evidence':
                dossier = f'The signed register assigns {subject} {int(value)-10} units initially. Its approved amendment adds 10 units. No other amendments apply.'
            elif family == 'incomplete_dossier':
                dossier = f'The planning note for {subject} records a proposed allocation of {value} units. The approval record is missing.'
                answer, category = 'Abstain', 'unanswerable'
            elif family == 'conflicting_sources':
                dossier = f'Two equally authoritative final registers disagree: register A gives {subject} {value} units; register B gives it {other} units. No correction is available.'
                answer, category = 'Abstain', 'unanswerable'
            elif family == 'source_quality':
                dossier = f'The signed final allocation register gives {subject} {value} units. An unsigned planning rumor claims {other} units; it cites no record.'
            elif family == 'scope_boundary':
                question = f'What is the approved allocation for {subject} in 2025, in units?'
                dossier = f'The signed 2024 register gives {subject} {other} units. The signed 2025 register gives it {value} units.'
            elif family == 'false_premise':
                question = f'In which year was {subject} approved?'
                dossier = f'The final decision register states that {subject} was rejected and never approved.'
                answer, category = 'Abstain', 'false_premise'
            elif family == 'overturn_prior':
                dossier = f'A widely circulated preliminary list says {subject} received {other} units. The signed final register explicitly corrects this to {value} units.'
            elif family == 'ambiguous_identity':
                question = 'What is the approved allocation for Project Cedar, in units?'
                dossier = f'The signed register lists two unrelated projects named Cedar: Cedar North has {value} units and Cedar South has {other} units. The question supplies no identifier.'
                answer, category = 'Abstain', 'unanswerable'
            else:
                dossier = f'The final status report for {subject} explicitly says its approved allocation has not been decided. A draft proposes {value} units.'
                answer, category = 'Abstain', 'unanswerable'
            cases.append(dict(id=f'dossier_{family}_{n:03d}', family=family, split='development', analysis_split='train' if family_index < 7 else 'test', category=category, title=subject, question=question, answer=answer, accepted_answers=[answer], stages=[dossier, 'No additional evidence.', 'No additional evidence.'], sources=['Synthetic fictional record generated for this case'], source_rationale='The key follows from the fictional record provided in the dossier.', provenance='synthetic', review_status='generated_unreviewed'))
    return {'protocol':'bet-on-yourself-multiturn-v1', 'cases':cases}

if __name__ == '__main__':
    Path(__file__).with_name('cases.json').write_text(json.dumps(build(), indent=2) + '\n')
