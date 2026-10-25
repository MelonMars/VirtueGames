"""Wait for the train-only evidence resample, extract, and validate both directions."""
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from runs import read_jsonl, write_json

RUN = ROOT / 'evidence/runs/runs_evidence_reason_resample/run-00'
REPORT = ROOT / 'logs/evidence_reason_20261011_vector_report.json'
ALPHAS = (-1.0, -0.5, 0.0, 0.5, 1.0)


def steering_metrics(directory, target_label):
    rows = read_jsonl(directory / 'samples.jsonl')
    by_id = {}
    for row in rows:
        by_id.setdefault(row['id'], {})[float(row['alpha'])] = row
    complete = {item: values for item, values in by_id.items()
                if all(alpha in values and values[alpha]['scored'] for alpha in ALPHAS)}
    report = {'n_paired_validation_items': len(complete), 'target_failure': target_label,
              'alphas': {}, 'working_alpha': None, 'working_on_held_out_validation': False}
    for alpha in ALPHAS:
        scores = [v[alpha]['scalar_score'] for v in complete.values()]
        failures = sum(v[alpha]['label'] == target_label for v in complete.values())
        report['alphas'][str(alpha)] = dict(n=len(scores), mean_scalar_score=sum(scores)/len(scores) if scores else None,
            targeted_failures=failures, paired_delta_vs_zero=(
                sum(v[alpha]['scalar_score']-v[0.0]['scalar_score'] for v in complete.values())/len(scores)
                if scores else None))
    candidates = [a for a in ALPHAS if a != 0.0 and report['alphas'][str(a)]['n'] >= 3
                  and report['alphas'][str(a)]['paired_delta_vs_zero'] > 0
                  and report['alphas'][str(a)]['targeted_failures'] < report['alphas']['0.0']['targeted_failures']]
    if candidates:
        best = max(candidates, key=lambda a: report['alphas'][str(a)]['paired_delta_vs_zero'])
        report['working_alpha'] = best
        report['working_on_held_out_validation'] = True
    return report


def main():
    deadline = time.monotonic() + 7200
    while time.monotonic() < deadline:
        status = json.loads((RUN / 'status.json').read_text())
        if status['status'] == 'complete':
            break
        if status['status'] == 'failed':
            raise RuntimeError(status)
        time.sleep(10)
    else:
        raise TimeoutError('Train resample did not finish within two hours')
    summary = json.loads((RUN / 'summary.json').read_text())
    result = {'run': str(RUN), 'resample_summary': summary,
              'contrastive_items': read_jsonl(RUN / 'variance.jsonl')}
    if not summary['phase1_ready']:
        result.update(confirmed_working_vector=False,
                      conclusion='The train resample still lacks both within-item negative contrasts.')
        write_json(REPORT, result)
        return result
    extracted = subprocess.run([sys.executable, 'evidence/analyze_directions.py', str(RUN)],
                               cwd=ROOT, capture_output=True, text=True, check=True)
    direction_root = RUN / 'directions_posterior'
    directions = {}
    for label in ('under_update', 'over_update'):
        output = subprocess.run([sys.executable, 'evidence/steering_check.py', str(RUN),
            str(direction_root / (label + '.safetensors')), '--layer', '19', '--split', 'val',
            '--device', 'cuda', '--alphas', '-1,-0.5,0,0.5,1'], cwd=ROOT,
            capture_output=True, text=True, check=True)
        path = Path(output.stdout.strip().splitlines()[-1])
        directions[label] = dict(extraction_report=json.loads((direction_root / 'report.json').read_text()),
            steering_directory=str(path), validation=steering_metrics(path, label))
    result.update(directions=directions,
        confirmed_working_vector=any(x['validation']['working_on_held_out_validation'] for x in directions.values()),
        conclusion='A direction counts as working only if a nonzero alpha improves paired scalar score and reduces its target failure on at least three held-out validation families.')
    write_json(REPORT, result)
    return result


if __name__ == '__main__':
    print(json.dumps(main(), indent=2))
