"""Finish the active remote pilot, relocate its outputs, and fit available contrasts."""
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from maintenance.reorganize_runs import reorganize
from evidence.score_evidence import rescore
from runs import write_json


def main():
    active = ROOT / 'runs_evidence_reason_variance/run-00'
    destination = ROOT / 'evidence/runs/runs_evidence_reason_variance/run-00'
    deadline = time.monotonic() + 7200
    while time.monotonic() < deadline:
        current = active if active.exists() else destination
        state = json.loads((current / 'status.json').read_text())
        if state['status'] == 'complete':
            break
        if state['status'] == 'failed':
            raise RuntimeError(state)
        time.sleep(10)
    else:
        raise TimeoutError('Variance run did not finish within two hours')
    reorganize(ROOT, apply=True)
    measurement = ROOT / 'evidence/runs/runs_evidence_reason_measurement/run-00'
    rescore(measurement)
    variance = json.loads((destination / 'summary.json').read_text())
    result = dict(measurement_directory=str(measurement), variance_directory=str(destination),
                  measurement=json.loads((measurement / 'rescored_summary.json').read_text()), variance=variance)
    if variance['phase1_ready']:
        from evidence.analyze_directions import analyze
        result['directions_directory'] = str(analyze(destination))
    else:
        result['directions_directory'] = None
        result['extraction_note'] = 'Insufficient usable contrastive train items for both negative sets'
    write_json(ROOT / 'logs/evidence_reason_20261011_results.json', result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('measurement','variance')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
