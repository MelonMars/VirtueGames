"""Extract train-only Change Your Mind residual directions."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from evidence.analyze_directions import main as analyze_main

INTERVENTIONS = 'reason_removed,reason_replaced,opposite_evidence,irrelevant'


if __name__ == '__main__':
    args = list(sys.argv[1:])
    if not any(a == '--interventions' or a.startswith('--interventions=') for a in args):
        args += ['--interventions', INTERVENTIONS]
    if not any(a == '--negatives' or a.startswith('--negatives=') for a in args):
        args += ['--negatives', 'under_update,over_update']
    analyze_main(args)
