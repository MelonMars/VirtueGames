"""Evaluate a Change Your Mind direction on held-out families."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from evidence.steering_check import main as steering_main

INTERVENTIONS = 'reason_removed,reason_replaced,opposite_evidence,irrelevant'


if __name__ == '__main__':
    args = list(sys.argv[1:])
    if not any(a == '--interventions' or a.startswith('--interventions=') for a in args):
        args += ['--interventions', INTERVENTIONS]
    steering_main(args)
