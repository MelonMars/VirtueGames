"""Evaluate a Hold The Line direction on held-out families."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from evidence.steering_check import main as steering_main


if __name__ == '__main__':
    args = list(sys.argv[1:])
    if not any(a == '--interventions' or a.startswith('--interventions=') for a in args):
        args += ['--interventions', 'reason_intact']
    steering_main(args)
