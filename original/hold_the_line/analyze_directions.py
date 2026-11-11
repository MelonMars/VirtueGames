"""Extract train-only Hold The Line residual directions."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from evidence.analyze_directions import main as analyze_main


if __name__ == '__main__':
    args = list(sys.argv[1:])
    if not any(a == '--interventions' or a.startswith('--interventions=') for a in args):
        args += ['--interventions', 'reason_intact']
    if not any(a == '--negatives' or a.startswith('--negatives=') for a in args):
        args += ['--negatives', 'over_update']
    analyze_main(args)
