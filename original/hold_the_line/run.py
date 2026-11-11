"""Run Hold The Line episodes from the shared evidence bank."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from evidence.run_evidence import main as run_main


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if not any(a == '--interventions' or a.startswith('--interventions=') for a in args):
        args += ['--interventions', 'reason_intact']
    if not any(a == '--out' or a.startswith('--out=') for a in args):
        args += ['--out', str(Path(__file__).parent / 'runs')]
    return run_main(args, variance=True)


if __name__ == '__main__':
    main()
