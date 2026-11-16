"""Run the synthetic dossier bank with the existing answer-first BOY runner."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from original.bet_on_yourself.singleturn.run_commit_confidence import main

if __name__ == '__main__':
    args = sys.argv[1:]
    if not any(x == '--questions' or x.startswith('--questions=') for x in args):
        args += ['--questions', str(Path(__file__).parent / 'challenge_bank' / 'cases.json')]
    if not any(x == '--out' or x.startswith('--out=') for x in args):
        args += ['--out', 'original/bet_on_yourself/dossier/runs']
    main(args)
