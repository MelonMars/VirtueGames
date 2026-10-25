"""Temperature-sampled posteriors, with one deterministic prior and control per item."""
import sys
from pathlib import Path
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evidence.run_evidence import main

if __name__ == '__main__':
    main(variance=True)
