"""Fit a full-response-mean candor direction at zero-based layer 19 by default."""
import argparse
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from response_direction_analysis import analyze


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir", type=Path)
    p.add_argument("--out", type=Path)
    p.add_argument("--layer", type=int, default=19)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--validation-fraction", type=float, default=.25)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--no-progress", action="store_true")
    args = p.parse_args()
    analyze(args.run_dir, args.out, args.seed, args.validation_fraction, device=args.device,
            progress=not args.no_progress, virtue="candor", layer=args.layer)
