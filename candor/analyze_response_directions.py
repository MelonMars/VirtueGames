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
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--no-progress", action="store_true")
    p.add_argument("--split-manifest", type=Path, required=True)
    p.add_argument("--pooling", choices=("response", "first-response", "last-prompt-token"), default="response")
    p.add_argument("--activation-dir", type=Path, help="Relocated original activations directory")
    args = p.parse_args()
    analyze(args.run_dir, args.out, device=args.device,
            progress=not args.no_progress, virtue="candor", layer=args.layer, split_manifest=args.split_manifest, pooling=args.pooling, activation_dir=args.activation_dir)
