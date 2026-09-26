"""Compare matching residual-space directions from two analysis directories."""
import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file


def cosine(a, b):
    if a.ndim != 1 or a.shape != b.shape:
        raise ValueError("Expected equal-width one-dimensional vectors")
    a, b = a.double(), b.double()
    if not torch.isfinite(a).all() or not torch.isfinite(b).all() or a.norm() == 0 or b.norm() == 0:
        raise ValueError("Cosine requires finite nonzero vectors")
    return float(torch.dot(a/a.norm(), b/b.norm()).clamp(-1, 1))


def compare(first, second, layer=19):
    paths = [Path(first), Path(second)]
    reports = [json.loads((p / "report.json").read_text(encoding="utf-8")) for p in paths]
    configs = [r["source_config"] for r in reports]
    identities = [r.get("capture_identity", {}) for r in reports]
    resolved = [i.get("resolved_revision") for i in identities]
    if all(resolved) and resolved[0] != resolved[1]:
        raise ValueError("Resolved checkpoint revisions differ; residual spaces are not comparable")
    models = [i.get("model") or c["model"] for i,c in zip(identities, configs)]
    if models[0] != models[1] and not (all(resolved) and resolved[0] == resolved[1]):
        raise ValueError("Model identities differ; matching widths do not establish a shared coordinate space")
    if configs[0].get("revision") != configs[1].get("revision") and not all(resolved):
        raise ValueError("Checkpoint revisions differ")
    warnings = []
    if not all(resolved):
        warnings.append("Resolved checkpoint identity is unavailable; confirm both runs used identical weights.")
    if configs[0].get("thinking") != configs[1].get("thinking"):
        warnings.append("Thinking settings differ; full-response pooling includes reasoning tokens when generated.")
    if any(c.get("activation_positions") not in ("response", "all") for c in configs):
        warnings.append("Capture settings do not establish full-response coverage; check the source activation indices.")
    hook = f"blocks.{layer}.hook_resid_post"
    vectors = [load_file(str(p / "directions.safetensors"), device="cpu")[hook] for p in paths]
    value = cosine(*vectors)
    return dict(hook=hook, cosine=value, angle_degrees=float(torch.rad2deg(torch.acos(torch.tensor(value, dtype=torch.float64)))),
        first=str(paths[0]), second=str(paths[1]), directions=[r["direction"] for r in reports],
        norms=[float(v.double().norm()) for v in vectors], warnings=warnings,
        interpretation="Positive: aligned; zero: orthogonal; negative: opposed. Geometry alone does not establish shared causal virtue mechanisms.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("first", type=Path, help="First analysis directory")
    p.add_argument("second", type=Path, help="Second analysis directory")
    p.add_argument("--layer", type=int, default=19)
    p.add_argument("--out", type=Path, help="Optional new JSON result file")
    args = p.parse_args()
    result = json.dumps(compare(args.first, args.second, args.layer), indent=2)
    if args.out:
        with args.out.open("x", encoding="utf-8") as stream:
            stream.write(result + "\n")
    print(result)
