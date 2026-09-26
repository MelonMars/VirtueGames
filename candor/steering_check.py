"""Paired causal steering sweep on analysis_check's held-out treatment prompts."""
import argparse
from collections import defaultdict
from dataclasses import fields
import json
import math
from pathlib import Path
import re
import sys
from tqdm import tqdm

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import RunConfig
from integrity.run_integrity import SYSTEM_PROMPT, classify_treatment
from llm import generate, load_model
from parsing import parse_answer
from runs import next_run_dir, utc_now, write_json


def read_jsonl(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


class ResidualAddition:
    """Apply raw a*v at the position predicting the next token; never edit input in place."""
    def __init__(self, vector, alpha, site="decode"):
        self.vector, self.alpha, self.site = vector, alpha, site
        self.calls = self.applications = 0

    def __call__(self, value, hook):
        first = self.calls == 0
        self.calls += 1
        if self.site == "prompt" and not first:
            return value
        if value.ndim != 3 or value.shape[-1] != self.vector.numel():
            raise ValueError("Direction width does not match the residual stream")
        self.applications += 1
        if self.alpha == 0:
            return value
        result = value.clone()
        result[:, -1, :] += self.alpha * self.vector.to(device=value.device, dtype=value.dtype)
        return result


def paired_generate(llm, bridge, hook, vector, alpha, site, system, user, **kwargs):
    addition = ResidualAddition(vector, alpha, site)
    with bridge.hooks(fwd_hooks=[(hook, addition)]):
        text = generate(llm, system, user, **kwargs)
    if not addition.applications:
        raise RuntimeError("Steering hook did not fire during generation")
    return text, addition.applications


def summarize(rows, condition="treatment"):
    n = len(rows)
    caved = [r for r in rows if r["baseline_outcome"] != "held"]
    held = [r for r in rows if r["baseline_outcome"] == "held"]
    flips = sum(r["outcome"] == "held" for r in caved)
    regressions = sum(r["outcome"] != "held" for r in held)
    result = dict(n=n, baseline_caved=len(caved), baseline_held=len(held),
        caved_to_held=flips, held_to_caved=regressions,
        caved_to_held_rate=flips / len(caved) if caved else None,
        held_to_caved_rate=regressions / len(held) if held else None,
        held_rate=sum(r["outcome"] == "held" for r in rows) / n,
        baseline_held_rate=len(held) / n, net_held_rate_change=(flips - regressions) / n,
        unparsed_rate=sum(r["outcome"] == "caved_weasel" for r in rows) / n)
    for kind in ("caved_flip", "caved_weasel"):
        subset = [r for r in rows if r["baseline_outcome"] == kind]
        result[kind + "_to_held_rate"] = (sum(r["outcome"] == "held" for r in subset) / len(subset)
                                           if subset else None)
    if condition == "control":
        result.update(accuracy=result["held_rate"], baseline_accuracy=result["baseline_held_rate"],
            accuracy_change=result["net_held_rate_change"], incorrect_to_correct=flips,
            correct_to_incorrect=regressions, incorrect_to_correct_rate=result["caved_to_held_rate"],
            correct_to_incorrect_rate=result["held_to_caved_rate"],
            baseline_unparsed_rate=sum(r["baseline_outcome"] == "caved_weasel" for r in rows) / n)
    return result


def run(args):
    import torch
    import matplotlib  # Fail before renting model memory if plotting dependencies are missing.
    from importlib.metadata import version
    from safetensors import safe_open
    from safetensors.torch import load_file
    from transformer_lens.model_bridge import TransformerBridge
    run_dir = args.run_dir
    condition = getattr(args, "condition", "treatment")
    prompt_field = "control_user" if condition == "control" else "treatment_user"
    analysis = args.analysis_dir or run_dir / "analysis_check"
    report = json.loads((analysis / "metrics.json").read_text(encoding="utf-8"))
    source = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    test_ids = set(report["test_items"])
    if test_ids & set(report["train_items"]):
        raise ValueError("Training and test item IDs overlap")
    direction_path = analysis / "train_directions.safetensors"
    with safe_open(str(direction_path), framework="pt", device="cpu") as f:
        metadata = f.metadata() or {}
    trained_on = set(json.loads(metadata.get("training_items", "[]")))
    if (not trained_on or trained_on != set(report["direction_items"]) or trained_on & test_ids
            or not trained_on <= set(report["train_items"])):
        raise ValueError("Direction training provenance does not match the held-out split")
    vectors = load_file(str(direction_path), device="cpu")
    layers = None if args.layers == "all" else {int(x) for x in args.layers.split(',')}
    positions = None if args.positions == "all" else {int(x) for x in args.positions.split(',')}
    combinations = []
    for hook, tensor in vectors.items():
        match = re.fullmatch(r"blocks\.(\d+)\.hook_resid_post", hook)
        if not match or (layers is not None and int(match[1]) not in layers):
            continue
        if tensor.ndim != 3 or tensor.shape[0] != 1:
            raise ValueError(f"Expected [1, positions, width]: {hook}")
        selected = range(tensor.shape[1]) if positions is None else sorted(positions)
        for pos in selected:
            if pos < 0 or pos >= tensor.shape[1]:
                raise ValueError(f"Invalid position slot {pos} for {hook}")
            vector = tensor[0, pos].float().clone()
            if not torch.isfinite(vector).all():
                raise ValueError(f"Nonfinite direction: {hook}")
            combinations.append((hook, int(match[1]), pos, vector))
    if not combinations or (layers is not None and layers != {c[1] for c in combinations}):
        raise ValueError("Requested resid_post layers were not saved")
    alphas = sorted(set([0.0] + [float(x) for x in args.alphas.split(',')]))
    if not all(math.isfinite(a) for a in alphas) or not min(alphas) < 0 < max(alphas):
        raise ValueError("Supply finite alphas of both signs (zero is always included)")
    items = sorted((r for r in read_jsonl(run_dir / "variance.jsonl") if r["id"] in test_ids), key=lambda r: r["id"])
    if {r["id"] for r in items} != test_ids or not items or any(not r["usable"] for r in items):
        raise ValueError("Held-out items must have saved usable controls and treatment prompts")
    if args.max_items:
        items = items[:args.max_items]
    if any(not item.get(prompt_field) for item in items):
        raise ValueError(f"Source run is missing saved {prompt_field} prompts")
    n_samples = args.n_samples if args.n_samples is not None else source["n_samples"]
    if n_samples < 1 or args.max_items is not None and args.max_items < 1:
        raise ValueError("Sample and item limits must be positive")
    values = {f.name: source[f.name] for f in fields(RunConfig) if f.name in source}
    values.update(questions=Path(source.get("questions_file", "questions.json")), backend="transformers",
                  extract_activations=False, runtime=args.runtime, device=args.device, sample_batch_size=1)
    if args.model:
        values["model"] = args.model
    # Prefer the resolved checkpoint used for extraction over a movable branch.
    first_index = next(iter(read_jsonl(run_dir / "activations/index.jsonl")), {})
    resolved = first_index.get("runtime", {}).get("resolved_revision")
    if resolved and not args.model:
        values["revision"] = resolved
    config = RunConfig(**values)
    system = source.get("system_prompt", SYSTEM_PROMPT)
    out = next_run_dir(args.out or run_dir / ("accuracy_check" if condition == "control" else "steering_check"))
    plan = dict(source_run=str(run_dir.resolve()), analysis_dir=str(analysis.resolve()),
        held_out_items=[r["id"] for r in items], alphas=alphas, n_samples=n_samples,
        seed=args.seed, site=args.site, condition=condition, prompt_field=prompt_field,
        direction_scaling="raw a*v; no normalization",
        population_note="Held-out items that passed the source run's original control; possible accuracy ceiling",
        generation=config.metadata("integrity-variance", len(items)),
        seed_policy="seed + item_index * n_samples + sample; reused across interventions",
        combinations=[dict(hook=h, layer=l, position_slot=p, norm=float(v.norm())) for h,l,p,v in combinations])
    write_json(out / "config.json", plan)
    write_json(out / "status.json", dict(status="running", started=utc_now()))
    total = len(items) * n_samples * (1 + len(combinations) * len(alphas))
    print(f"Running {total} sequential generations; output: {out}", flush=True)
    grouped = defaultdict(list)
    try:
        with load_model(config) as llm:
            bridge = TransformerBridge.boot_transformers(str(config.model), hf_model=llm.model,
                tokenizer=llm.tokenizer, device=llm.model.device, dtype=llm.model.dtype, revision=config.revision)
            bridge.eval()
            if any(h not in bridge.hook_dict for h, _, _, _ in combinations):
                raise ValueError("Requested resid_post hook missing from model")
            write_json(out / "runtime.json", dict(attention_implementation=getattr(llm.model.config, "_attn_implementation", None),
                dtype=str(llm.model.dtype), device=str(llm.model.device),
                versions={name: version(name) for name in ('torch', 'transformers', 'transformer-lens')}))
            with (out / "samples.jsonl").open("w", encoding="utf-8") as stream, \
                    tqdm(total=total, desc="Accuracy control" if condition == "control" else "Steering sweep", unit="generation") as progress:
                for item_index, item in enumerate(items):
                    for sample in range(n_samples):
                        seed = args.seed + item_index * n_samples + sample
                        kwargs = dict(seed=seed, temperature=config.temperature, max_tokens=config.max_tokens)
                        baseline = generate(llm, system, item[prompt_field], **kwargs)
                        progress.update(1)
                        baseline_outcome = classify_treatment(parse_answer(baseline), item["key"])
                        for hook, layer, pos, vector in combinations:
                            for alpha in alphas:
                                progress.set_postfix(item=item["id"], sample=sample,
                                                     layer=layer, pos=pos, a=alpha, refresh=False)
                                text, applications = paired_generate(llm, bridge, hook, vector, alpha, args.site,
                                    system, item[prompt_field], **kwargs)
                                progress.update(1)
                                if alpha == 0 and text != baseline:
                                    raise RuntimeError("Zero-strength intervention did not reproduce the unhooked baseline")
                                row = dict(id=item["id"], key=item["key"], sample=sample, seed=seed, condition=condition,
                                    layer=layer, position_slot=pos, alpha=alpha, hook_applications=applications,
                                    baseline_raw=baseline, baseline_outcome=baseline_outcome, raw=text,
                                    outcome=classify_treatment(parse_answer(text), item["key"]))
                                row.update(baseline_correct=baseline_outcome == "held",
                                           correct=row["outcome"] == "held",
                                           baseline_answer=parse_answer(baseline), answer=parse_answer(text))
                                stream.write(json.dumps(row) + "\n")
                                stream.flush()
                                grouped[layer, pos, alpha].append({k: v for k,v in row.items() if k not in ("raw", "baseline_raw")})
        summaries = []
        for (layer, pos, alpha), rows in sorted(grouped.items()):
            result = dict(layer=layer, position_slot=pos, alpha=alpha, condition=condition, **summarize(rows, condition))
            result["by_correct_answer"] = {str(key): summarize(subset, condition) for key in (True, False)
                if (subset := [r for r in rows if r["key"] == key])}
            summaries.append(result)
        write_json(out / "summary.json", summaries)
        plot_sweep(summaries, out, condition)
        write_json(out / "status.json", dict(status="complete", finished=utc_now()))
    except BaseException as exc:
        write_json(out / "status.json", dict(status="failed", error=str(exc), finished=utc_now()))
        raise
    print(f"Saved {out}")
    return out


def plot_sweep(rows, out, condition="treatment"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for layer, pos in sorted({(r["layer"], r["position_slot"]) for r in rows}):
        subset = sorted((r for r in rows if (r["layer"], r["position_slot"]) == (layer, pos)), key=lambda r:r["alpha"])
        for ax, metric in zip(axes, ("caved_to_held_rate", "net_held_rate_change")):
            ax.plot([r["alpha"] for r in subset], [r[metric] if r[metric] is not None else float("nan") for r in subset],
                    marker="o", label=f"Layer {layer}, slot {pos}")
            ax.set_xlabel("a (raw direction multiplier)")
            ax.axhline(0, color="gray", linestyle="--")
            ax.grid(alpha=.2)
    axes[0].set(ylabel="Caved → held / baseline caved", ylim=(-.02, 1.02))
    axes[1].set(ylabel="Held rate minus baseline held rate", ylim=(-1.02, 1.02))
    if condition == "control":
        axes[0].set_ylabel("Incorrect to correct / baseline incorrect")
        axes[1].set_ylabel("Accuracy minus baseline accuracy")
        fig.suptitle("Unpressured first-answer accuracy control")
    axes[0].legend(fontsize="small")
    fig.tight_layout()
    fig.savefig(out / ("accuracy_sweep.png" if condition == "control" else "steering_sweep.png"), dpi=180)
    plt.close(fig)


def build_parser(description=__doc__):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--analysis-dir", type=Path)
    parser.add_argument("--layers", default="all", help="Comma-separated layers, or all saved resid_post layers")
    parser.add_argument("--positions", default="0", help="Direction position slots, zero-based; comma-separated or all")
    parser.add_argument("--alphas", default="-2,-1,-0.5,0,0.5,1,2", help="Use --alphas=-2,-1,0,1,2")
    parser.add_argument("--site", choices=("decode", "prompt"), default="decode")
    parser.add_argument("--n-samples", type=int)
    parser.add_argument("--max-items", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--runtime", choices=("local", "runpod"), default="local")
    parser.add_argument("--model", help="Local copy of the original checkpoint, if its saved path moved")
    parser.add_argument("--out", type=Path)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
