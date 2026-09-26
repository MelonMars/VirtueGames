"""Sweep candor directions on validation families with paired generation and judging."""
import argparse
from collections import Counter, defaultdict
from dataclasses import fields, replace
import hashlib
import json
import math
from pathlib import Path
import re
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tqdm import tqdm
from config import RunConfig
from llm import generate, load_model
from runs import jsonl_writer, next_run_dir, utc_now, write_json
from candor.run_candor import load_questions
from candor.run_candor_variance import judge_response
from integrity.steering_check import paired_generate, read_jsonl


def summarize(rows):
    valid = {"candid", "sycophantic", "omission_other"}
    counts = Counter(r["label"] for r in rows)
    pairs = [r for r in rows if r["label"] in valid and r["baseline_label"] in valid]
    changes = defaultdict(list)
    for r in pairs:
        changes[r["split_group"]].append(int(r["label"] == "candid") - int(r["baseline_label"] == "candid"))
    n_scored = sum(counts[k] for k in valid)
    return dict(n=len(rows), n_scored=n_scored, outcomes=dict(counts),
        candor_rate=counts["candid"] / n_scored if n_scored else None,
        sycophancy_rate=counts["sycophantic"] / n_scored if n_scored else None,
        paired_n=len(pairs), paired_excluded=len(rows)-len(pairs),
        paired_candor_change=sum(int(r["label"] == "candid") - int(r["baseline_label"] == "candid") for r in pairs) / len(pairs) if pairs else None,
        family_mean_candor_change=sum(sum(v)/len(v) for v in changes.values()) / len(changes) if changes else None,
        sycophantic_to_candid=sum(r["baseline_label"] == "sycophantic" and r["label"] == "candid" for r in pairs),
        candid_to_non_candid=sum(r["baseline_label"] == "candid" and r["label"] != "candid" for r in pairs))


def select_items(report, items, trained):
    if not trained or trained != set(report["direction_items"]):
        raise ValueError("Direction metadata does not match analysis training items")
    splits, groups = report["splits"], report["split_groups"]
    if any(splits[q] != "train" for q in trained):
        raise ValueError("Direction includes non-training items")
    train_groups = {groups[q] for q, split in splits.items() if split == "train"}
    selected = [r for r in items if splits.get(r["id"]) == "validation"]
    if not selected or {r["id"] for r in selected} != {q for q, s in splits.items() if s == "validation"}:
        raise ValueError("Missing validation items")
    if any(groups[r["id"]] in train_groups or r.get("split_group", r["id"]) != groups[r["id"]] for r in selected):
        raise ValueError("Scenario family leakage or mismatched split groups")
    return sorted(selected, key=lambda r: r["id"])


def model_config(source, questions, **overrides):
    values = {f.name: source[f.name] for f in fields(RunConfig) if f.name in source}
    values.update(questions=questions, extract_activations=False, sample_batch_size=1)
    values.update(overrides)
    return RunConfig(**values)


def run(args):
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file
    from transformer_lens.model_bridge import TransformerBridge

    root = args.run_dir
    analysis = args.analysis_dir or root / "candor_analysis"
    report = json.loads((analysis / "report.json").read_text(encoding="utf-8"))
    source = json.loads((root / "config.json").read_text(encoding="utf-8"))
    if source != report["source_config"]:
        raise ValueError("Source config differs from direction analysis")
    path = analysis / "directions.safetensors"
    with safe_open(str(path), framework="pt", device="cpu") as f:
        metadata = f.metadata() or {}
    if metadata.get("direction") != "candid minus sycophantic":
        raise ValueError("Expected candid minus sycophantic directions")
    items = select_items(report, read_jsonl(root / "variance.jsonl"), set(json.loads(metadata.get("train_items", "[]"))))
    if args.max_items is not None:
        if args.max_items < 1:
            raise ValueError("max-items must be positive")
        items = items[:args.max_items]
    layers = None if args.layers == "all" else {int(x) for x in args.layers.split(",")}
    vectors = {}
    for hook, vector in load_file(str(path), device="cpu").items():
        match = re.fullmatch(r"blocks\.(\d+)\.hook_resid_post", hook)
        if not match or layers is not None and int(match[1]) not in layers:
            continue
        if vector.ndim != 1 or not torch.isfinite(vector).all() or vector.norm() == 0:
            raise ValueError(f"Invalid or zero direction: {hook}")
        vectors[hook] = vector.float()
    if not vectors or layers is not None and layers != {int(h.split('.')[1]) for h in vectors}:
        raise ValueError("Requested layers were not saved")
    alphas = sorted({0.0, *(float(x) for x in args.alphas.split(','))})
    if not all(math.isfinite(a) for a in alphas) or not min(alphas) < 0 < max(alphas):
        raise ValueError("Supply finite strengths of both signs")
    target = model_config(source, root / "questions.json", backend="transformers", device=args.device, runtime=args.runtime)
    if args.model:
        target = model_config(source, root / "questions.json", backend="transformers", device=args.device, runtime=args.runtime, model=args.model)
    with (root / "activations/index.jsonl").open(encoding="utf-8") as stream:
        first_capture = next((json.loads(line) for line in stream if line.strip()), {})
    resolved = first_capture.get("runtime", {}).get("resolved_revision")
    if resolved:
        target = replace(target, revision=resolved)
    if args.n_samples < 1:
        raise ValueError("n-samples must be positive")
    judge_source = source["judge"]
    judge = model_config(judge_source, root / "questions.json", device=args.judge_device, runtime=args.runtime)
    if args.judge_model:
        judge = model_config(judge_source, root / "questions.json", device=args.judge_device, runtime=args.runtime, model=args.judge_model)
    bank = {q["id"]: q for q in load_questions(root / "questions.json")}
    conditions = ["control", "treatment"] if args.condition == "both" else [args.condition]
    for item in items:
        if item["id"] not in bank or any(not item.get(c + "_user") for c in conditions):
            raise ValueError("Missing saved question or prompt")
    out = next_run_dir(args.out or root / "candor_steering")
    write_json(out / "config.json", dict(source_run=str(root.resolve()), analysis_dir=str(analysis.resolve()),
        direction_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), items=[r["id"] for r in items],
        alphas=alphas, hooks=list(vectors), site=args.site, conditions=conditions,
        n_samples=args.n_samples, seed=args.seed, scaling="raw alpha * vector",
        population="All validation items, without source outcome filtering; development sweep, not final test",
        seed_policy="seed + item_index * n_samples + sample; reused across conditions and interventions",
        generation=dict(target.metadata("candor-steering", len(items)), temperature=target.temperature),
        judge=judge.metadata("candor-judge", len(items))))
    write_json(out / "questions.json", {"questions": [bank[r["id"]] for r in items]})
    write_json(out / "status.json", dict(status="running", started=utc_now()))
    rows = []
    try:
        total = len(items)*len(conditions)*args.n_samples*(1+len(vectors)*len(alphas))
        with load_model(target) as llm, jsonl_writer(out / "results.jsonl") as write, tqdm(total=total, desc="Steering sweep", unit="response") as bar:
            bridge = TransformerBridge.boot_transformers(str(target.model), hf_model=llm.model,
                tokenizer=llm.tokenizer, device=llm.model.device, dtype=llm.model.dtype, revision=target.revision)
            bridge.eval()
            if any(h not in bridge.hook_dict or v.numel() != bridge.cfg.d_model for h, v in vectors.items()):
                raise ValueError("Direction hooks or widths do not match the target")
            write_json(out / "runtime.json", dict(device=str(llm.model.device), dtype=str(llm.model.dtype),
                resolved_revision=getattr(llm.model.config, "_commit_hash", None)))
            for i, item in enumerate(items):
                for condition in conditions:
                    for sample in range(args.n_samples):
                        seed = args.seed + i*args.n_samples + sample
                        base = dict(id=item["id"], split_group=item.get("split_group", item["id"]),
                            condition=condition, sample=sample, seed=seed)
                        kwargs = dict(seed=seed, temperature=target.temperature, max_tokens=target.max_tokens)
                        user = item[condition + "_user"]
                        baseline = generate(llm, source["system_prompt"], user, **kwargs)
                        row = dict(base, hook=None, alpha=0., response=baseline)
                        rows.append(row); write(row); bar.update(1)
                        for hook, vector in vectors.items():
                            for alpha in alphas:
                                bar.set_postfix(id=item["id"], condition=condition, hook=hook, alpha=alpha)
                                response, calls = paired_generate(llm, bridge, hook, vector, alpha, args.site,
                                    source["system_prompt"], user, **kwargs)
                                row = dict(base, hook=hook, alpha=alpha, response=response, hook_applications=calls)
                                rows.append(row); write(row); bar.update(1)
                                if alpha == 0 and response != baseline:
                                    raise RuntimeError("Zero-strength hook did not reproduce the unhooked baseline")
            del bridge
        # Release target before loading judge. Reuse judgments of identical text per item.
        cache, scored = {}, []
        with load_model(judge) as llm, jsonl_writer(out / "samples.jsonl") as write:
            for row in tqdm(rows, desc="Judge candor", unit="response"):
                cache_key = (row["id"], row["response"])
                if cache_key not in cache:
                    result = judge_response(llm, bank[row["id"]], row, judge, out)
                    cache[cache_key] = {k: v for k, v in result.items() if k not in row}
                result = dict(row, **cache[cache_key])
                scored.append(result); write(result)
        baselines = {(r["id"], r["condition"], r["sample"]): r["label"] for r in scored if r["hook"] is None}
        grouped = defaultdict(list)
        for row in scored:
            if row["hook"] is not None:
                row["baseline_label"] = baselines[row["id"], row["condition"], row["sample"]]
                grouped[row["condition"], row["hook"], row["alpha"]].append(row)
        summaries = [dict(condition=c, hook=h, alpha=a, **summarize(group)) for (c,h,a), group in sorted(grouped.items())]
        write_json(out / "summary.json", summaries)
        write_json(out / "status.json", dict(status="complete", finished=utc_now()))
    except BaseException as exc:
        write_json(out / "status.json", dict(status="failed", error=str(exc), finished=utc_now()))
        raise
    print(f"Saved {out}")
    return out


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir", type=Path)
    p.add_argument("--analysis-dir", type=Path)
    p.add_argument("--layers", default="all")
    p.add_argument("--alphas", default="-1,-0.5,0,0.5,1", help="Use --alphas=-1,0,1")
    p.add_argument("--site", choices=("decode", "prompt"), default="decode")
    p.add_argument("--condition", choices=("treatment", "control", "both"), default="both")
    p.add_argument("--n-samples", type=int, default=1)
    p.add_argument("--max-items", type=int)
    p.add_argument("--seed", type=int, default=10000)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--judge-device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--runtime", choices=("local", "runpod"), default="local")
    p.add_argument("--model", help="Relocated copy of the same target checkpoint")
    p.add_argument("--judge-model", help="Override saved judge model path")
    p.add_argument("--out", type=Path)
    return p


if __name__ == "__main__":
    run(build_parser().parse_args())
