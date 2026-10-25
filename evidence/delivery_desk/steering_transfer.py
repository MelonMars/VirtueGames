"""Extract matched Delivery Desk directions and test transfer on held-out dockets.

This is an exploratory diagnostic, not a confirmatory efficacy estimate. It
freezes a baseline reference history for each held-out case, then compares
baseline and decode-time-steered rulings at each evidence update.
"""
import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
import sys

import torch
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import RunConfig
from evidence.delivery_desk.protocol import (PROTOCOL, RULEBOOK,
    load_bank, opening_message, parse_decision, update_message)
from llm import load_model
from runs import jsonl_writer, utc_now, write_json


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def extract_direction(run_dirs, layer, min_class_count):
    hook = f"blocks.{layer}.hook_resid_post"
    batch_directions = []
    training_batches = []
    used_ids = set()
    observed_ids = set()
    for run_dir in run_dirs:
        samples = read_jsonl(run_dir / "samples.jsonl")
        observed_ids.update(row["id"] for row in samples)
        index = read_jsonl(run_dir / "activations" / "index.jsonl")
        calls = {}
        for item in index:
            context = item.get("context", {})
            if item.get("file") and not context.get("reference_only"):
                key = (context.get("decision_batch"), int(context.get("sample", -1)))
                calls[key] = item
        groups = defaultdict(list)
        for row in samples:
            batch = row.get("decision_batch")
            if batch and row.get("scored") and row.get("decision_correct") is not None:
                groups[batch].append(row)
        for batch, rows in sorted(groups.items()):
            good = [row for row in rows if row["decision_correct"]]
            bad = [row for row in rows if not row["decision_correct"]]
            if len(good) < min_class_count or len(bad) < min_class_count:
                continue

            def mean_activation(selected):
                vectors = []
                for row in selected:
                    record = calls.get((batch, int(row["sample"])))
                    if record is None:
                        raise ValueError(f"Missing activation for {batch}, sample {row['sample']}")
                    tensor = load_file(str(run_dir / "activations" / record["file"]))[hook]
                    vectors.append(tensor[0, 0].float())  # answer-tokens-1
                return torch.stack(vectors).mean(0)

            delta = mean_activation(good) - mean_activation(bad)
            norm = delta.norm()
            if not torch.isfinite(delta).all() or norm == 0:
                continue
            batch_directions.append(delta / norm)
            used_ids.add(rows[0]["id"])
            training_batches.append(dict(
                run=str(run_dir), decision_batch=batch, id=rows[0]["id"],
                stage=int(rows[0]["stage"]), target=rows[0]["target"],
                correct=len(good), incorrect=len(bad),
                correct_answers=dict(Counter(row["answer"] for row in good)),
                incorrect_answers=dict(Counter(row["answer"] for row in bad)),
            ))
    if not batch_directions:
        raise ValueError("No exact-prefix batches meet the minimum correct/incorrect counts")
    direction = torch.stack(batch_directions).mean(0)
    if direction.norm() == 0:
        raise ValueError("Matched batch directions canceled to zero")
    direction = direction / direction.norm()
    cosines = [float(torch.dot(direction, item)) for item in batch_directions]
    for batch, cosine in zip(training_batches, cosines):
        batch["cosine_to_aggregate"] = cosine
    return direction, training_batches, used_ids, observed_ids, hook


def summarize(rows, alphas):
    output = {}
    for alpha in alphas:
        group = [row for row in rows if row["alpha"] == alpha]
        output[str(alpha)] = {}
        for stage in (1, 2):
            stage_rows = [row for row in group if row["stage"] == stage]
            output[str(alpha)][str(stage)] = dict(
                n=len(stage_rows),
                accuracy=(sum(row["correct"] for row in stage_rows) / len(stage_rows)
                          if stage_rows else None),
                answer_counts=dict(Counter(row["answer"] for row in stage_rows)),
                unknown_rate=(sum(row["answer"] == "Unknown" for row in stage_rows) / len(stage_rows)
                              if stage_rows else None),
                mean_confidence=(sum(row["parsed"]["confidence"] for row in stage_rows
                                     if row["parsed"]["confidence"] is not None) /
                    sum(row["parsed"]["confidence"] is not None for row in stage_rows)
                    if any(row["parsed"]["confidence"] is not None for row in stage_rows) else None),
                mean_confidence_delta=(sum(row["parsed"]["confidence"] - row["baseline_parsed"]["confidence"]
                    for row in stage_rows if row["parsed"]["confidence"] is not None and
                    row["baseline_parsed"]["confidence"] is not None) /
                    sum(row["parsed"]["confidence"] is not None and
                        row["baseline_parsed"]["confidence"] is not None for row in stage_rows)
                    if any(row["parsed"]["confidence"] is not None and
                           row["baseline_parsed"]["confidence"] is not None for row in stage_rows) else None),
                answer_changes=sum(row["answer"] != row["baseline_answer"] for row in stage_rows),
                accuracy_changes=sum(row["correct"] != row["baseline_correct"] for row in stage_rows),
                text_changes=sum(row["steered_raw"] != row["baseline_raw"] for row in stage_rows),
                citation_changes=sum(set(row["parsed"]["evidence"]) !=
                    set(row["baseline_parsed"]["evidence"]) for row in stage_rows),
            )
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("training_runs", nargs="+", type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--case-bank", type=Path, default=Path(__file__).with_name("cases.json"))
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="cuda")
    parser.add_argument("--layer", type=int, default=16)
    parser.add_argument("--min-class-count", type=int, default=5)
    parser.add_argument("--alphas", default="-2,-1,0,1,2",
                        help="Comma-separated symmetric steering strengths; default is the empirically milder range")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Decode temperature; use repeated seeds at nonzero temperature to measure response distributions")
    parser.add_argument("--seed", type=int, default=790000,
                        help="Seed base; rerun with distinct bases for independent stochastic replications")
    args = parser.parse_args()

    run_dirs = [path.resolve() for path in args.training_runs]
    alphas = [float(value.strip()) for value in args.alphas.split(",") if value.strip()]
    if 0.0 not in alphas:
        alphas.append(0.0)
    direction, training_batches, training_ids, observed_ids, hook = extract_direction(
        run_dirs, args.layer, args.min_class_count)
    bank = load_bank(args.case_bank)
    heldout = [case for case in bank if case["id"] not in observed_ids]
    if not heldout:
        raise ValueError("No held-out cases remain after excluding extraction dockets")

    args.out.mkdir(parents=True, exist_ok=False)
    vector_path = args.out / "direction.safetensors"
    save_file({hook: direction.contiguous()}, str(vector_path), metadata={
        "method": "equal-weight mean of unit-normalized exact-prefix correct-minus-incorrect means",
        "source_runs": json.dumps([str(path) for path in run_dirs]),
        "training_batches": json.dumps(training_batches),
        "layer": str(args.layer), "slot": "0 (answer-tokens-1)"})
    config_sources = [json.loads((path / "config.json").read_text()) for path in run_dirs]
    first_config = config_sources[0]
    values = {key: first_config[key] for key in RunConfig.__dataclass_fields__ if key in first_config}
    cfg = RunConfig(**{**values, "questions": args.case_bank, "model": args.model,
        "device": args.device, "backend": "transformers", "extract_activations": False,
        "temperature": args.temperature, "max_tokens": 256, "thinking": "off"})

    rows = []
    try:
        with load_model(cfg) as llm:
            from transformer_lens.model_bridge import TransformerBridge
            bridge = TransformerBridge.boot_transformers(str(cfg.model), hf_model=llm.model,
                tokenizer=llm.tokenizer, device=llm.model.device, dtype=llm.model.dtype,
                revision=cfg.revision)
            bridge.eval()
            with jsonl_writer(args.out / "samples.jsonl") as write:
                for case_index, case in enumerate(heldout):
                    messages = [dict(role="system", content=RULEBOOK),
                                dict(role="user", content=opening_message(case))]
                    stages = [(0, None, case["initial_answer"])]
                    stages += [(index, update, update["answer"])
                               for index, update in enumerate(case["updates"], start=1)]
                    previous_target = None
                    for stage, update, target in stages:
                        if update is not None:
                            messages.append(dict(role="user", content=update_message(case, update, stage)))
                        seed = args.seed + case_index * 10 + stage
                        baseline_raw = llm.create_chat_completion(messages=messages,
                            temperature=args.temperature, seed=seed, max_tokens=256)["choices"][0]["message"].get("content") or ""
                        baseline = parse_decision(baseline_raw)
                        if stage > 0:
                            for alpha in alphas:
                                if alpha == 0.0:
                                    steered_raw = baseline_raw
                                else:
                                    def add_direction(value, hook=None):
                                        result = value.clone()
                                        result[:, -1, :] += alpha * direction.to(
                                            device=value.device, dtype=value.dtype)
                                        return result
                                    with bridge.hooks(fwd_hooks=[(hook, add_direction)]):
                                        steered_raw = llm.create_chat_completion(messages=messages,
                                            temperature=args.temperature, seed=seed, max_tokens=256)["choices"][0]["message"].get("content") or ""
                                parsed = parse_decision(steered_raw)
                                row = dict(id=case["id"], family=case["family"], stage=stage,
                                    update_type=update["type"], target=target,
                                    previous_target=previous_target, alpha=alpha,
                                    baseline_answer=baseline["answer"], answer=parsed["answer"],
                                    baseline_parsed=baseline, parsed=parsed,
                                    baseline_correct=baseline["answer"] == target,
                                    correct=parsed["answer"] == target,
                                    answer_changed=parsed["answer"] != baseline["answer"],
                                    baseline_raw=baseline_raw, steered_raw=steered_raw,
                                    seed=seed)
                                write(row)
                                rows.append(row)
                        messages.append(dict(role="assistant", content=baseline_raw))
                        previous_target = target
        write_json(args.out / "summary.json", dict(
            status="exploratory-heldout-transfer", protocol=PROTOCOL,
            training_runs=[str(path) for path in run_dirs], training_batches=training_batches,
            training_target_categories=sorted({item["target"] for item in training_batches}),
            warnings=([] if len({item["target"] for item in training_batches}) > 1 else
                ["All extracted matched batches share one target category; direction may encode answer-category preference rather than evidence-responsiveness."]),
            training_ids=sorted(training_ids), excluded_ids=sorted(observed_ids),
            heldout_ids=[case["id"] for case in heldout],
            direction_sha256=hashlib.sha256(vector_path.read_bytes()).hexdigest(),
            layer=args.layer, hook=hook, activation_position="answer-tokens-1",
            temperature=args.temperature, seed_base=args.seed,
            alpha_results=summarize(rows, alphas),
            interpretation="Held-out docket transfer only; reference histories use baseline answers. Report Unknown rate alongside accuracy because extraction correctness may be category-confounded.",
            created=utc_now()))
    except BaseException as exc:
        write_json(args.out / "status.json", dict(status="failed", error=str(exc), finished=utc_now()))
        raise
    write_json(args.out / "status.json", dict(status="complete", finished=utc_now()))
    print(args.out)


if __name__ == "__main__":
    main()
