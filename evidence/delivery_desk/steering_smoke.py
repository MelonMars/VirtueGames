"""Tiny, explicitly exploratory decode-time check for a matched Delivery Desk vector."""
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import sys

import torch
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import RunConfig
from evidence.delivery_desk.protocol import (PROTOCOL, RULEBOOK,
    load_bank, opening_message, parse_decision)
from llm import load_model
from runs import jsonl_writer, utc_now, write_json


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def matched_direction(run_dir, layer):
    """Average normalized correct-minus-incorrect means within identical-prefix batches."""
    samples = read_jsonl(run_dir / "samples.jsonl")
    index = read_jsonl(run_dir / "activations" / "index.jsonl")
    calls = {(item["context"]["id"], int(item["context"]["condition"].split("-")[-1]),
              int(item["context"]["sample"])): item for item in index if item.get("file")}
    groups = defaultdict(list)
    for row in samples:
        if row.get("scored") and row.get("decision_correct") is not None:
            groups[(row["id"], int(row["stage"]))].append(row)
    differences, training_batches = [], []
    hook = f"blocks.{layer}.hook_resid_post"
    for (case_id, stage), rows in sorted(groups.items()):
        good = [row for row in rows if row["decision_correct"]]
        bad = [row for row in rows if not row["decision_correct"]]
        if not good or not bad:
            continue
        def activations(selected):
            vectors = []
            for row in selected:
                record = calls[(case_id, stage, int(row["sample"]))]
                tensor = load_file(str(run_dir / "activations" / record["file"]))[hook]
                vectors.append(tensor[0, 0].float())  # answer-tokens-1
            return torch.stack(vectors)
        delta = activations(good).mean(0) - activations(bad).mean(0)
        if not torch.isfinite(delta).all() or delta.norm() == 0:
            continue
        differences.append(delta / delta.norm())
        training_batches.append(dict(id=case_id, stage=stage, correct=len(good), incorrect=len(bad)))
    if not differences:
        raise ValueError("No exact-prefix batch contains both correct and incorrect scored decisions")
    vector = torch.stack(differences).mean(0)
    if vector.norm() == 0:
        raise ValueError("Matched batch directions canceled to zero")
    return vector / vector.norm(), training_batches, hook


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--case-bank", type=Path,
                        help="Full Delivery Desk bank; defaults to the bank stored with the run")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="cuda")
    parser.add_argument("--layer", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=5.0,
                        help="Raw residual addition magnitude for a unit-normalized vector")
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    config_source = json.loads((run_dir / "config.json").read_text())
    if config_source.get("protocol") != PROTOCOL or config_source.get("sampling_design") != "decision-branches-v1":
        raise ValueError("Expected a Delivery Desk decision-branch run")
    vector, training_batches, hook = matched_direction(run_dir, args.layer)
    bank = load_bank(args.case_bank or (run_dir / "cases.json"))
    cases = {case["id"]: case for case in bank}
    # Concentrate the tiny diagnostic on prompts that actually produced both
    # correct and incorrect outcomes. This is deliberately in-sample: the same
    # batches also define the direction, so this is not a generalization test.
    transcripts = read_jsonl(run_dir / "transcripts.jsonl")
    sample_rows = read_jsonl(run_dir / "samples.jsonl")
    eval_prompts = []
    for batch in training_batches:
        for repeat in range(2):
            transcript = next(item for item in transcripts
                              if item["id"] == batch["id"] and item["stage"] == batch["stage"] and item["sample"] == 0)
            scored = next(item for item in sample_rows
                          if item["id"] == batch["id"] and item["stage"] == batch["stage"] and item["sample"] == 0)
            eval_prompts.append(dict(id=batch["id"], stage=batch["stage"], repeat=repeat,
                messages=transcript["conversation"][:-1], target=scored["target"]))

    out = run_dir / "steering_smoke_heterogeneous"
    out.mkdir(exist_ok=False)
    vector_path = out / "direction.safetensors"
    save_file({hook: vector.contiguous()}, str(vector_path), metadata={
        "method": "mean of unit-normalized within-exact-prefix correct-minus-incorrect activation means",
        "source_run": str(run_dir), "training_batches": json.dumps(training_batches),
        "layer": str(args.layer), "slot": "0 (answer-tokens-1)"})
    plan = dict(status="exploratory-only", source_run=str(run_dir), layer=args.layer,
        hook=hook, activation_position="answer-tokens-1", matched_training_batches=training_batches,
        direction_sha256=hashlib.sha256(vector_path.read_bytes()).hexdigest(),
        alpha=args.alpha, injection="decode-time residual addition at every generated token",
        eval_prompts=[dict(id=item["id"], stage=item["stage"], target=item["target"])
                      for item in eval_prompts], created=utc_now())
    write_json(out / "config.json", plan)
    values = {key: config_source[key] for key in RunConfig.__dataclass_fields__ if key in config_source}
    values["questions"] = run_dir / "cases.json"
    cfg = RunConfig(**values)
    cfg = RunConfig(**{**cfg.__dict__, "questions": run_dir / "cases.json", "model": args.model,
        "device": args.device, "backend": "transformers", "extract_activations": False,
        "temperature": 0.0, "max_tokens": 256, "thinking": "off"})
    from transformer_lens.model_bridge import TransformerBridge
    rows = []
    try:
        with load_model(cfg) as llm:
            bridge = TransformerBridge.boot_transformers(str(cfg.model), hf_model=llm.model,
                tokenizer=llm.tokenizer, device=llm.model.device, dtype=llm.model.dtype,
                revision=cfg.revision)
            bridge.eval()
            with jsonl_writer(out / "samples.jsonl") as write:
                for number, item in enumerate(eval_prompts):
                    seed = 81000 + number
                    baseline = llm.create_chat_completion(messages=item["messages"], temperature=0,
                        seed=seed, max_tokens=256)["choices"][0]["message"].get("content") or ""
                    def add_direction(value, hook):
                        result = value.clone()
                        result[:, -1, :] += args.alpha * vector.to(device=value.device, dtype=value.dtype)
                        return result
                    with bridge.hooks(fwd_hooks=[(hook, add_direction)]):
                        steered = llm.create_chat_completion(messages=item["messages"], temperature=0,
                            seed=seed, max_tokens=256)["choices"][0]["message"].get("content") or ""
                    base_parsed, steer_parsed = parse_decision(baseline), parse_decision(steered)
                    row = dict(id=item["id"], stage=item["stage"], repeat=item["repeat"],
                        target=item["target"], seed=seed,
                        alpha=args.alpha, baseline=baseline, steered=steered,
                        baseline_answer=base_parsed["answer"], steered_answer=steer_parsed["answer"],
                        baseline_correct=base_parsed["answer"] == item["target"],
                        steered_correct=steer_parsed["answer"] == item["target"],
                        response_changed=baseline != steered,
                        answer_changed=base_parsed["answer"] != steer_parsed["answer"])
                    write(row); rows.append(row)
        write_json(out / "summary.json", dict(n_pairs=len(rows), generations=2 * len(rows), alpha=args.alpha,
            response_changes=sum(row["response_changed"] for row in rows),
            answer_changes=sum(row["answer_changed"] for row in rows),
            baseline_accuracy=sum(row["baseline_correct"] for row in rows) / len(rows),
            steered_accuracy=sum(row["steered_correct"] for row in rows) / len(rows),
            unique_prefixes=len({(row["id"], row["stage"]) for row in rows}),
            interpretation="Eight deterministic generations repeated across heterogeneous source prefixes; direction and evaluation reuse the same batches, so this is an in-sample diagnostic, not an efficacy estimate."))
    except BaseException as exc:
        write_json(out / "status.json", dict(status="failed", error=str(exc), finished=utc_now()))
        raise
    write_json(out / "status.json", dict(status="complete", finished=utc_now()))
    print(out)


if __name__ == "__main__":
    main()
