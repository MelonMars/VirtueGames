"""Causally sweep validation-selected calibration directions on test families."""
import argparse
from collections import defaultdict
from dataclasses import fields
import json
import math
from pathlib import Path
import re
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tqdm import tqdm
from config import RunConfig
from llm import generate, load_model
from original.bet_on_yourself.multiturn.protocol import parse_decision, score_decision
from runs import next_run_dir, utc_now, write_json


def read_jsonl(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


class ResidualAddition:
    """Add alpha times a unit vector at the current decode position."""
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
        edited = value.clone()
        edited[:, -1, :] += self.alpha * self.vector.to(device=value.device, dtype=value.dtype)
        return edited


def score_response(text, case):
    parsed = parse_decision(text)
    scoring = (score_decision(parsed["answer"], case) if parsed["parse_ok"] else
               dict(answer_matches_key=None, is_abstention=False,
                    abstention_appropriate=None, decision_correct=None))
    answer_brier = (
        (parsed["confidence"] / 100 - float(scoring["answer_matches_key"])) ** 2
        if parsed["confidence_parse_ok"] and not scoring["is_abstention"]
        and scoring["answer_matches_key"] is not None else None)
    decision_brier = (
        (parsed["decision_confidence"] / 100 - float(scoring["decision_correct"])) ** 2
        if parsed["decision_confidence_parse_ok"] and scoring["decision_correct"] is not None else None)
    return dict(**parsed, **scoring, confidence_brier=answer_brier,
                decision_brier=decision_brier)


def summarize(rows):
    parsed = [r for r in rows if r["parse_ok"]]
    answer = [r for r in rows if r["confidence_brier"] is not None]
    decisions = [r for r in rows if r["decision_brier"] is not None]
    abstentions = [r for r in parsed if r["is_abstention"]]
    appropriate = [r for r in abstentions if r["abstention_appropriate"] is not None]
    answer_correct = [r for r in parsed if not r["is_abstention"] and r["answer_matches_key"] is not None]
    return dict(n=len(rows), parse_coverage=len(parsed),
        decision_accuracy=(sum(bool(r["decision_correct"]) for r in parsed) / len(parsed) if parsed else None),
        factual_answer_accuracy=(sum(bool(r["answer_matches_key"]) for r in answer_correct) / len(answer_correct)
                                 if answer_correct else None),
        answer_brier_coverage=len(answer),
        mean_answer_brier=(sum(r["confidence_brier"] for r in answer) / len(answer) if answer else None),
        decision_brier_coverage=len(decisions),
        mean_decision_brier=(sum(r["decision_brier"] for r in decisions) / len(decisions) if decisions else None),
        mean_answer_confidence=(sum(r["confidence"] for r in answer) / len(answer) if answer else None),
        mean_decision_confidence=(sum(r["decision_confidence"] for r in decisions) / len(decisions) if decisions else None),
        abstention_count=len(abstentions), abstention_rate=(len(abstentions) / len(parsed) if parsed else None),
        appropriate_abstention_rate=(sum(r["abstention_appropriate"] is True for r in appropriate) / len(appropriate)
                                      if appropriate else None))


def main(argv=None):
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file
    from transformer_lens.model_bridge import TransformerBridge

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Captured single-turn Bet On Yourself run")
    parser.add_argument("--analysis-dir", type=Path, help="Output from analyze_calibration.py")
    parser.add_argument("--alphas", default="-4,-2,-1,0,1,2,4",
                        help="Signed, raw unit-vector additions; include both signs")
    parser.add_argument("--site", choices=("decode", "prompt"), default="decode")
    parser.add_argument("--n-samples", type=int, help="Max source samples per test item")
    parser.add_argument("--max-items", type=int, help="Limit test items for a pilot sweep")
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--runtime", choices=("local", "runpod"), default="local")
    parser.add_argument("--model", help="Local path to original checkpoint if its saved path moved")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    alphas = sorted(set(float(x) for x in args.alphas.split(",")) | {0.0})
    if not all(math.isfinite(x) for x in alphas) or not min(alphas) < 0 < max(alphas):
        raise ValueError("Supply finite alpha values of both signs")
    if args.n_samples is not None and args.n_samples < 1:
        raise ValueError("--n-samples must be positive")
    if args.max_items is not None and args.max_items < 1:
        raise ValueError("--max-items must be positive")

    run_dir = args.run_dir.resolve()
    analysis_dir = (args.analysis_dir or run_dir / "calibration_analysis").resolve()
    analysis = json.loads((analysis_dir / "report.json").read_text(encoding="utf-8"))
    source = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    if analysis.get("source_run") != str(run_dir):
        raise ValueError("Analysis provenance does not match the source run")
    if source != analysis.get("source_config"):
        raise ValueError("Source configuration changed after candidate extraction")
    test_families = set(analysis["test_families_reserved"])
    split_map = analysis["family_splits"]
    if any(split_map.get(family) != "test" for family in test_families):
        raise ValueError("Invalid family-level test split")
    direction_path = analysis_dir / "directions.safetensors"
    with safe_open(str(direction_path), framework="pt", device="cpu") as direction_file:
        direction_metadata = direction_file.metadata() or {}
    saved_test_families = set(json.loads(direction_metadata.get("test_families", "[]")))
    saved_train_families = set(json.loads(direction_metadata.get("training_families", "[]")))
    saved_validation_families = set(json.loads(direction_metadata.get("validation_families", "[]")))
    if (saved_test_families != test_families
            or saved_train_families != {f for f, split in split_map.items() if split == "train"}
            or saved_validation_families != {f for f, split in split_map.items() if split == "validation"}
            or (saved_test_families & saved_train_families)
            or (saved_test_families & saved_validation_families)
            or (saved_train_families & saved_validation_families)):
        raise ValueError("Direction tensors do not match the reserved family split")
    directions = load_file(str(direction_path), device="cpu")
    selected = analysis.get("selected_candidates", {})
    if set(directions) != {key for rows in selected.values() for key in rows}:
        raise ValueError("Selected candidate list does not match saved directions")
    candidate_evidence = {item["direction_key"]: item
                          for rows in analysis.get("candidates", {}).values() for item in rows
                          if item.get("direction_key")}
    candidate_specs = []
    for target in ("answer", "decision"):
        for key in selected.get(target, []):
            if key not in directions:
                raise ValueError(f"Missing selected direction tensor: {key}")
            match = re.fullmatch(r"(answer|decision)\.layer_(\d+)\.hook_resid_post", key)
            if not match or match.group(1) != target:
                raise ValueError(f"Invalid candidate direction key: {key}")
            vector = directions[key].float().flatten()
            if not torch.isfinite(vector).all() or abs(float(vector.norm()) - 1.0) > 1e-3:
                raise ValueError(f"Expected a finite unit direction: {key}")
            candidate_specs.append(dict(key=key, target=target, layer=int(match.group(2)),
                                        hook=f"blocks.{match.group(2)}.hook_resid_post", vector=vector,
                                        validation_correlation=candidate_evidence.get(key, {}).get("validation_correlation"),
                                        validation_r2=candidate_evidence.get(key, {}).get("validation_r2")))
    if not candidate_specs:
        raise ValueError("No validation-selected candidate directions found")

    samples = [row for row in read_jsonl(run_dir / "samples.jsonl")
               if split_map.get(row["family"]) == "test"]
    if not samples:
        raise ValueError("No source samples in the reserved test families")
    n_samples = args.n_samples or int(source["n_samples"])
    samples = [row for row in samples if int(row["sample"]) < n_samples]
    item_ids = sorted({row["id"] for row in samples})
    if args.max_items is not None:
        item_ids = item_ids[:args.max_items]
        samples = [row for row in samples if row["id"] in set(item_ids)]
    if not samples:
        raise ValueError("No samples remain after limits")

    transcripts = read_jsonl(run_dir / "transcripts.jsonl")
    prompt_by_key = {}
    for transcript in transcripts:
        key = (transcript.get("id"), transcript.get("stage"), transcript.get("sample"))
        conversation = transcript.get("conversation", [])
        if len(conversation) >= 2 and conversation[0].get("role") == "system" and conversation[1].get("role") == "user":
            prompt_by_key[key] = conversation[:2]
    cases_data = json.loads((run_dir / "cases.json").read_text(encoding="utf-8"))
    cases = {case["id"]: case for case in cases_data["cases"]}
    for row in samples:
        key = (row["id"], row["stage"], row["sample"])
        if key not in prompt_by_key or row["id"] not in cases:
            raise ValueError(f"Missing prompt or answer key for test sample {key}")

    run_meta = read_jsonl(run_dir / "activations" / "index.jsonl")[0]
    values = {field.name: source[field.name] for field in fields(RunConfig) if field.name in source}
    values["questions"] = Path(source.get("questions_file", run_dir / "cases.json"))
    values["difficulty"] = tuple(source.get("difficulty_filter") or ())
    values.update(backend="transformers", extract_activations=False, runtime=args.runtime,
                  device=args.device, sample_batch_size=1)
    if args.model:
        values["model"] = args.model
    resolved_revision = run_meta.get("runtime", {}).get("resolved_revision")
    if resolved_revision and not args.model:
        values["revision"] = resolved_revision
    config = RunConfig(**values)

    out = next_run_dir(args.out or run_dir / "calibration_steering")
    plan = dict(source_run=str(run_dir), analysis_dir=str(analysis_dir),
        analysis_seed=analysis["seed"], test_families=sorted(test_families),
        held_out_items=item_ids, candidate_directions=[{k: v for k, v in spec.items() if k != "vector"}
            for spec in candidate_specs], alphas=alphas, site=args.site, n_samples=n_samples,
        steering_rule="unit resid_post vector added at the final token position during prompt/decode; each candidate swept separately",
        selection_rule="directions and layers were selected on validation families only",
        generation=config.metadata("bet-on-yourself-calibration-steering", len(item_ids)),
        seed_policy="fixed per source item/sample across baseline and every direction/alpha")
    write_json(out / "config.json", plan)
    write_json(out / "status.json", dict(status="running", started=utc_now()))

    # Use a fixed, paired seed for a fresh unsteered baseline and every intervention.
    source_order = sorted(samples, key=lambda row: (row["id"], int(row["sample"])))
    test_index = {item_id: index for index, item_id in enumerate(item_ids)}
    baseline_by_key = {}
    grouped = defaultdict(list)
    total = len(source_order) * (1 + len(candidate_specs) * len(alphas))
    try:
        with load_model(config) as llm:
            if resolved_revision and getattr(llm.model.config, "_commit_hash", None) not in (None, resolved_revision):
                raise ValueError("Loaded model revision differs from the activation source checkpoint")
            bridge = TransformerBridge.boot_transformers(str(config.model), hf_model=llm.model,
                tokenizer=llm.tokenizer, device=llm.model.device, dtype=llm.model.dtype,
                revision=config.revision)
            bridge.eval()
            for spec in candidate_specs:
                if spec["hook"] not in bridge.hook_dict:
                    raise ValueError(f"Model has no requested steering hook: {spec['hook']}")
            with (out / "samples.jsonl").open("w", encoding="utf-8") as stream, \
                    tqdm(total=total, desc="Calibration steering", unit="generation") as progress:
                for row in source_order:
                    key = (row["id"], row["stage"], row["sample"])
                    prompt = prompt_by_key[key]
                    user = prompt[1]["content"]
                    system = prompt[0]["content"]
                    seed = args.seed + test_index[row["id"]] * n_samples + int(row["sample"])
                    gen_kwargs = dict(seed=seed, temperature=config.temperature, max_tokens=config.max_tokens)
                    baseline_text = generate(llm, system, user, **gen_kwargs)
                    baseline_score = score_response(baseline_text, cases[row["id"]])
                    baseline_by_key[key] = baseline_score
                    progress.update(1)
                    baseline_row = dict(id=row["id"], family=row["family"], stage=row["stage"],
                        sample=row["sample"], seed=seed, target="baseline", direction_key=None,
                        layer=None, alpha=0.0, raw=baseline_text, **baseline_score)
                    stream.write(json.dumps(baseline_row) + "\n")
                    for spec in candidate_specs:
                        for alpha in alphas:
                            addition = ResidualAddition(spec["vector"], alpha, args.site)
                            with bridge.hooks(fwd_hooks=[(spec["hook"], addition)]):
                                text = generate(llm, system, user, **gen_kwargs)
                            if not addition.applications:
                                raise RuntimeError(f"Steering hook did not fire for {spec['key']}")
                            if alpha == 0 and text != baseline_text:
                                raise RuntimeError("Zero-strength intervention changed the generated response")
                            score = score_response(text, cases[row["id"]])
                            result = dict(id=row["id"], family=row["family"], stage=row["stage"],
                                sample=row["sample"], seed=seed, target=spec["target"],
                                direction_key=spec["key"], layer=spec["layer"], alpha=alpha,
                                hook_applications=addition.applications, raw=text, **score)
                            result["paired_answer_brier_change"] = (
                                score["confidence_brier"] - baseline_score["confidence_brier"]
                                if score["confidence_brier"] is not None and baseline_score["confidence_brier"] is not None else None)
                            result["paired_decision_brier_change"] = (
                                score["decision_brier"] - baseline_score["decision_brier"]
                                if score["decision_brier"] is not None and baseline_score["decision_brier"] is not None else None)
                            result["paired_decision_accuracy_change"] = (
                                int(bool(score["decision_correct"])) - int(bool(baseline_score["decision_correct"]))
                                if score["decision_correct"] is not None and baseline_score["decision_correct"] is not None else None)
                            grouped[spec["key"], alpha].append(result)
                            stream.write(json.dumps(result) + "\n")
                            stream.flush()
                            progress.update(1)
                            progress.set_postfix(item=row["id"], layer=spec["layer"], a=alpha, refresh=False)

        summary = []
        for (direction_key, alpha), rows in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
            target = rows[0]["target"]
            by_family = {}
            for family in sorted({row["family"] for row in rows}):
                family_rows = [row for row in rows if row["family"] == family]
                deltas = [row["paired_decision_brier_change"] for row in family_rows
                          if row["paired_decision_brier_change"] is not None]
                by_family[family] = dict(**summarize(family_rows),
                    paired_mean_decision_brier_change=sum(deltas) / len(deltas) if deltas else None)
            family_deltas = [item["paired_mean_decision_brier_change"] for item in by_family.values()
                             if item["paired_mean_decision_brier_change"] is not None]
            summary.append(dict(direction_key=direction_key, target=target,
                layer=rows[0]["layer"], alpha=alpha, test_families=sorted(test_families),
                **summarize(rows), by_family=by_family,
                equal_family_mean_decision_brier_change=(sum(family_deltas) / len(family_deltas)
                                                         if family_deltas else None),
                paired_mean_answer_brier_change=(
                    sum(r["paired_answer_brier_change"] for r in rows if r["paired_answer_brier_change"] is not None)
                    / sum(r["paired_answer_brier_change"] is not None for r in rows)
                    if any(r["paired_answer_brier_change"] is not None for r in rows) else None),
                paired_mean_decision_brier_change=(
                    sum(r["paired_decision_brier_change"] for r in rows if r["paired_decision_brier_change"] is not None)
                    / sum(r["paired_decision_brier_change"] is not None for r in rows)
                    if any(r["paired_decision_brier_change"] is not None for r in rows) else None),
                paired_mean_decision_accuracy_change=(
                    sum(r["paired_decision_accuracy_change"] for r in rows if r["paired_decision_accuracy_change"] is not None)
                    / sum(r["paired_decision_accuracy_change"] is not None for r in rows)
                    if any(r["paired_decision_accuracy_change"] is not None for r in rows) else None)))
        write_json(out / "summary.json", summary)
        write_json(out / "status.json", dict(status="complete", finished=utc_now()))
    except BaseException as exc:
        write_json(out / "status.json", dict(status="failed", error=str(exc), finished=utc_now()))
        raise
    print(f"Saved {out}")
    return out


if __name__ == "__main__":
    main()
