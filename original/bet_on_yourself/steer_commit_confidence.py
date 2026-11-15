"""Tune each confidence intervention on validation and test it once on held-out families."""
import argparse
from collections import defaultdict
from dataclasses import fields
import hashlib
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
from original.bet_on_yourself.singleturn.commit_confidence import (
    PROTOCOL, ANSWER_CONFIDENCE_SYSTEM, DECISION_CONFIDENCE_SYSTEM,
    WAGER_SYSTEM, assemble_response,
)
from original.bet_on_yourself.steer_calibration import ResidualAddition, read_jsonl, score_response
from runs import next_run_dir, utc_now, write_json


def family_mean(rows, field):
    by_family = defaultdict(list)
    for row in rows:
        if row.get(field) is not None:
            by_family[row["family"]].append(row[field])
    means = [sum(values) / len(values) for values in by_family.values()]
    return sum(means) / len(means) if means else None


def matched_family_mean(rows, baseline_rows, field):
    baseline_keys = {(row["id"], row["sample"]) for row in baseline_rows if row.get(field) is not None}
    candidate_keys = {(row["id"], row["sample"]) for row in rows if row.get(field) is not None}
    if not baseline_keys or candidate_keys != baseline_keys:
        return None
    return family_mean(rows, field)


def main(argv=None):
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file
    from transformer_lens.model_bridge import TransformerBridge

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Separated-confidence BOY run")
    parser.add_argument("--analysis-dir", type=Path)
    parser.add_argument("--alphas", default="-4,-2,-1,1,2,4")
    parser.add_argument("--n-samples", type=int)
    parser.add_argument("--max-items", type=int)
    parser.add_argument("--seed", type=int, default=2701)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--runtime", choices=("local", "runpod"), default="local")
    parser.add_argument("--model", help="Local path if the saved checkpoint path moved")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    alphas = sorted(set(float(value) for value in args.alphas.split(",")) - {0.0})
    if not alphas or not all(math.isfinite(value) for value in alphas) or not min(alphas) < 0 < max(alphas):
        raise ValueError("Supply finite alpha values of both signs")
    if args.n_samples is not None and args.n_samples < 1:
        raise ValueError("--n-samples must be positive")
    if args.max_items is not None and args.max_items < 1:
        raise ValueError("--max-items must be positive")

    run_dir = args.run_dir.resolve()
    analysis_dir = (args.analysis_dir or run_dir / "calibration_analysis").resolve()
    source = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    analysis = json.loads((analysis_dir / "report.json").read_text(encoding="utf-8"))
    if source.get("protocol") != PROTOCOL:
        raise ValueError(f"Expected {PROTOCOL}; rerun with the separated-confidence runner")
    if analysis.get("source_run") != str(run_dir) or analysis.get("source_config") != source:
        raise ValueError("Analysis provenance does not match the source run")
    split_map = analysis["family_splits"]
    if any(not any(value == split for value in split_map.values()) for split in ("validation", "test")):
        raise ValueError("Analysis must reserve validation and test families")

    direction_path = analysis_dir / "directions.safetensors"
    with safe_open(str(direction_path), framework="pt", device="cpu") as tensor_file:
        tensor_metadata = tensor_file.metadata() or {}
    for split, metadata_key in (("test", "test_families"), ("train", "training_families"),
                                ("validation", "validation_families")):
        expected = {family for family, value in split_map.items() if value == split}
        if set(json.loads(tensor_metadata.get(metadata_key, "[]"))) != expected:
            raise ValueError("Direction tensors do not match the report's family split")
    directions = load_file(str(direction_path), device="cpu")
    selected_keys = {key for values in analysis.get("selected_candidates", {}).values() for key in values}
    if set(directions) != selected_keys:
        raise ValueError("Selected candidate list does not match direction tensors")
    evidence = {row["direction_key"]: row for values in analysis.get("candidates", {}).values()
                for row in values if row.get("direction_key")}
    candidates = []
    for target in ("answer", "decision"):
        for key in analysis.get("selected_candidates", {}).get(target, []):
            match = re.fullmatch(r"(answer|decision)\.layer_(\d+)\.hook_resid_post", key)
            if not match or match.group(1) != target:
                raise ValueError(f"Invalid direction key: {key}")
            vector = directions[key].float().flatten()
            if not torch.isfinite(vector).all() or abs(float(vector.norm()) - 1.0) > 1e-3:
                raise ValueError(f"Expected a finite unit direction: {key}")
            candidates.append(dict(key=key, target=target, layer=int(match.group(2)),
                hook=f"blocks.{match.group(2)}.hook_resid_post", vector=vector,
                validation_correlation=evidence.get(key, {}).get("validation_correlation"),
                validation_r2=evidence.get(key, {}).get("validation_r2")))
    if not candidates:
        raise ValueError("No selected candidate directions")

    cases_data = json.loads((run_dir / "cases.json").read_text(encoding="utf-8"))
    cases = {case["id"]: case for case in cases_data["cases"]}
    samples = read_jsonl(run_dir / "samples.jsonl")
    n_samples = args.n_samples or int(source["n_samples"])
    by_split = {}
    for split in ("validation", "test"):
        rows = [row for row in samples if split_map.get(row["family"]) == split
                and int(row["sample"]) < n_samples]
        item_ids = sorted({row["id"] for row in rows})
        if args.max_items is not None:
            item_ids = item_ids[:args.max_items]
            rows = [row for row in rows if row["id"] in set(item_ids)]
        required = ("answer_response", "answer_confidence_prompt", "decision_confidence_prompt", "wager_prompt")
        if not rows or any(not all(row.get(field) for field in required) for row in rows):
            raise ValueError(f"Missing samples or phase prompts in {split} split")
        by_split[split] = rows

    run_meta = read_jsonl(run_dir / "activations" / "index.jsonl")[0]
    values = {field.name: source[field.name] for field in fields(RunConfig) if field.name in source}
    values["questions"] = Path(source.get("questions_file", run_dir / "cases.json"))
    values["difficulty"] = tuple(source.get("difficulty_filter") or ())
    values.update(backend="transformers", extract_activations=False, runtime=args.runtime,
                  device=args.device, sample_batch_size=1)
    if args.model:
        values["model"] = args.model
    revision = run_meta.get("runtime", {}).get("resolved_revision")
    if revision and not args.model:
        values["revision"] = revision
    config = RunConfig(**values)
    out = next_run_dir(args.out or run_dir / "calibration_steering")
    write_json(out / "config.json", dict(source_run=str(run_dir), analysis_dir=str(analysis_dir),
        analysis_seed=analysis["seed"], layer_range=analysis.get("layer_range"),
        validation_families=sorted(f for f, split in split_map.items() if split == "validation"),
        test_families=sorted(f for f, split in split_map.items() if split == "test"),
        alphas=alphas, n_samples=n_samples, max_items=args.max_items, answer_is_fixed=True,
        steering_scope="numeric output of one confidence field only",
        selection_rule="select target-specific layer/direction/alpha on validation; evaluate once on test",
        seed_policy="paired per item and phase; answer and other forecast remain fixed"))
    write_json(out / "status.json", dict(status="running", started=utc_now()))

    def score_fields(row, answer_conf, decision_conf, wager):
        response, _ = assemble_response(row["answer_response"], answer_conf, decision_conf, wager)
        return dict(id=row["id"], family=row["family"], category=row["category"],
            sample=row["sample"], committed_answer=row["answer"], raw=response,
            answer_confidence_raw=answer_conf, decision_confidence_raw=decision_conf,
            wager_raw=wager, **score_response(response, cases[row["id"]]))

    def seed_for(row, split, offset):
        item_seed = int(hashlib.sha256(row["id"].encode()).hexdigest()[:8], 16)
        return args.seed + (0 if split == "validation" else 1000000) + item_seed * n_samples + int(row["sample"]) + offset

    validation_records, test_records = [], []
    selected, sweep_diagnostics = {}, {}
    try:
        with load_model(config) as llm:
            if revision and getattr(llm.model.config, "_commit_hash", None) not in (None, revision):
                raise ValueError("Loaded model revision differs from the activation source checkpoint")
            bridge = TransformerBridge.boot_transformers(str(config.model), hf_model=llm.model,
                tokenizer=llm.tokenizer, device=llm.model.device, dtype=llm.model.dtype,
                revision=config.revision)
            bridge.eval()
            for candidate in candidates:
                if candidate["hook"] not in bridge.hook_dict:
                    raise ValueError(f"Model has no requested steering hook: {candidate['hook']}")

            val_rows, test_rows = by_split["validation"], by_split["test"]
            work = len(val_rows) * (3 + len(candidates) * len(alphas)) + \
                   len(test_rows) * (3 + len(candidates))
            with tqdm(total=work, desc="Isolated confidence steering", unit="generation") as bar:
                for row in val_rows:
                    common = dict(temperature=config.temperature, max_tokens=8)
                    answer_base = generate(llm, ANSWER_CONFIDENCE_SYSTEM,
                        row["answer_confidence_prompt"], seed=seed_for(row, "validation", 1), **common)
                    decision_base = generate(llm, DECISION_CONFIDENCE_SYSTEM,
                        row["decision_confidence_prompt"], seed=seed_for(row, "validation", 2), **common)
                    wager_base = generate(llm, WAGER_SYSTEM, row["wager_prompt"],
                        seed=seed_for(row, "validation", 3), temperature=config.temperature, max_tokens=4)
                    baseline = score_fields(row, answer_base, decision_base, wager_base)
                    baseline.update(split="validation", target="baseline", direction_key=None,
                                    layer=None, alpha=0.0)
                    validation_records.append(baseline)
                    bar.update(3)
                    for candidate in candidates:
                        prompt_field = f"{candidate['target']}_confidence_prompt"
                        system = (ANSWER_CONFIDENCE_SYSTEM if candidate["target"] == "answer"
                                  else DECISION_CONFIDENCE_SYSTEM)
                        phase_offset = 1 if candidate["target"] == "answer" else 2
                        for alpha in alphas:
                            addition = ResidualAddition(candidate["vector"], alpha,
                                                        site="prompt-and-decode")
                            with bridge.hooks(fwd_hooks=[(candidate["hook"], addition)]):
                                phase = generate(llm, system, row[prompt_field],
                                    seed=seed_for(row, "validation", phase_offset), **common)
                            if not addition.applications:
                                raise RuntimeError(f"Steering hook did not fire for {candidate['key']}")
                            answer_value = phase if candidate["target"] == "answer" else answer_base
                            decision_value = phase if candidate["target"] == "decision" else decision_base
                            result = score_fields(row, answer_value, decision_value, wager_base)
                            result.update(split="validation", target=candidate["target"],
                                direction_key=candidate["key"], layer=candidate["layer"], alpha=alpha)
                            validation_records.append(result)
                            bar.update(1)

                # Alpha/direction selection uses only validation and requires
                # complete paired metric coverage against its baseline.
                for target, metric in (("answer", "confidence_brier"), ("decision", "decision_brier")):
                    baseline_rows = [row for row in validation_records if row["target"] == "baseline"]
                    baseline_score = family_mean(baseline_rows, metric)
                    target_candidates = [candidate for candidate in candidates if candidate["target"] == target]
                    options = []
                    for candidate in target_candidates:
                        for alpha in alphas:
                            rows = [row for row in validation_records
                                    if row.get("direction_key") == candidate["key"] and row["alpha"] == alpha]
                            score = matched_family_mean(rows, baseline_rows, metric)
                            if score is not None:
                                options.append((score, candidate, alpha, len(rows)))
                    options.sort(key=lambda item: item[0])
                    sweep_diagnostics[target] = [dict(direction_key=item[1]["key"], layer=item[1]["layer"],
                        alpha=item[2], validation_equal_family_brier=item[0],
                        baseline_validation_equal_family_brier=baseline_score,
                        improvement=(baseline_score - item[0] if baseline_score is not None else None),
                        metric_coverage=item[3]) for item in options]
                    best = options[0] if options else None
                    if best and baseline_score is not None and best[0] < baseline_score:
                        selected[target] = dict(direction_key=best[1]["key"], alpha=best[2],
                            validation_equal_family_brier=best[0],
                            baseline_validation_equal_family_brier=baseline_score,
                            selection_reason="improved validation Brier")
                    else:
                        selected[target] = dict(direction_key=None, alpha=0.0,
                            validation_equal_family_brier=baseline_score,
                            baseline_validation_equal_family_brier=baseline_score,
                            best_candidate_direction=best[1]["key"] if best else None,
                            best_candidate_alpha=best[2] if best else None,
                            best_candidate_validation_brier=best[0] if best else None,
                            selection_reason=("no candidate improved validation Brier" if best else
                                "analysis selected no direction for this target" if not target_candidates else
                                "no candidate had complete paired coverage"))

                # The test set is touched only after validation has frozen one
                # direction/strength (or a no-op) for each forecast target.
                for row in test_rows:
                    common = dict(temperature=config.temperature, max_tokens=8)
                    answer_base = generate(llm, ANSWER_CONFIDENCE_SYSTEM,
                        row["answer_confidence_prompt"], seed=seed_for(row, "test", 1), **common)
                    decision_base = generate(llm, DECISION_CONFIDENCE_SYSTEM,
                        row["decision_confidence_prompt"], seed=seed_for(row, "test", 2), **common)
                    wager_base = generate(llm, WAGER_SYSTEM, row["wager_prompt"],
                        seed=seed_for(row, "test", 3), temperature=config.temperature, max_tokens=4)
                    baseline = score_fields(row, answer_base, decision_base, wager_base)
                    baseline.update(split="test", target="baseline", direction_key=None,
                                    layer=None, alpha=0.0)
                    test_records.append(baseline)
                    bar.update(3)
                    for target in ("answer", "decision"):
                        choice = selected[target]
                        candidate = next((item for item in candidates
                                          if item["key"] == choice["direction_key"]), None)
                        if candidate is None:
                            continue
                        system = ANSWER_CONFIDENCE_SYSTEM if target == "answer" else DECISION_CONFIDENCE_SYSTEM
                        offset = 1 if target == "answer" else 2
                        addition = ResidualAddition(candidate["vector"], choice["alpha"],
                                                    site="prompt-and-decode")
                        with bridge.hooks(fwd_hooks=[(candidate["hook"], addition)]):
                            phase = generate(llm, system, row[f"{target}_confidence_prompt"],
                                seed=seed_for(row, "test", offset), **common)
                        if not addition.applications:
                            raise RuntimeError(f"Steering hook did not fire for {candidate['key']}")
                        answer_value = phase if target == "answer" else answer_base
                        decision_value = phase if target == "decision" else decision_base
                        result = score_fields(row, answer_value, decision_value, wager_base)
                        result.update(split="test", target=target, direction_key=candidate["key"],
                            layer=candidate["layer"], alpha=choice["alpha"])
                        test_records.append(result)
                        bar.update(1)

        with (out / "samples.jsonl").open("w", encoding="utf-8") as stream:
            for row in validation_records + test_records:
                stream.write(json.dumps(row) + "\n")
        baseline_test = [row for row in test_records if row["target"] == "baseline"]
        test_interventions = {}
        baseline_by_key = {(row["id"], row["sample"]): row for row in baseline_test}
        for target in ("answer", "decision"):
            rows = [row for row in test_records if row["target"] == target]
            if selected[target]["direction_key"] is None:
                reason = selected[target]["selection_reason"]
                status = ("not_run_no_candidate_direction" if reason ==
                          "analysis selected no direction for this target" else
                          "not_run_no_validation_improvement")
                test_interventions[target] = dict(status=status,
                    reason=reason, metrics=None, metric_coverage=0,
                    paired_equal_family_brier_change=None)
                continue
            metric = "confidence_brier" if target == "answer" else "decision_brier"
            deltas = []
            for row in rows:
                baseline = baseline_by_key[(row["id"], row["sample"])]
                if row.get(metric) is not None and baseline.get(metric) is not None:
                    deltas.append(dict(family=row["family"], delta=row[metric] - baseline[metric]))
            test_interventions[target] = dict(status="tested_once_after_validation_selection",
                metrics={metric: family_mean(rows, metric)},
                metric_coverage=sum(row.get(metric) is not None for row in rows),
                paired_equal_family_brier_change=family_mean(deltas, "delta"),
                direction_key=selected[target]["direction_key"], alpha=selected[target]["alpha"])

        summary = dict(source_run=str(run_dir), analysis_dir=str(analysis_dir), selected=selected,
            validation=dict(baseline={metric: family_mean(
                [row for row in validation_records if row["target"] == "baseline"], metric)
                for metric in ("confidence_brier", "decision_brier")},
                candidate_sweep=sweep_diagnostics,
                candidate_sweep_count=len(validation_records) - len(by_split["validation"])),
            test=dict(n=len(by_split["test"]), baseline={metric: family_mean(baseline_test, metric)
                for metric in ("confidence_brier", "decision_brier")},
                interventions=test_interventions,
                test_families=sorted(f for f, split in split_map.items() if split == "test"),
                test_item_ids=sorted({row["id"] for row in by_split["test"]})),
            selection_rule="direction and alpha selected on validation families only",
            test_rule="only selected target-specific interventions evaluated on test; answer and other forecast fixed")
        write_json(out / "summary.json", summary)
        write_json(out / "status.json", dict(status="complete", finished=utc_now()))
    except BaseException as exc:
        write_json(out / "status.json", dict(status="failed", error=str(exc), finished=utc_now()))
        raise
    print(f"Saved {out}")
    return out


if __name__ == "__main__":
    main()
