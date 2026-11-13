"""Tune confidence-only steering on validation families and evaluate once on test families."""
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
    PROTOCOL, CONFIDENCE_SYSTEM, assemble_response,
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
    """Compare candidates only when they score the same item/sample set."""
    baseline_keys = {(row["id"], row["sample"]) for row in baseline_rows
                     if row.get(field) is not None}
    candidate_keys = {(row["id"], row["sample"]) for row in rows
                      if row.get(field) is not None}
    if not baseline_keys or candidate_keys != baseline_keys:
        return None
    return family_mean(rows, field)


def main(argv=None):
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file
    from transformer_lens.model_bridge import TransformerBridge

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Answer-first confidence BOY run")
    parser.add_argument("--analysis-dir", type=Path)
    parser.add_argument("--alphas", default="-4,-2,-1,1,2,4",
                        help="Validation sweep strengths; finite values of both signs")
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
        raise ValueError(f"Expected {PROTOCOL}; use steer_calibration.py for the one-response protocol")
    if analysis.get("source_run") != str(run_dir) or analysis.get("source_config") != source:
        raise ValueError("Analysis provenance does not match the source run")
    split_map = analysis["family_splits"]
    for split in ("validation", "test"):
        if not any(value == split for value in split_map.values()):
            raise ValueError(f"No {split} families in the analysis split")

    direction_path = analysis_dir / "directions.safetensors"
    with safe_open(str(direction_path), framework="pt", device="cpu") as file:
        metadata = file.metadata() or {}
    for split, metadata_key in (("test", "test_families"), ("train", "training_families"),
                                ("validation", "validation_families")):
        expected = {family for family, value in split_map.items() if value == split}
        if set(json.loads(metadata.get(metadata_key, "[]"))) != expected:
            raise ValueError("Direction tensors do not match the report's family split")
    directions = load_file(str(direction_path), device="cpu")
    candidate_keys = {key for values in analysis.get("selected_candidates", {}).values() for key in values}
    if set(directions) != candidate_keys:
        raise ValueError("Selected candidate list does not match saved direction tensors")
    evidence = {row["direction_key"]: row for values in analysis.get("candidates", {}).values()
                for row in values if row.get("direction_key")}
    candidates = []
    for target in ("answer", "decision"):
        for key in analysis.get("selected_candidates", {}).get(target, []):
            match = re.fullmatch(r"(answer|decision)\.layer_(\d+)\.hook_resid_post", key)
            if not match or match.group(1) != target:
                raise ValueError(f"Invalid candidate key: {key}")
            vector = directions[key].float().flatten()
            if not torch.isfinite(vector).all() or abs(float(vector.norm()) - 1.0) > 1e-3:
                raise ValueError(f"Expected a finite unit direction: {key}")
            candidates.append(dict(key=key, target=target, layer=int(match.group(2)),
                hook=f"blocks.{match.group(2)}.hook_resid_post", vector=vector,
                validation_correlation=evidence.get(key, {}).get("validation_correlation"),
                validation_r2=evidence.get(key, {}).get("validation_r2")))
    if not candidates:
        raise ValueError("No selected directions")

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
        if not rows:
            raise ValueError(f"No {split} samples remain after limits")
        if any(not row.get("confidence_prompt") or not row.get("answer_response") for row in rows):
            raise ValueError(f"Missing confidence phase prompt or fixed answer in {split} rows")
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
    plan = dict(source_run=str(run_dir), analysis_dir=str(analysis_dir),
        analysis_seed=analysis["seed"], layer_range=analysis.get("layer_range"),
        validation_families=sorted(f for f, split in split_map.items() if split == "validation"),
        test_families=sorted(f for f, split in split_map.items() if split == "test"),
        alphas=alphas, max_items=args.max_items, n_samples=n_samples,
        answer_is_fixed=True, steering_scope="generated confidence-and-wager response only",
        selection_rule="choose direction and alpha by equal-family validation Brier; evaluate selected choices once on test",
        seed_policy="paired per item/sample; answer commitment is reused unchanged")
    write_json(out / "config.json", plan)
    write_json(out / "status.json", dict(status="running", started=utc_now()))

    def scored(row, confidence_text):
        response, _ = assemble_response(row["answer_response"], confidence_text)
        return dict(id=row["id"], family=row["family"], category=row["category"],
            sample=row["sample"], committed_answer=row["answer"], raw=response,
            confidence_phase_raw=confidence_text, **score_response(response, cases[row["id"]]))

    def seed_for(row, split):
        item_seed = int(hashlib.sha256(row["id"].encode()).hexdigest()[:8], 16)
        return args.seed + (0 if split == "validation" else 1000000) + item_seed * n_samples + int(row["sample"])

    validation_records = []
    selected = {}
    test_records = []
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
            work = len(val_rows) * (1 + len(candidates) * len(alphas)) + len(test_rows) * (1 + len(candidates))
            with tqdm(total=work, desc="Confidence-only calibration steering", unit="generation") as bar:
                for row in val_rows:
                    seed = seed_for(row, "validation")
                    kwargs = dict(seed=seed, temperature=config.temperature, max_tokens=min(config.max_tokens, 96))
                    base_text = generate(llm, CONFIDENCE_SYSTEM, row["confidence_prompt"], **kwargs)
                    base = scored(row, base_text)
                    base.update(split="validation", target="baseline", direction_key=None, layer=None, alpha=0.0)
                    validation_records.append(base)
                    bar.update(1)
                    for candidate in candidates:
                        for alpha in alphas:
                            addition = ResidualAddition(candidate["vector"], alpha, "decode")
                            with bridge.hooks(fwd_hooks=[(candidate["hook"], addition)]):
                                phase = generate(llm, CONFIDENCE_SYSTEM, row["confidence_prompt"], **kwargs)
                            if not addition.applications:
                                raise RuntimeError(f"Decode steering hook did not fire for {candidate['key']}")
                            result = scored(row, phase)
                            result.update(split="validation", target=candidate["target"],
                                direction_key=candidate["key"], layer=candidate["layer"], alpha=alpha)
                            validation_records.append(result)
                            bar.update(1)

                # Select one direction/strength per calibration target using
                # validation families only. Alpha zero is the fallback.
                for target, metric in (("answer", "confidence_brier"), ("decision", "decision_brier")):
                    baseline_rows = [r for r in validation_records if r["target"] == "baseline"]
                    baseline_score = family_mean(baseline_rows, metric)
                    options = []
                    for candidate in candidates:
                        if candidate["target"] != target:
                            continue
                        for alpha in alphas:
                            rows = [r for r in validation_records if r.get("direction_key") == candidate["key"]
                                    and r["alpha"] == alpha]
                            options.append((matched_family_mean(rows, baseline_rows, metric), candidate, alpha))
                    options = [option for option in options if option[0] is not None]
                    best = min(options, key=lambda option: option[0]) if options else None
                    selected[target] = (dict(direction_key=best[1]["key"], alpha=best[2],
                        validation_equal_family_brier=best[0], baseline_validation_equal_family_brier=baseline_score)
                        if best and baseline_score is not None and best[0] < baseline_score else
                        dict(direction_key=None, alpha=0.0,
                             validation_equal_family_brier=baseline_score,
                             baseline_validation_equal_family_brier=baseline_score))

                selected_specs = {target: next((c for c in candidates
                    if c["key"] == choice["direction_key"]), None)
                    for target, choice in selected.items()}
                for row in test_rows:
                    seed = seed_for(row, "test")
                    kwargs = dict(seed=seed, temperature=config.temperature, max_tokens=min(config.max_tokens, 96))
                    base_text = generate(llm, CONFIDENCE_SYSTEM, row["confidence_prompt"], **kwargs)
                    base = scored(row, base_text)
                    base.update(split="test", target="baseline", direction_key=None, layer=None, alpha=0.0)
                    test_records.append(base)
                    bar.update(1)
                    for target in ("answer", "decision"):
                        candidate = selected_specs[target]
                        if candidate is None:
                            continue
                        alpha = selected[target]["alpha"]
                        addition = ResidualAddition(candidate["vector"], alpha, "decode")
                        with bridge.hooks(fwd_hooks=[(candidate["hook"], addition)]):
                            phase = generate(llm, CONFIDENCE_SYSTEM, row["confidence_prompt"], **kwargs)
                        if not addition.applications:
                            raise RuntimeError(f"Decode steering hook did not fire for {candidate['key']}")
                        result = scored(row, phase)
                        result.update(split="test", target=target,
                            direction_key=candidate["key"], layer=candidate["layer"], alpha=alpha)
                        test_records.append(result)
                        bar.update(1)

        with (out / "samples.jsonl").open("w", encoding="utf-8") as stream:
            for row in validation_records + test_records:
                stream.write(json.dumps(row) + "\n")
        choices = {target: {**choice, **({"validation_correlation": next(
            c["validation_correlation"] for c in candidates if c["key"] == choice["direction_key"])}
            if choice["direction_key"] else {})} for target, choice in selected.items()}
        baseline_test_rows = [r for r in test_records if r["target"] == "baseline"]
        baseline_test_by_key = {(r["id"], r["sample"]): r for r in baseline_test_rows}
        test_interventions = {}
        for target in ("answer", "decision"):
            rows = [r for r in test_records if r["target"] == target]
            paired = []
            for row in rows:
                baseline = baseline_test_by_key[(row["id"], row["sample"])]
                for metric in ("confidence_brier", "decision_brier"):
                    if row.get(metric) is not None and baseline.get(metric) is not None:
                        paired.append(dict(family=row["family"], metric=metric,
                                           delta=row[metric] - baseline[metric]))
            test_interventions[target] = dict(
                metrics={metric: family_mean(rows, metric)
                         for metric in ("confidence_brier", "decision_brier")},
                metric_coverage={metric: sum(r.get(metric) is not None for r in rows)
                                 for metric in ("confidence_brier", "decision_brier")},
                paired_equal_family_brier_change={metric: family_mean(
                    [dict(family=row["family"], change=row["delta"])
                     for row in paired if row["metric"] == metric], "change")
                    for metric in ("confidence_brier", "decision_brier")},
                decision_accuracy=(sum(bool(r["decision_correct"]) for r in rows) / len(rows)
                                   if rows else None),
                abstention_rate=(sum(bool(r["is_abstention"]) for r in rows) / len(rows)
                                 if rows else None))
        summary = dict(source_run=str(run_dir), analysis_dir=str(analysis_dir),
            selection="alpha and direction selected on validation families only",
            test_evaluation="one selected intervention per target; fixed answers; test metrics not used for selection",
            selected=choices,
            validation=dict(baseline={metric: family_mean(
                [r for r in validation_records if r["target"] == "baseline"], metric)
                for metric in ("confidence_brier", "decision_brier")},
                candidate_sweep_count=len(validation_records) - len(by_split["validation"])),
            test=dict(n=len(test_rows), baseline={metric: family_mean(
                baseline_test_rows, metric)
                for metric in ("confidence_brier", "decision_brier")},
                interventions=test_interventions),
            test_families=sorted(f for f, split in split_map.items() if split == "test"),
            test_item_ids=sorted({r["id"] for r in test_rows}))
        write_json(out / "summary.json", summary)
        write_json(out / "status.json", dict(status="complete", finished=utc_now()))
    except BaseException as exc:
        write_json(out / "status.json", dict(status="failed", error=str(exc), finished=utc_now()))
        raise
    print(f"Saved {out}")
    return out


if __name__ == "__main__":
    main()
