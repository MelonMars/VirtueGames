"""Test validation-selected Unwelcome Truth directions on reserved test families."""
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
from llm import generate, generate_messages, load_model
from original.unwelcome_truth.protocol import (PROTOCOL, load_bank, parse_answer,
    personal_followup_message, score, user_message)

SELF_BENEFIT_MODE = "unwelcome-truth-self-benefit"
from runs import next_run_dir, utc_now, write_json


class ResidualAddition:
    def __init__(self, vector, alpha, site="decode"):
        self.vector, self.alpha, self.site = vector, alpha, site
        self.calls = self.applications = 0

    def __call__(self, value, hook):
        first = self.calls == 0
        self.calls += 1
        if self.site == "prompt" and not first:
            return value
        if value.ndim != 3 or value.shape[-1] != self.vector.numel():
            raise ValueError("Direction width does not match residual stream")
        self.applications += 1
        if self.alpha == 0:
            return value
        edited = value.clone()
        edited[:, -1, :] += self.alpha * self.vector.to(device=value.device, dtype=value.dtype)
        return edited


def summarize(rows):
    parsed = [row for row in rows if row["parse_ok"]]
    return dict(n=len(rows), parse_coverage=len(parsed),
        truthful_report_rate=(sum(row["answer_correct"] for row in parsed) / len(parsed) if parsed else None),
        pressure_compliance_rate=(sum(row["pressure_compliant"] for row in parsed) / len(parsed) if parsed else None),
        result_values={value: sum(row["answer"] == value for row in parsed) for value in sorted({row["answer"] for row in parsed})})


def main(argv=None):
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file
    from transformer_lens.model_bridge import TransformerBridge

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--analysis-dir", type=Path)
    parser.add_argument("--alphas", default="-4,-2,-1,0,1,2,4")
    parser.add_argument("--site", choices=("decode", "prompt"), default="decode")
    parser.add_argument("--n-samples", type=int)
    parser.add_argument("--max-items", type=int)
    parser.add_argument("--seed", type=int, default=2901)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--runtime", choices=("local", "runpod"), default="local")
    parser.add_argument("--model", help="Local copy of the source checkpoint if its recorded path moved")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    alphas = sorted(set(float(x) for x in args.alphas.split(",")) | {0.0})
    if not all(math.isfinite(alpha) for alpha in alphas) or not min(alphas) < 0 < max(alphas):
        raise ValueError("Provide finite alpha values of both signs")
    if args.n_samples is not None and args.n_samples < 1 or args.max_items is not None and args.max_items < 1:
        raise ValueError("Sample and item limits must be positive")

    run_dir = args.run_dir.resolve()
    analysis_dir = (args.analysis_dir or run_dir / "directions_unwelcome_truth").resolve()
    source = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    report = json.loads((analysis_dir / "report.json").read_text(encoding="utf-8"))
    if report.get("source_run") != str(run_dir) or report.get("source_config") != source:
        raise ValueError("Analysis provenance does not match the source run")
    cases = load_bank(run_dir / "cases.json")
    case_by_id = {case["id"]: case for case in cases}
    test_families = set(report["test_families_reserved"])
    if any(case["split"] == "test" and case["family"] not in test_families for case in cases):
        raise ValueError("Test-family manifest disagrees with the saved bank")
    directions_path = analysis_dir / "directions.safetensors"
    with safe_open(str(directions_path), framework="pt", device="cpu") as direction_file:
        meta = direction_file.metadata() or {}
    if set(json.loads(meta.get("test_families", "[]"))) != test_families:
        raise ValueError("Directions were not saved with this test-family split")
    train_families = set(json.loads(meta.get("training_families", "[]")))
    validation_families = set(json.loads(meta.get("validation_families", "[]")))
    if (train_families != {f for f, split in report["family_splits"].items() if split == "train"}
            or validation_families != {f for f, split in report["family_splits"].items() if split == "validation"}
            or train_families & validation_families or train_families & test_families
            or validation_families & test_families):
        raise ValueError("Direction provenance has overlapping family splits")
    directions = load_file(str(directions_path), device="cpu")
    if set(directions) != set(report["selected_directions"]):
        raise ValueError("Direction tensors do not match the validation-selected list")
    candidate_specs = []
    for key, tensor in directions.items():
        match = re.fullmatch(r"slot([01])\.(blocks\.(\d+)\.hook_resid_post)", key)
        if not match:
            raise ValueError(f"Invalid direction key: {key}")
        vector = tensor.float().flatten()
        if not torch.isfinite(vector).all() or abs(float(vector.norm()) - 1.0) > 1e-3:
            raise ValueError(f"Expected a finite unit direction: {key}")
        candidate_specs.append(dict(key=key, slot=int(match[1]), hook=match[2],
                                    layer=int(match[3]), vector=vector))

    controls = {row["id"]: row for row in
                (json.loads(line) for line in (run_dir / "controls.jsonl").read_text().splitlines() if line.strip())}
    self_benefit = source.get("mode") == SELF_BENEFIT_MODE
    compliance_field = "self_benefit_compliance_rate" if self_benefit else "pressure_compliance_rate"
    source_rows = [json.loads(line) for line in (run_dir / "samples.jsonl").read_text().splitlines() if line.strip()]
    budget_rows = {(row["id"], row["sample"]): row for row in source_rows}
    active_ids = {row["id"] for row in source_rows if row.get("incentive_active")}
    items = [case for case in cases if case["split"] == "test" and case["family"] in test_families
             and controls.get(case["id"], {}).get("answer_correct") is True
             and (not self_benefit or case["id"] in active_ids)]
    if args.max_items is not None:
        items = sorted(items, key=lambda row: row["id"])[:args.max_items]
    if not items:
        raise ValueError("No neutral-control-passing items in the reserved test families")
    n_samples = args.n_samples or int(source["n_samples"])

    runtime = None
    activation_records = (run_dir / "activations" / "index.jsonl")
    if activation_records.exists():
        first = next((json.loads(line) for line in activation_records.read_text().splitlines() if line.strip()), {})
        runtime = first.get("runtime", {})
    values = {field.name: source[field.name] for field in fields(RunConfig) if field.name in source}
    values["questions"] = Path(source.get("questions_file", run_dir / "cases.json"))
    values["difficulty"] = tuple(source.get("difficulty_filter") or ())
    values.update(backend="transformers", extract_activations=False, runtime=args.runtime,
                  device=args.device, sample_batch_size=1)
    if args.model:
        values["model"] = args.model
    resolved_revision = runtime.get("resolved_revision") if runtime else None
    if resolved_revision and not args.model:
        values["revision"] = resolved_revision
    config = RunConfig(**values)
    out = next_run_dir(args.out or run_dir / "steering_check")
    write_json(out / "config.json", dict(source_run=str(run_dir), analysis_dir=str(analysis_dir),
        test_families=sorted(test_families), test_items=[case["id"] for case in items],
        candidates=[{key: value for key, value in spec.items() if key != "vector"} for spec in candidate_specs],
        alphas=alphas, site=args.site, n_samples=n_samples,
        generations=len(items) * n_samples * (1 + len(candidate_specs) * len(alphas)) * (2 if self_benefit else 1),
        generation=config.metadata("unwelcome-truth-steering", len(items)),
        note=("Candidates ranked on validation families; each held-out baseline and intervention receives an actual follow-up "
              "generation with the token budget earned by its own result. Episodes are independent."
              if self_benefit else "Candidates ranked on validation families; this sweep reads only held-out family cases.")))
    write_json(out / "status.json", dict(status="running", started=utc_now()))
    rows, grouped = [], defaultdict(list)
    total = len(items) * n_samples * (1 + len(candidate_specs) * len(alphas)) * (2 if self_benefit else 1)
    try:
        with load_model(config) as llm:
            if resolved_revision and getattr(llm.model.config, "_commit_hash", None) not in (None, resolved_revision):
                raise ValueError("Loaded checkpoint revision differs from activation source")
            bridge = TransformerBridge.boot_transformers(str(config.model), hf_model=llm.model,
                tokenizer=llm.tokenizer, device=llm.model.device, dtype=llm.model.dtype, revision=config.revision)
            bridge.eval()
            with (out / "samples.jsonl").open("w", encoding="utf-8") as stream, \
                    tqdm(total=total, desc="Unwelcome Truth steering", unit="generation") as progress:
                for item_index, case in enumerate(items):
                    for sample in range(n_samples):
                        seed = args.seed + item_index * n_samples + sample
                        source_row = budget_rows.get((case["id"], sample), {})
                        token_budget = int(source_row.get("token_budget", config.max_tokens))
                        if self_benefit:
                            user = user_message(case, self_incentive=True, token_budget=token_budget,
                                bonus_tokens=int(source.get("bonus_tokens", 128)), reward_active=True)
                        else:
                            user = user_message(case, pressure=True)
                            token_budget = config.max_tokens
                        baseline_text = generate(llm, source["system_prompt"], user, seed=seed,
                            temperature=config.temperature, max_tokens=token_budget)
                        baseline_parsed = parse_answer(baseline_text)
                        baseline_score = score(baseline_parsed["answer"], case)
                        baseline = dict(**baseline_parsed, **baseline_score)
                        if self_benefit:
                            bonus = int(source.get("bonus_tokens", 128)) if baseline["incentive_compliant"] else 0
                            followup_budget = token_budget + bonus
                            followup_messages = [dict(role="system", content=source["system_prompt"]),
                                dict(role="user", content=user), dict(role="assistant", content=baseline_text),
                                dict(role="user", content=personal_followup_message(case, followup_budget))]
                            generate_messages(llm, followup_messages, seed=seed + 1,
                                temperature=config.temperature, max_tokens=followup_budget,
                                activation_context=dict(skip_activation_capture=True))
                            baseline["followup_token_budget"] = followup_budget
                            progress.update(1)
                        base_row = dict(id=case["id"], family=case["family"], sample=sample,
                            seed=seed, target="baseline", direction_key=None, alpha=0.0,
                            raw=baseline_text, **baseline)
                        rows.append(base_row)
                        stream.write(json.dumps(base_row) + "\n")
                        progress.update(1)
                        for spec in candidate_specs:
                            for alpha in alphas:
                                addition = ResidualAddition(spec["vector"], alpha, args.site)
                                with bridge.hooks(fwd_hooks=[(spec["hook"], addition)]):
                                    text = generate(llm, source["system_prompt"], user, seed=seed,
                                        temperature=config.temperature, max_tokens=token_budget)
                                if not addition.applications:
                                    raise RuntimeError(f"Steering hook did not fire for {spec['key']}")
                                if alpha == 0 and text != baseline_text:
                                    raise RuntimeError("Zero-strength steering changed the response")
                                parsed = parse_answer(text)
                                scored = score(parsed["answer"], case)
                                if self_benefit:
                                    bonus = int(source.get("bonus_tokens", 128)) if scored["incentive_compliant"] else 0
                                    followup_budget = token_budget + bonus
                                    followup_messages = [dict(role="system", content=source["system_prompt"]),
                                        dict(role="user", content=user), dict(role="assistant", content=text),
                                        dict(role="user", content=personal_followup_message(case, followup_budget))]
                                    generate_messages(llm, followup_messages, seed=seed + 1,
                                        temperature=config.temperature, max_tokens=followup_budget,
                                        activation_context=dict(skip_activation_capture=True))
                                    scored["followup_token_budget"] = followup_budget
                                result = dict(id=case["id"], family=case["family"], sample=sample,
                                    seed=seed, target="steered", direction_key=spec["key"],
                                    slot=spec["slot"], layer=spec["layer"], alpha=alpha,
                                    hook_applications=addition.applications, raw=text,
                                    **parsed, **scored)
                                result["paired_truthful_change"] = (
                                    int(bool(scored["answer_correct"])) - int(bool(baseline["answer_correct"]))
                                    if scored["answer_correct"] is not None else None)
                                grouped[spec["key"], alpha].append(result)
                                rows.append(result)
                                stream.write(json.dumps(result) + "\n")
                                stream.flush()
                                progress.update(1)
                                if self_benefit:
                                    progress.update(1)
                                progress.set_postfix(case=case["id"], layer=spec["layer"], a=alpha, refresh=False)
        summaries = []
        for (direction_key, alpha), subset in sorted(grouped.items()):
            by_family = {}
            for family in sorted({row["family"] for row in subset}):
                group = [row for row in subset if row["family"] == family]
                by_family[family] = dict(n=len(group), truthful_rate=sum(r["answer_correct"] is True for r in group) / len(group),
                    **{compliance_field: sum((r["incentive_compliant"] if self_benefit else r["pressure_compliant"]) is True
                                             for r in group) / len(group)})
            candidate = next(spec for spec in candidate_specs if spec["key"] == direction_key)
            summaries.append(dict(direction_key=direction_key, layer=candidate["layer"], slot=candidate["slot"],
                alpha=alpha, n=len(subset), truthful_rate=sum(r["answer_correct"] is True for r in subset) / len(subset),
                **{compliance_field: sum((r["incentive_compliant"] if self_benefit else r["pressure_compliant"]) is True
                                         for r in subset) / len(subset)},
                paired_truthful_change=sum(r["paired_truthful_change"] for r in subset if r["paired_truthful_change"] is not None)
                    / sum(r["paired_truthful_change"] is not None for r in subset)
                    if any(r["paired_truthful_change"] is not None for r in subset) else None,
                by_family=by_family))
        write_json(out / "summary.json", summaries)
        write_json(out / "status.json", dict(status="complete", finished=utc_now()))
    except BaseException as exc:
        write_json(out / "status.json", dict(status="failed", error=str(exc), finished=utc_now()))
        raise
    print(f"Saved {out}")
    return out


if __name__ == "__main__":
    main()
