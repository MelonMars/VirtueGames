"""integrity-variance experiment. Execute this file directly."""
import sys
from pathlib import Path

# Direct script execution needs the repository root for shared helpers.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import statistics
import time
from collections import Counter
from datetime import datetime, timezone
from tqdm import tqdm

from llm import generate, generate_samples, without_activation_capture
from contextlib import nullcontext
from runs import jsonl_writer, write_json
from integrity.run_integrity import SYSTEM_PROMPT, SCENARIOS, build_user, classify_treatment
from parsing import parse_answer


def run(llm, questions, args, run_dir):
    """Collect matched neutral/pressure samples; semantic labels are a separate pass."""
    from scoring import integrity_result
    from rescore_runs import integrity_summary
    items, all_rows, neutral_rows = [], [], []
    with jsonl_writer(run_dir / "samples.jsonl") as write_sample, \
            jsonl_writer(run_dir / "neutral_samples.jsonl") as write_neutral:
        for idx, q in enumerate(tqdm(questions, desc="Integrity matched sampling")):
            key = q["answer"]
            push = "No" if key else "Yes"
            scen = SCENARIOS[idx % len(SCENARIOS)]
            control_user = build_user(scen, q["question"], push, False)
            treat_user = build_user(scen, q["question"], push, True)
            control_raw = generate(llm, SYSTEM_PROMPT, control_user, temperature=0,
                seed=args.seed, max_tokens=args.max_tokens,
                activation_context=dict(id=q["id"], condition="control", sample=0))
            control = integrity_result(control_raw, key)
            treatment = []
            for condition, user, write, destination in (
                    ("neutral", control_user, write_neutral, neutral_rows),
                    ("treatment", treat_user, write_sample, treatment)):
                with without_activation_capture(llm) if condition == "neutral" else nullcontext():
                    for sample, raw in generate_samples(llm, SYSTEM_PROMPT, user,
                            n_samples=args.n_samples, batch_size=args.sample_batch_size,
                            temperature=args.temperature, max_tokens=args.max_tokens, seed=args.seed,
                            activation_context=dict(id=q["id"], condition=condition, scenario=scen["id"])):
                        result = integrity_result(raw, key)
                        row = dict(id=q["id"], sample=sample, condition=condition, scenario=scen["id"],
                                   response=raw, treatment_raw=raw, treatment_answer=result["answer"],
                                   key=key, question=q["question"], seed=args.seed + sample // args.sample_batch_size * args.sample_batch_size,
                                   **result)
                        write(row); destination.append(row)
            all_rows.extend(treatment)
            counts = Counter(r["outcome"] for r in treatment)
            items.append(dict(id=q["id"], topic=q.get("topic"), difficulty=q.get("difficulty"),
                key=key, pushed_answer=push, scenario=scen["id"],
                usable=control["outcome"] == "held", control_assessment=control,
                discard_reason=None if control["outcome"] == "held" else "control_"+control["outcome"],
                control_answer=control["answer"], control_raw=control_raw,
                control_user=control_user, treatment_user=treat_user,
                n_samples=len(treatment), n_held=counts["held"], n_caved_flip=counts["caved_flip"],
                n_caved_weasel=0, outcomes=dict(counts), contrastive=False, scoring_version=2))
    with jsonl_writer(run_dir / "variance.jsonl") as write:
        for item in items:
            write(item)
    summary = dict(n_items=len(items), treatment=integrity_summary(all_rows), neutral=integrity_summary(neutral_rows),
                   diagnostic_only=True, next_step="Run rescore_runs.py integrity with a validated semantic judge")
    if getattr(llm, "performance", None):
        summary["performance"] = dict(llm.performance)
    write_json(run_dir / "summary.json", summary)
    return summary


def main(argv=None):
    from config import parse_config
    from llm import load_model
    from questions import load_questions
    from runs import next_run_dir, utc_now

    config, output = parse_config(argv, max_tokens=1024, out="runs_integrity_variance",
                                  variance=True, calibration=False)
    questions = load_questions(config.questions, config.difficulty)
    print(f"Loading model: {config.model}", file=sys.stderr)
    with load_model(config) as llm:
        run_dir = next_run_dir(output)
        metadata = config.metadata("integrity-variance", len(questions))
        metadata["system_prompt"] = SYSTEM_PROMPT
        metadata.update(scoring_version=2, diagnostic_only=True, protocol="matched-neutral-pressure",
                        neutral_temperature=config.temperature, split_unit="explicit-family-manifest")
        write_json(run_dir / "questions.json", {"questions": questions})
        write_json(run_dir / "config.json", metadata)
        write_json(run_dir / "status.json", {"status": "running", "started": utc_now()})
        try:
            if config.extract_activations:
                llm.begin_run(run_dir)
            summary = run(llm, questions, config, run_dir)
        except BaseException as exc:
            write_json(run_dir / "status.json", {
                "status": "failed", "error": str(exc), "finished": utc_now()})
            raise
        write_json(run_dir / "status.json", {"status": "complete", "finished": utc_now()})
        return summary


if __name__ == "__main__":
    main()
