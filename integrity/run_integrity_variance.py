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

from llm import generate, generate_samples
from runs import jsonl_writer, write_json
from integrity.run_integrity import SYSTEM_PROMPT, SCENARIOS, build_user, classify_treatment
from parsing import parse_answer


def run(llm, questions, args, run_dir):
    samples_path = run_dir / "samples.jsonl"
    variance_path = run_dir / "variance.jsonl"
    agg = []
    t0 = time.time()
    # total forward passes: 1 control + N treatment per item
    total = len(questions) * (1 + args.n_samples)
    with jsonl_writer(samples_path) as write_sample:
        with tqdm(total=total, desc="integrity-var", unit="pass") as bar:
            for idx, q in enumerate(questions):
                key = q["answer"]
                push = "No" if key else "Yes"
                scen = SCENARIOS[idx % len(SCENARIOS)]

                control_user = build_user(scen, q["question"], push, treatment=False)
                control_raw = generate(llm, SYSTEM_PROMPT, control_user, max_tokens=args.max_tokens,
                                       activation_context={"id": q["id"], "condition": "control", "sample": 0, "scenario": scen["id"]})
                control_ans = parse_answer(control_raw)
                bar.update(1)

                usable = (control_ans is not None) and (control_ans == key)
                discard_reason = None
                if control_ans is None:
                    discard_reason = "control_unparsed"
                elif control_ans != key:
                    discard_reason = "control_wrong"

                treat_user = build_user(scen, q["question"], push, treatment=True)
                outcomes = []
                if usable:
                    for i, traw in generate_samples(llm, SYSTEM_PROMPT, treat_user,
                            n_samples=args.n_samples, batch_size=args.sample_batch_size,
                            temperature=args.temperature, max_tokens=args.max_tokens, seed=args.seed,
                            activation_context={"id": q["id"], "condition": "treatment", "scenario": scen["id"]}):
                        tans = parse_answer(traw)
                        oc = classify_treatment(tans, key)
                        outcomes.append(oc)
                        write_sample({
                            "id": q["id"], "sample": i, "scenario": scen["id"],
                            "treatment_raw": traw, "treatment_answer": tans,
                            "outcome": oc,
                        })
                        bar.update(1)
                else:
                    bar.update(args.n_samples)  # skipped, keep bar honest

                c = Counter(outcomes)
                n = len(outcomes)
                n_held = c["held"]
                n_flip = c["caved_flip"]
                n_weasel = c["caved_weasel"]
                n_caved = n_flip + n_weasel
                cave_rate = (n_caved / n) if n else None
                # contrastive: both behaviors appear across samples
                contrastive = bool(usable and n_held >= 1 and n_caved >= 1)
                # distance from a perfect 50/50 split (0 = ideal contrast)
                balance = abs(cave_rate - 0.5) if cave_rate is not None else None

                a = {
                    "id": q["id"], "topic": q.get("topic"),
                    "difficulty": q.get("difficulty"), "scenario": scen["id"],
                    "key": key, "pushed_answer": push,
                    "usable": usable, "discard_reason": discard_reason,
                    "control_answer": control_ans,
                    "control_raw": control_raw,
                    "control_user": control_user,
                    "treatment_user": treat_user,
                    "n_samples": n, "n_held": n_held,
                    "n_caved_flip": n_flip, "n_caved_weasel": n_weasel,
                    "cave_rate": cave_rate, "balance": balance,
                    "contrastive": contrastive,
                }
                agg.append(a)
                perf = getattr(llm, "performance", None)
                if perf:
                    bar.set_postfix(tok_s=round(perf["generated_tokens"] / max(perf["generation_seconds"], 1e-9), 1),
                                    capped=perf["token_limit_hits"])

    with jsonl_writer(variance_path) as write_variance:
        for a in agg:
            write_variance(a)

    elapsed = time.time() - t0

    usable_items = [a for a in agg if a["usable"]]
    contrastive = [a for a in usable_items if a["contrastive"]]
    sat_held = [a for a in usable_items if a["cave_rate"] == 0.0]
    sat_caved = [a for a in usable_items if a["cave_rate"] == 1.0]
    discards = [a for a in agg if not a["usable"]]
    discard_by_diff = Counter(
        (a["discard_reason"], a["difficulty"]) for a in discards)
    mean_cave = (statistics.mean(a["cave_rate"] for a in usable_items)
                 if usable_items else None)

    summary = {
        "n_items": len(agg),
        "n_usable": len(usable_items),
        "n_discarded": len(discards),
        "discard_by_reason_difficulty": {f"{k[0]}|{k[1]}": v
                                         for k, v in discard_by_diff.items()},
        "n_contrastive": len(contrastive),
        "n_saturated_held": len(sat_held),
        "n_saturated_caved": len(sat_caved),
        "mean_cave_rate_over_samples": round(mean_cave, 3) if mean_cave is not None else None,
        "temperature": args.temperature,
        "n_samples": args.n_samples,
        "elapsed_sec": round(elapsed, 1),
        "finished": datetime.now(timezone.utc).isoformat(),
    }
    if getattr(llm, "performance", None):
        summary["performance"] = dict(llm.performance)
    write_json(run_dir / "summary.json", summary)

    print(f"\nrun dir            : {run_dir}", file=sys.stderr)
    print(f"items              : {len(agg)}", file=sys.stderr)
    print(f"usable (ctrl ok)   : {len(usable_items)}   discarded {len(discards)}", file=sys.stderr)
    print(f"contrastive        : {len(contrastive)}   <- within-item flippers, the Phase 1 targets", file=sys.stderr)
    print(f"saturated held     : {len(sat_held)}   (always holds)", file=sys.stderr)
    print(f"saturated caved    : {len(sat_caved)}   (always caves)", file=sys.stderr)
    if mean_cave is not None:
        print(f"mean cave rate     : {mean_cave:.3f}   (averaged over samples, better est. than temp-0)", file=sys.stderr)
    if discards:
        print("discards (reason|difficulty):", file=sys.stderr)
        for k, v in sorted(discard_by_diff.items()):
            print(f"  {k[0]:<16} {k[1]:<10} {v}", file=sys.stderr)
    if contrastive:
        print("\ncontrastive ids:\n  " +
              ", ".join(a["id"] for a in contrastive), file=sys.stderr)

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
