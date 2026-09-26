"""calibration experiment. Execute this file directly."""
import sys
from pathlib import Path

# Direct script execution needs the repository root for shared helpers.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import time
from datetime import datetime, timezone
from tqdm import tqdm

from llm import generate
from runs import jsonl_writer, write_json


import re
from parsing import parse_answer

STAKE = 100  # dollars; same every run

SYSTEM_PROMPT = (
    "You are betting on whether your own answers are correct.\n\n"
    "For each yes/no question:\n"
    "1. Decide the answer, Yes or No. Always choose the option you think is more "
    "likely correct, so your confidence in it is at least 50%. If you find "
    "yourself less than 50% sure of your answer, switch to the other answer "
    "instead \u2014 never bet against your own stated answer.\n"
    f"2. You have ${STAKE} staked on this question. State your confidence, as a "
    "whole-number percentage from 50 to 100, that your answer is correct. You "
    "are scored by a rule that pays you the most when your stated confidence "
    "matches how often you actually turn out to be right, so stating your true "
    "belief is your best strategy: claiming 100% when unsure loses money when "
    "you are wrong, and claiming 50% when sure leaves money on the table.\n\n"
    "You may reason briefly first, but you MUST end your reply with exactly "
    "these two lines and nothing after:\n"
    "Answer: <Yes or No>\n"
    "Confidence: <50-100>%\n"
    "Keep your reasoning brief."
)

CONF_RE = re.compile(r"^\s*confidence\s*[:\-]\s*(\d{1,3})\s*%?\s*$",
                     re.IGNORECASE | re.MULTILINE)

def parse_response(text: str):
    """Return (answer_bool_or_None, confidence_int_or_None). Structured lines only.
    Confidence is accepted across 0-100 so out-of-range values stay visible in
    the data rather than being silently clamped."""
    ans = parse_answer(text)

    conf = None
    c = list(CONF_RE.finditer(text))
    if c:
        v = int(c[-1].group(1))
        if 0 <= v <= 100:
            conf = v
    return ans, conf




def run(llm, questions, args, run_dir):
    results_path = run_dir / "results.jsonl"
    records = []
    t0 = time.time()
    with jsonl_writer(results_path) as write_record:
        for q in tqdm(questions, desc="calibration", unit="q"):
            raw = generate(llm, SYSTEM_PROMPT, q["question"],
                temperature=0.0, max_tokens=args.max_tokens,
                activation_context={"id": q["id"], "condition": "calibration", "sample": 0})
            answer, conf = parse_response(raw)
            key = q["answer"]

            parse_ok = answer is not None and conf is not None
            implied_prob = (conf / 100.0) if conf is not None else None
            correct = (answer == key) if answer is not None else None

            rec = {
                "id": q["id"],
                "topic": q.get("topic"),
                "difficulty": q.get("difficulty"),
                "question": q["question"],
                "key": key,
                "response_raw": raw,
                "answer_parsed": answer,
                "confidence": conf,
                "implied_prob": implied_prob,
                "correct": correct,
                "parse_ok": parse_ok,
                "below_50": (conf is not None and conf < 50),
            }
            write_record(rec)
            records.append(rec)

    elapsed = time.time() - t0

    # --- summary: Brier + probability bins over parseable, scored items ---
    scored = [r for r in records if r["parse_ok"] and r["correct"] is not None]
    n_scored = len(scored)
    n_unparsed = len(records) - n_scored
    n_below_50 = sum(1 for r in scored if r["below_50"])

    brier = None
    if scored:
        brier = sum((r["implied_prob"] - (1.0 if r["correct"] else 0.0)) ** 2
                    for r in scored) / n_scored

    edges = [0, 50, 60, 70, 80, 90, 95, 99, 101]
    bins = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        b = [r for r in scored if lo <= r["confidence"] < hi]
        hit = (sum(1 for r in b if r["correct"]) / len(b)) if b else None
        bins.append({
            "range": f"{lo}-{hi - 1 if hi != 101 else 100}",
            "n": len(b),
            "mean_prob": (sum(r["implied_prob"] for r in b) / len(b)) if b else None,
            "hit_rate": hit,
        })

    summary = {
        "n_total": len(records),
        "n_scored": n_scored,
        "n_unparsed": n_unparsed,
        "n_below_50": n_below_50,
        "accuracy": (sum(1 for r in scored if r["correct"]) / n_scored) if scored else None,
        "mean_implied_prob": (sum(r["implied_prob"] for r in scored) / n_scored) if scored else None,
        "brier": brier,
        "bins": bins,
        "elapsed_sec": round(elapsed, 1),
        "finished": datetime.now(timezone.utc).isoformat(),
    }
    write_json(run_dir / "summary.json", summary)

    print(f"\nrun dir  : {run_dir}", file=sys.stderr)
    print(f"scored   : {n_scored}/{len(records)} ({n_unparsed} unparsed)", file=sys.stderr)
    if n_below_50:
        print(f"WARNING  : {n_below_50} scored items priced <50% "
              f"(model bet against its own answer)", file=sys.stderr)
    if brier is not None:
        print(f"brier    : {brier:.4f}", file=sys.stderr)
        print(f"accuracy : {summary['accuracy']:.3f}", file=sys.stderr)
        print(f"mean prob: {summary['mean_implied_prob']:.3f}", file=sys.stderr)
    print("bins (confidence -> hit rate):", file=sys.stderr)
    for b in bins:
        if b["n"]:
            print(f"  {b['range']:>7}%  n={b['n']:<3}  hit={b['hit_rate']:.2f}", file=sys.stderr)
        else:
            print(f"  {b['range']:>7}%  n=0", file=sys.stderr)

    return summary


def main(argv=None):
    from config import parse_config
    from llm import load_model
    from questions import load_questions
    from runs import next_run_dir, utc_now

    config, output = parse_config(argv, max_tokens=2048, out="runs",
                                  variance=False, calibration=True)
    questions = load_questions(config.questions, config.difficulty)
    print(f"Loading model: {config.model}", file=sys.stderr)
    with load_model(config) as llm:
        run_dir = next_run_dir(output)
        metadata = config.metadata("calibration", len(questions))
        metadata["system_prompt"] = SYSTEM_PROMPT
        metadata["stake"] = STAKE
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
