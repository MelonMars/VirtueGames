#!/usr/bin/env python3
"""
Extract failed questions from a calibration run.

Two kinds of failure, reported separately:
  wrong    - model answered but the answer was incorrect (correct == false)
  unparsed - no answer/confidence could be extracted (parse_ok == false)

Wrong answers are sorted by confidence descending, so confidently-wrong
items (the overconfidence signal) are at the top.

Usage:
    python calibration/extract_failures.py calibration/runs/run-03
"""

import sys
from pathlib import Path

# Direct script execution needs the repository root for shared helpers.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from runs import latest_run, read_jsonl, write_json


def extract_failures(run_dir):
    run_dir = Path(run_dir)
    results = run_dir / "results.jsonl"
    if not results.exists():
        raise ValueError(f"no results.jsonl in {run_dir}")

    rows = read_jsonl(results)

    wrong = [r for r in rows if r.get("correct") is False]
    unparsed = [r for r in rows if not r.get("parse_ok")]

    # confidently-wrong first
    wrong.sort(key=lambda r: (r.get("confidence") or 0), reverse=True)

    out = {
        "run": str(run_dir),
        "n_total": len(rows),
        "n_wrong": len(wrong),
        "n_unparsed": len(unparsed),
        "wrong": [
            {
                "id": r["id"],
                "difficulty": r.get("difficulty"),
                "topic": r.get("topic"),
                "confidence": r.get("confidence"),
                "answer_parsed": r.get("answer_parsed"),
                "key": r.get("key"),
                "question": r.get("question"),
            }
            for r in wrong
        ],
        "unparsed": [
            {
                "id": r["id"],
                "difficulty": r.get("difficulty"),
                "topic": r.get("topic"),
                "question": r.get("question"),
            }
            for r in unparsed
        ],
    }

    out_path = run_dir / "failures.json"
    write_json(out_path, out)

    # terminal summary
    print(f"run      : {run_dir}")
    print(f"wrong    : {len(wrong)}")
    print(f"unparsed : {len(unparsed)}")
    print(f"written  : {out_path}\n")

    if wrong:
        print("confidently-wrong (top of list):")
        for r in wrong[:15]:
            print(f"  {r['id']}  conf={r.get('confidence')}%  "
                  f"said={r.get('answer_parsed')} key={r.get('key')}  "
                  f"[{r.get('difficulty')}]  {r['question'][:70]}")
    if unparsed:
        print("\nunparsed ids:", ", ".join(r["id"] for r in unparsed))
    return out


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description="Extract failed calibration questions")
    parser.add_argument("run_dir", type=Path, nargs="?")
    args = parser.parse_args(argv)
    return extract_failures(args.run_dir if args.run_dir is not None else latest_run(Path("runs")))


if __name__ == "__main__":
    main()
