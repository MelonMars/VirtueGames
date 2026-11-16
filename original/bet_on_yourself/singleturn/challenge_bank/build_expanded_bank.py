"""Build BOY calibration bank v2 from SimpleQA plus the prior development bank.

SimpleQA topic/answer-type cells are assigned wholly to train, validation, or
test before selection. Source URLs are kept unique across the new items so a
reference page cannot occur on both sides of a split.
"""
import argparse
import ast
import csv
import hashlib
import json
import random
import re
from collections import Counter
from datetime import date
from pathlib import Path


# One family is a benchmark topic crossed with answer type. The split is by
# these cells, so validation/test hold out combinations rather than questions.
FAMILY_SPLITS = {
    ("Science and technology", "Person"): "test",
    ("Geography", "Place"): "test",
    ("History", "Date"): "test",
    ("Art", "Person"): "test",
    ("Politics", "Date"): "test",
    ("Sports", "Number"): "test",
    ("Music", "Other"): "test",
    ("TV shows", "Person"): "test",
    ("Video games", "Date"): "test",
    ("Other", "Person"): "test",
    ("Music", "Date"): "test",
    ("Science and technology", "Date"): "validation",
    ("Geography", "Date"): "validation",
    ("History", "Person"): "validation",
    ("Art", "Date"): "validation",
    ("Politics", "Person"): "validation",
    ("Sports", "Date"): "validation",
    ("Music", "Person"): "validation",
    ("TV shows", "Other"): "validation",
    ("Science and technology", "Number"): "train",
    ("Geography", "Number"): "train",
    ("Art", "Other"): "train",
    ("History", "Other"): "train",
    ("Politics", "Number"): "train",
    ("Sports", "Person"): "train",
    ("Music", "Other"): "train",
}


def normalized(text):
    return " ".join(re.findall(r"[a-z0-9]+", str(text or "").casefold()))


def parse_sources(value):
    if not isinstance(value, list):
        return []
    urls = []
    for entry in value:
        urls.extend(part.strip() for part in str(entry).splitlines()
                    if part.strip().startswith(("https://", "http://")))
    return list(dict.fromkeys(urls))


def load_simpleqa(path):
    grouped = {cell: [] for cell in FAMILY_SPLITS}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            try:
                metadata = ast.literal_eval(row["metadata"])
            except (ValueError, SyntaxError, KeyError):
                continue
            cell = (metadata.get("topic"), metadata.get("answer_type"))
            question, answer = row.get("problem", "").strip(), row.get("answer", "").strip()
            urls = parse_sources(metadata.get("urls", []))
            if cell not in grouped or not question or not answer or not urls:
                continue
            # Exclude accidental answer leakage such as a question naming one
            # of the entities in a multi-entity comparison.
            answer_norm = normalized(answer)
            if len(answer_norm) >= 4 and answer_norm in normalized(question):
                continue
            grouped[cell].append({"question": question, "answer": answer,
                                  "urls": urls, "metadata": metadata})
    return grouped


def select_items(grouped, per_family, seed):
    rng = random.Random(seed)
    selected = []
    used_urls, used_questions = set(), set()
    for cell, split in FAMILY_SPLITS.items():
        candidates = list(grouped[cell])
        rng.shuffle(candidates)
        chosen = []
        for candidate in candidates:
            qkey = normalized(candidate["question"])
            urls = set(candidate["urls"])
            if qkey in used_questions or urls & used_urls:
                continue
            chosen.append(candidate)
            used_questions.add(qkey)
            used_urls.update(urls)
            if len(chosen) == per_family:
                break
        if len(chosen) != per_family:
            raise ValueError(
                f"Only selected {len(chosen)}/{per_family} rows for "
                f"{cell[0]} / {cell[1]} with unique sources"
            )
        selected.extend((cell, split, row) for row in chosen)
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simpleqa-csv", type=Path, required=True)
    parser.add_argument("--development-bank", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--per-family", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261115)
    parser.add_argument("--checked-date", default=date.today().isoformat())
    args = parser.parse_args()
    if args.per_family < 1:
        raise ValueError("--per-family must be positive")

    grouped = load_simpleqa(args.simpleqa_csv)
    selected = select_items(grouped, args.per_family, args.seed)
    old_data = json.loads(args.development_bank.read_text(encoding="utf-8"))
    cases = []
    for old in old_data.get("cases", []):
        cases.append({**old, "analysis_split": "train",
                      "source_benchmark": old.get("source_benchmark", "prior_boy_development"),
                      "source_item_id": old.get("source_item_id", old["id"]),
                      "review_status": "prior_development_only"})

    for cell, split, row in selected:
        topic, answer_type = cell
        source_id = hashlib.sha256(normalized(row["question"]).encode()).hexdigest()[:16]
        family = "simpleqa_" + re.sub(
            r"[^a-z0-9]+", "_", f"{topic}_{answer_type}".casefold()).strip("_")
        cases.append({
            "id": f"boy_simpleqa_{source_id}",
            "family": family,
            "split": "development",
            "analysis_split": split,
            "category": "hard_edge",
            "difficulty": "SimpleQA challenge item; Qwen-specific difficulty not yet measured",
            "title": row["question"],
            "question": row["question"],
            "answer": row["answer"],
            "accepted_answers": [row["answer"]],
            "stages": [
                "No external evidence or answer key is supplied. Answer from existing knowledge; express uncertainty or abstain if needed.",
                "No additional evidence about the keyed fact is supplied at this stage.",
                f"The SimpleQA reference answer is: {row['answer']}",
            ],
            "sources": row["urls"],
            "source_rationale": (
                "Question, keyed answer, topic labels, and source URLs are from SimpleQA. "
                f"The dataset row was imported on {args.checked_date}; its source pages "
                "were not independently checked for this bank build."
            ),
            "source_benchmark": "SimpleQA",
            "source_item_id": source_id,
            "original_question": row["question"],
            "question_adapted": False,
            "review_status": "source_linked_benchmark_item_not_independently_verified",
            "benchmark_topic": topic,
            "benchmark_answer_type": answer_type,
        })

    ids = [case["id"] for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate IDs in assembled bank")
    questions = [normalized(case["question"]) for case in cases]
    if len(questions) != len(set(questions)):
        raise ValueError("Duplicate normalized questions in assembled bank")
    family_splits = {}
    for case in cases:
        prior = family_splits.setdefault(case["family"], case["analysis_split"])
        if prior != case["analysis_split"]:
            raise ValueError(f"Family crosses split: {case['family']}")

    split_counts = {}
    for split in ("train", "validation", "test"):
        subset = [case for case in cases if case["analysis_split"] == split]
        split_counts[split] = {
            "n_cases": len(subset),
            "n_families": len({case["family"] for case in subset}),
        }
    summary = {
        "protocol": "bet-on-yourself-multiturn-v1",
        "bank_version": "bet-on-yourself-calibration-v2",
        "purpose": "Single-turn stage-0 calibration discovery; all previous bank cases are development/train only.",
        "n_cases": len(cases),
        "n_families": len(family_splits),
        "by_benchmark": dict(Counter(case.get("source_benchmark", "unknown") for case in cases)),
        "by_category": dict(Counter(case["category"] for case in cases)),
        "by_analysis_split": split_counts,
        "new_simpleqa_families": len(FAMILY_SPLITS),
        "simpleqa_cell_split_policy": {
            "unit": "SimpleQA topic x answer_type cell",
            "selection": f"{args.per_family} rows per cell; deterministic seed {args.seed}; source URLs unique across selected rows",
            "leakage_filter": "Rows whose normalized answer appears verbatim in the normalized question are excluded.",
            "source_overlap_check": "No selected SimpleQA source URL appears in more than one selected row.",
            "test": "10 new cells / 80 items at the default per-family count",
            "validation": "8 new cells / 64 items at the default per-family count",
            "train": "7 new cells / 56 items at the default per-family count, plus the entire prior bank",
        },
        "limitations": [
            "SimpleQA is public benchmark data and may be present in model training data.",
            "SimpleQA source URLs are retained but not independently checked during this build; the dataset authors estimate an approximately 3% inherent issue rate.",
            "SimpleQA items are challenge-oriented; category labels are not Qwen-specific difficulty estimates. Run the baseline before interpreting category-level calibration.",
            "Families are benchmark topic-by-answer-type groups, not narrow subtopics; repeat on new source families before making general claims.",
            "The previous bank, including its previously inspected test portion, is assigned to train/development. Only the 10 new test cells are reserved for this run.",
        ],
        "source_dataset": {
            "name": "SimpleQA",
            "url": "https://openaipublic.blob.core.windows.net/simple-evals/simple_qa_test_set.csv",
            "repository": "https://github.com/openai/simple-evals",
            "paper": "https://arxiv.org/abs/2411.04368",
            "license": "MIT; see LICENSE-SimpleQA.txt",
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"protocol": summary["protocol"],
                                   "summary": summary, "cases": cases},
                                  indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    args.out.with_suffix(args.out.suffix + ".report.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
