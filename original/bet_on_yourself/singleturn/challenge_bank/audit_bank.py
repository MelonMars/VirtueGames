"""Audit structure and split balance for a BOY calibration challenge bank."""
import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


CATEGORIES = {
    "easy_verifiable", "medium_factual", "hard_edge", "unanswerable",
    "false_premise",
}
SPLITS = {"train", "validation", "test"}


def normalized(text):
    return " ".join(re.findall(r"[a-z0-9]+", str(text).casefold()))


def audit(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    cases = data.get("cases") if isinstance(data, dict) else None
    if not isinstance(cases, list) or not cases:
        raise ValueError("Expected a nonempty JSON object with a cases list")

    errors, warnings = [], []
    ids, questions = {}, {}
    family_splits = defaultdict(set)
    family_categories = defaultdict(Counter)
    category_counts, split_counts = Counter(), Counter()

    for index, case in enumerate(cases):
        label = case.get("id", f"row-{index + 1}")
        required = ("id", "family", "split", "category", "question", "answer",
                    "accepted_answers", "stages", "sources")
        missing = [field for field in required if field not in case]
        if missing:
            errors.append(f"{label}: missing {', '.join(missing)}")
            continue
        cid = case["id"]
        if cid in ids:
            errors.append(f"Duplicate ID {cid!r} (rows {ids[cid]} and {index + 1})")
        ids[cid] = index + 1
        qnorm = normalized(case["question"])
        if qnorm in questions:
            other = questions[qnorm]
            errors.append(f"Duplicate normalized question: {other!r} and {cid!r}")
        questions[qnorm] = cid
        if case["category"] not in CATEGORIES:
            errors.append(f"{cid}: unknown category {case['category']!r}")
        if case["split"] != "development":
            errors.append(f"{cid}: keep split='development'; analysis assigns family splits")
        family_splits[case["family"]].add(case["split"])
        family_categories[case["family"]][case["category"]] += 1
        category_counts[case["category"]] += 1
        split_counts[case["split"]] += 1
        if not isinstance(case["accepted_answers"], list) or not case["accepted_answers"]:
            errors.append(f"{cid}: accepted_answers must be a nonempty list")
        if not isinstance(case["stages"], list) or len(case["stages"]) != 3:
            errors.append(f"{cid}: expected exactly three evidence stages")
        if not isinstance(case["sources"], list) or not case["sources"]:
            errors.append(f"{cid}: sources must be a nonempty list")
        if case.get("review_status") not in {
            "verified", "existing_development_case", "prior_development_only",
            "wikidata_single_value_and_reference_checked",
            "source_linked_benchmark_item_not_independently_verified",
        }:
            warnings.append(f"{cid}: review_status is not verified")
        if not case.get("source_rationale"):
            warnings.append(f"{cid}: missing source_rationale")

    mixed = [family for family, splits in family_splits.items() if len(splits) > 1]
    if mixed:
        errors.append("Families cross splits: " + ", ".join(sorted(mixed)))

    real_splits = {split for split in family_splits.values() for split in split}
    # Development-only banks are valid inputs for analysis, where family splits
    # are assigned later. Explicit split banks must contain all three subsets.
    if real_splits != {"development"} and real_splits != SPLITS:
        warnings.append("Explicit bank should contain all of train, validation, test")

    n_families = len(family_splits)
    if len(cases) < 300:
        warnings.append(f"{len(cases)} questions: target is at least 300")
    if n_families < 30:
        warnings.append(f"{n_families} families: target is at least 30")
    for category in sorted(CATEGORIES):
        if not category_counts[category]:
            warnings.append(f"No items in category {category}")
    if max(category_counts.values(), default=0) > 0.4 * len(cases):
        warnings.append("One category exceeds 40% of the bank")

    report = {
        "path": str(Path(path)),
        "n_questions": len(cases),
        "n_families": n_families,
        "by_category": dict(sorted(category_counts.items())),
        "by_split": dict(sorted(split_counts.items())),
        "family_sizes": dict(sorted(Counter(case.get("family") for case in cases).items())),
        "family_category_counts": {
            family: dict(sorted(counts.items()))
            for family, counts in sorted(family_categories.items())
        },
        "errors": errors,
        "warnings": warnings,
    }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bank", type=Path, help="BOY JSON bank to audit")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    args = parser.parse_args()
    report = audit(args.bank)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"Questions: {report['n_questions']}  Families: {report['n_families']}")
        print("Categories:", ", ".join(
            f"{name}={count}" for name, count in report["by_category"].items()))
        print("Splits:", ", ".join(
            f"{name}={count}" for name, count in report["by_split"].items()))
        for message in report["warnings"]:
            print(f"WARNING: {message}")
        for message in report["errors"]:
            print(f"ERROR: {message}")
    raise SystemExit(1 if report["errors"] else 0)


if __name__ == "__main__":
    main()
