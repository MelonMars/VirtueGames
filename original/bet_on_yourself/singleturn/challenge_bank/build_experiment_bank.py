"""Build the runnable, family-split BOY calibration pilot bank."""
import argparse
import json
import re
from collections import Counter
from datetime import date
from pathlib import Path


RELATION_SPLITS = {
    "author": "train",
    "director": "train",
    "sport": "train",
    "father": "train",
    "screenwriter": "validation",
    "country": "validation",
    "composer": "validation",
    "capital": "validation",
    "genre": "test",
    "place of birth": "test",
    "producer": "test",
    "occupation": "test",
    "religion": "test",
}
PROPERTY_NAMES = {
    "author": "author", "capital": "capital", "composer": "composer",
    "country": "country", "director": "director", "father": "father",
    "genre": "genre", "occupation": "occupation", "place of birth": "place of birth",
    "producer": "producer", "religion": "religion", "screenwriter": "screenwriter",
    "sport": "sport",
}


def normalized(text):
    return " ".join(re.findall(r"[a-z0-9]+", str(text or "").casefold()))


def safe_aliases(key, aliases):
    kept = [key]
    for alias in aliases:
        alias = str(alias).strip()
        clean = normalized(alias)
        if len(clean) >= 4 and clean not in {normalized(item) for item in kept}:
            kept.append(alias)
    return kept


def popqa_case(row, checked_date):
    relation = row["benchmark_category"].casefold()
    split = RELATION_SPLITS.get(relation)
    verification = row["wikidata_verification"]
    if (verification["status"] != "exact_match_single_value"
            or verification.get("reference_count", 0) < 1):
        return None
    if not split or not row.get("proposed_adapted_question"):
        return None
    key = row["proposed_key"].strip()
    if relation == "genre" and key.casefold() == "world war ii":
        return None
    item_id = row["candidate_id"].replace("candidate_popqa_", "")
    subject_title = row.get("source_page_title") or row.get("subj") or row["original_question"]
    relation_name = PROPERTY_NAMES[relation]
    question = row["proposed_adapted_question"].strip()
    source_uri = row.get("source_entity_uri")
    object_uri = row.get("source_object_uri")
    prop_uri = "https://www.wikidata.org/wiki/" + verification["property_id"]
    sources = [row["source_url"], f"https://www.wikidata.org/wiki/{verification['subject_id']}"]
    source_rationale = (
        f"On {checked_date}, Wikidata contained one active value for {verification['property_id']} "
        f"({relation_name}) on {verification['subject_id']}, and it matched {verification['expected_object_id']}. "
        "The linked entity page is retained for a reader-readable cross-check."
    )
    population = int(row.get("benchmark_subject_popularity") or 0)
    return {
        "id": f"boy_popqa_{item_id}",
        "family": f"popqa_relation_{re.sub(r'[^a-z0-9]+', '_', relation).strip('_')}",
        "split": "development",
        "analysis_split": split,
        "category": "hard_edge",
        "difficulty": "hard" if population <= 100 else "medium",
        "title": subject_title,
        "question": question,
        "answer": key,
        "accepted_answers": safe_aliases(key, row.get("proposed_aliases", [])),
        "stages": [
            "No external evidence or answer key is supplied. Answer from your existing knowledge; express uncertainty or abstain if needed.",
            "No additional evidence about the keyed fact is supplied at this stage.",
            f"A current reference record lists the {relation_name} of {subject_title} as {key}.",
        ],
        "sources": sources,
        "source_rationale": source_rationale,
        "source_benchmark": "PopQA",
        "source_item_id": row["candidate_id"],
        "original_question": row["original_question"],
        "question_adapted": normalized(question) != normalized(row["original_question"]),
        "review_status": "wikidata_single_value_and_reference_checked",
        "wikidata_subject_uri": source_uri,
        "wikidata_property_uri": prop_uri,
        "wikidata_object_uri": object_uri,
        "subject_page_views": population,
        "benchmark_note": "Long-tail entity question; popularity is the source dataset's page-view count and difficulty remains provisional.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wikidata-checked", type=Path, required=True)
    parser.add_argument("--original-bank", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--checked-date", default=date.today().isoformat())
    args = parser.parse_args()
    public_rows = [json.loads(line) for line in args.wikidata_checked.read_text(encoding="utf-8").splitlines()
                   if line.strip()]
    original = json.loads(args.original_bank.read_text(encoding="utf-8"))
    cases = []
    for case in original.get("cases", []):
        # Previously used question families are development-only in this
        # calibration pilot; none can enter its held-out validation/test sets.
        rationale = case.get("source_rationale") or (
            "Inherited from the previously used BOY development bank. The listed "
            "source URLs are retained, but this builder did not re-verify the "
            "answer key or source entailment."
        )
        cases.append({**case, "analysis_split": "train",
                      "source_benchmark": "original_boy_development",
                      "source_item_id": case["id"],
                      "question_adapted": False,
                      "source_rationale": rationale,
                      "review_status": "existing_development_case"})
    for row in public_rows:
        if row.get("benchmark") != "PopQA":
            continue
        case = popqa_case(row, args.checked_date)
        if case:
            cases.append(case)

    ids = [case["id"] for case in cases]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate case IDs after bank assembly")
    families = {}
    for case in cases:
        previous = families.setdefault(case["family"], case["analysis_split"])
        if previous != case["analysis_split"]:
            raise ValueError(f"Family {case['family']} crosses analysis splits")
    split_families = Counter(case["analysis_split"] for case in cases)
    summary = {
        "protocol": "bet-on-yourself-multiturn-v1",
        "bank_version": "bet-on-yourself-calibration-pilot-v1",
        "design": "Single-turn stage-0 calibration pilot. Previously used original bank is train-only; PopQA relation families are held out by split.",
        "n_cases": len(cases),
        "n_families": len(families),
        "by_benchmark": dict(Counter(case["source_benchmark"] for case in cases)),
        "by_analysis_split": {
            split: {
                "n_cases": sum(case["analysis_split"] == split for case in cases),
                "n_families": len({case["family"] for case in cases if case["analysis_split"] == split}),
            }
            for split in ("train", "validation", "test")
        },
        "split_policy": {
            "train": "All 70 previously used cases plus PopQA author, director, sport, and father relations.",
            "validation": "PopQA screenwriter, country, composer, and capital relations.",
            "test": "PopQA genre, place of birth, producer, occupation, and religion relations.",
            "note": "Held-out relation families test generalization to new answer relations, not broad unseen subject domains. Test items are public-benchmark-derived.",
        },
        "limitations": [
            "This is a pilot bank with 27 relation/topic families, not the planned 30+ family final bank.",
            "PopQA factual keys were checked for one current non-deprecated Wikidata value with at least one attached reference; this is not independent primary-source verification.",
            "Wikipedia page-text corroboration was rate-limited and is not complete; the five manually spot-checked rows are documented separately.",
            "Public benchmark-derived items may be familiar to Qwen/Qwen3-4B; report those results separately from new original items.",
            "All runs must use stage 0 for unaided knowledge; stages 1 and 2 are retained for protocol compatibility and are not part of this pilot.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"protocol": summary["protocol"], "summary": summary,
                                    "cases": cases}, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")
    args.out.with_suffix(args.out.suffix + ".report.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
