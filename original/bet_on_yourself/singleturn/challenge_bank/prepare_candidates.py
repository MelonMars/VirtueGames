"""Prepare auditable, human-review candidates from public QA benchmark files.

This intentionally does not emit a runnable BOY bank. Candidate answers,
aliases, categories, and family labels require review before use.
"""
import argparse
import csv
import datetime as dt
import hashlib
import json
import random
import re
from pathlib import Path


TRUTHFULQA_URL = "https://github.com/sylinrl/TruthfulQA/blob/main/data/v0/TruthfulQA.csv"
POPQA_URL = "https://github.com/AlexTMallen/adaptive-retrieval/blob/main/data/popQA.tsv"
WIKIDATA_PROPERTIES = {
    "author": "P50", "capital": "P36", "composer": "P86", "country": "P17",
    "director": "P57", "father": "P22", "genre": "P136", "occupation": "P106",
    "place of birth": "P19", "producer": "P162", "religion": "P140",
    "screenwriter": "P58", "sport": "P641",
}


def stable_id(source, question):
    digest = hashlib.sha256((source + "\0" + question).encode("utf-8")).hexdigest()[:12]
    return f"candidate_{source}_{digest}"


def split_semicolon(value):
    return [part.strip() for part in (value or "").split(";") if part.strip()]


def truthfulqa_review_flags(category, answer, source):
    flags = []
    if category.startswith("Indexical Error") or source.strip().casefold() == "indexical":
        flags.append("model_or_context_dependent")
    if category in {"Fiction", "Subjective", "Proverbs", "Myths and Fairytales",
                    "Advertising", "Conspiracies", "Mandela Effect", "Confusion: Other"}:
        flags.append("high_ambiguity_or_nonfactual_category")
    if answer.strip().casefold() in {"i have no comment", "no comment"}:
        flags.append("refusal_used_as_key")
    if source.strip().casefold() in {"tautology", "indexical"}:
        flags.append("no_external_fact_source")
    if re.search(r"\b(?:20[3-9]\d|[1-9]\d{3})\b", answer):
        flags.append("date_or_time_sensitive_answer")
    return flags


def rewrite_popqa_question(prop, subject, page_title):
    """Make the fixed PopQA relation templates read as ordinary questions."""
    templates = {
        "author": "Who wrote {subject}?",
        "capital": "What is the capital of {subject}?",
        "composer": "Who composed {subject}?",
        "country": "Which country is {subject} in?",
        "director": "Who directed {subject}?",
        "father": "Who was the father of {subject}?",
        "genre": "What genre is {subject}?",
        "occupation": "What occupation is {subject} known for?",
        "place of birth": "Where was {subject} born?",
        "producer": "Who produced {subject}?",
        "religion": "What religious affiliation is listed for {subject}?",
        "screenwriter": "Who wrote the screenplay for {subject}?",
        "sport": "Which sport is associated with {subject}?",
    }
    template = templates.get((prop or "").casefold())
    # Preserve a dataset-supplied page qualifier (for example, a specific
    # film, book, or TV episode) so common titles are less ambiguous.
    display_subject = page_title or subject
    return template.format(subject=display_subject) if template else ""


def truthfulqa_candidates(path, limit, seed):
    candidates = []
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            question = (row.get("Question") or "").strip()
            answer = (row.get("Best Answer") or "").strip()
            if not question or not answer:
                continue
            category = (row.get("Category") or "Uncategorized").strip()
            source = (row.get("Source") or "").strip()
            candidates.append({
                "candidate_id": stable_id("truthfulqa", question),
                "benchmark": "TruthfulQA",
                "benchmark_url": TRUTHFULQA_URL,
                "benchmark_category": category,
                "benchmark_type": (row.get("Type") or "").strip(),
                "original_question": question,
                "adapted_question": "",
                "proposed_key": answer,
                "proposed_aliases": split_semicolon(row.get("Correct Answers")),
                "benchmark_incorrect_answers": split_semicolon(row.get("Incorrect Answers")),
                "source_url": source,
                "review_flags": truthfulqa_review_flags(category, answer, source),
                "family": "",
                "proposed_stratum": "",
                "review_status": "unreviewed",
                "review_notes": "Verify current factual support, make a minimal adaptation if useful, assign family and stratum, then approve aliases.",
            })
    # Preserve spread across the benchmark's finer-grained categories.
    grouped = {}
    for candidate in candidates:
        grouped.setdefault(candidate["benchmark_category"], []).append(candidate)
    rng = random.Random(seed)
    for group in grouped.values():
        rng.shuffle(group)
    selected = []
    category_names = sorted(grouped)
    while len(selected) < min(limit, len(candidates)):
        advanced = False
        for category in category_names:
            if grouped[category] and len(selected) < limit:
                selected.append(grouped[category].pop())
                advanced = True
        if not advanced:
            break
    return selected


def popqa_candidates(path, limit, seed, max_subject_popularity):
    candidates = []
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        rows = csv.DictReader(stream, delimiter="\t")
        for row in rows:
            question = (row.get("question") or "").strip()
            possible = row.get("possible_answers") or ""
            try:
                aliases = json.loads(possible)
            except json.JSONDecodeError:
                aliases = []
            aliases = [str(value).strip() for value in aliases if str(value).strip()]
            if not question or not aliases:
                continue
            try:
                popularity = int(row.get("s_pop") or 0)
            except ValueError:
                popularity = 0
            if max_subject_popularity is not None and popularity > max_subject_popularity:
                continue
            title = (row.get("s_wiki_title") or row.get("subj") or "").strip()
            subject = (row.get("subj") or "").strip()
            wiki_url = "https://en.wikipedia.org/wiki/" + title.replace(" ", "_") if title else ""
            candidates.append({
                "candidate_id": stable_id("popqa", question),
                "benchmark": "PopQA",
                "benchmark_url": POPQA_URL,
                "benchmark_category": (row.get("prop") or "").strip(),
                "benchmark_subject_popularity": popularity,
                "original_question": question,
                "adapted_question": "",
                "proposed_adapted_question": rewrite_popqa_question(
                    row.get("prop"), subject, title.replace("_", " ")),
                "proposed_key": aliases[0],
                "proposed_aliases": aliases,
                "source_url": wiki_url,
                "source_page_title": title,
                "source_entity_uri": (row.get("s_uri") or "").strip(),
                "source_relation_index": (row.get("prop_id") or "").strip(),
                "wikidata_property_id": WIKIDATA_PROPERTIES.get((row.get("prop") or "").casefold()),
                "source_object_uri": (row.get("o_uri") or "").strip(),
                "review_flags": ["template_generated_question", "verify_subject_is_unambiguous"],
                "family": "",
                "proposed_stratum": "hard_edge",
                "review_status": "unreviewed",
                "review_notes": "Check the relation and answer against a current reliable source; add family and approve aliases. Adapt wording only if meaning stays identical.",
            })
    rng = random.Random(seed)
    rng.shuffle(candidates)
    return candidates[:limit] if limit else candidates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truthfulqa-csv", type=Path)
    parser.add_argument("--truthfulqa-limit", type=int, default=120)
    parser.add_argument("--popqa-tsv", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--popqa-limit", type=int, default=250)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-subject-popularity", type=int, default=1000,
                        help="Keep PopQA's long tail; use a negative value for no cap")
    parser.add_argument("--manifest", type=Path,
                        help="Manifest output (defaults to OUT.manifest.json)")
    args = parser.parse_args()
    if not args.truthfulqa_csv and not args.popqa_tsv:
        parser.error("provide at least one benchmark file")
    candidates = []
    if args.truthfulqa_csv:
        candidates.extend(truthfulqa_candidates(args.truthfulqa_csv,
                                                args.truthfulqa_limit, args.seed))
    if args.popqa_tsv:
        cap = None if args.max_subject_popularity < 0 else args.max_subject_popularity
        candidates.extend(popqa_candidates(args.popqa_tsv, args.popqa_limit, args.seed, cap))
    seen = set()
    unique = []
    for candidate in candidates:
        key = re.sub(r"\W+", " ", candidate["original_question"].casefold()).strip()
        if key not in seen:
            seen.add(key)
            unique.append(candidate)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as stream:
        for candidate in unique:
            stream.write(json.dumps(candidate, ensure_ascii=False) + "\n")
    manifest_path = args.manifest or args.out.with_suffix(args.out.suffix + ".manifest.json")
    inputs = {}
    for benchmark, input_path in (("TruthfulQA", args.truthfulqa_csv), ("PopQA", args.popqa_tsv)):
        if input_path:
            inputs[benchmark] = {
                "path": str(input_path),
                "sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
            }
    manifest = {
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "generator": "prepare_candidates.py",
        "seed": args.seed,
        "truthfulqa_limit": args.truthfulqa_limit,
        "popqa_limit": args.popqa_limit,
        "max_subject_popularity": args.max_subject_popularity,
        "source_inputs": inputs,
        "output": str(args.out),
        "output_sha256": hashlib.sha256(args.out.read_bytes()).hexdigest(),
        "n_candidates": len(unique),
        "by_benchmark": {name: sum(row["benchmark"] == name for row in unique)
                         for name in sorted({row["benchmark"] for row in unique})},
        "review_policy": "All rows are unreviewed candidates, not approved BOY cases.",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(unique)} unreviewed candidates to {args.out}")
    print(f"Wrote input/output provenance manifest to {manifest_path}")
    print("These JSONL rows are not a runnable BOY bank. Review and adapt them first.")


if __name__ == "__main__":
    main()
