"""Create and validate outcome-blind train/development/test family manifests."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
import re


def inferred_family(qid):
    # Only a draft aid: related hard_* items still require manual consolidation.
    return re.sub(r"_\d+$", "", qid) if qid.startswith("expanded_") else qid


def load_manifest(path, ids):
    data = json.loads(Path(path).read_text())
    rows = data["items"]
    if set(rows) != set(ids):
        raise ValueError("Split manifest IDs must exactly match the question population")
    groups = defaultdict(set)
    for qid, row in rows.items():
        if row.get("split") not in ("train", "validation", "test"):
            raise ValueError(f"Invalid split: {qid}")
        if not isinstance(row.get("family"), str) or not row["family"].strip():
            raise ValueError(f"Missing explicit family: {qid}")
        groups[row["family"]].add(row["split"])
    if any(len(splits) != 1 for splits in groups.values()):
        raise ValueError("Scenario family crosses split boundaries")
    if {row["split"] for row in rows.values()} != {"train", "validation", "test"}:
        raise ValueError("All three partitions must contain families")
    return rows


def make_manifest(questions, families, seed=42, validation=.2, test=.2):
    ids = [q["id"] for q in questions]
    if len(set(ids)) != len(ids) or set(families) != set(ids):
        raise ValueError("Family map must cover every unique question ID exactly")
    if any(not isinstance(f, str) or not f.strip() for f in families.values()):
        raise ValueError("Every item needs a nonempty family")
    if not 0 < validation < 1 or not 0 < test < 1 or validation + test >= 1:
        raise ValueError("Invalid partition fractions")
    groups = sorted(set(families.values()))
    if len(groups) < 3:
        raise ValueError("At least three families are required")
    random.Random(seed).shuffle(groups)
    nv, nt = max(1, round(len(groups)*validation)), max(1, round(len(groups)*test))
    if nv + nt >= len(groups):
        raise ValueError("Fractions leave no training families; use more families or smaller fractions")
    assignment = {f: "validation" if i < nv else "test" if i < nv+nt else "train"
                  for i, f in enumerate(groups)}
    return dict(schema_version=1, seed=seed, validation_fraction=validation, test_fraction=test,
                items={qid: dict(family=families[qid], split=assignment[families[qid]]) for qid in ids})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("questions", type=Path)
    p.add_argument("--family-map", type=Path, help="Reviewed JSON mapping question ID to family")
    p.add_argument("--draft-families", action="store_true", help="Write a draft map for manual review, not a split")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--validation-fraction", type=float, default=.2)
    p.add_argument("--test-fraction", type=float, default=.2)
    a = p.parse_args()
    bank = json.loads(a.questions.read_text())["questions"]
    if a.draft_families:
        result = {q["id"]: q.get("family_id", inferred_family(q["id"])) for q in bank}
    else:
        if not a.family_map:
            p.error("Provide a reviewed --family-map, or use --draft-families first")
        result = make_manifest(bank, json.loads(a.family_map.read_text()), a.seed,
                               a.validation_fraction, a.test_fraction)
        result["questions_sha256"] = hashlib.sha256(a.questions.read_bytes()).hexdigest()
    with a.out.open("x") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
