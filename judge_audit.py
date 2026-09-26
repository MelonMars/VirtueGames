"""Export blinded audit examples and compare judge predictions with human labels."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random

from runs import read_jsonl, jsonl_writer, write_json
from scoring import answer_diagnostic


def export(kind, root, out, random_n, challenge_n, seed):
    if min(random_n, challenge_n) < 0:
        raise ValueError("Sample counts must be nonnegative")
    rows = read_jsonl(root / "samples.jsonl")
    if kind == "candor":
        bank = {q["id"]: q for q in json.loads((root / "questions.json").read_text())["questions"]}
    else:
        bank = {q["id"]: q for q in read_jsonl(root / "variance.jsonl")}
    rng = random.Random(seed)
    random_indices = set(rng.sample(range(len(rows)), min(random_n, len(rows))))
    difficult = []
    for i, row in enumerate(rows):
        text = row.get("response", row.get("treatment_raw", ""))
        j = row.get("judgment") or {}
        if (row.get("label", row.get("outcome")) in ("mixed", "unclear", "omission_other", "judge_error", "caved_weasel", "contradictory", "review_pending")
                or j.get("names_flaw") and j.get("approves")
                or kind == "integrity" and not answer_diagnostic(text)["format_valid"]):
            if i not in random_indices:
                difficult.append(i)
    challenge_indices = set(rng.sample(difficult, min(challenge_n, len(difficult))))
    selected = list(random_indices | challenge_indices)
    rng.shuffle(selected)
    out.mkdir(parents=True, exist_ok=False)
    with jsonl_writer(out / "human_labels.jsonl") as write, jsonl_writer(out / "predictions.jsonl") as prediction:
        for i in selected:
            row = rows[i]; q = bank[row["id"]]
            text = row.get("response", row.get("treatment_raw", ""))
            uid = hashlib.sha256(json.dumps([row["id"], row.get("condition", "treatment"), row["sample"], text]).encode()).hexdigest()
            item = dict(uid=uid, kind=kind, id=row["id"], sample=row["sample"],
                        stratum="random" if i in random_indices else "challenge", response=text,
                        human_label=None, notes="")
            if kind == "candor":
                item.update(work=q["work"], flaw=q["flaw"], rubric=q["rubric"])
            else:
                item.update(question=q["control_user"], key=q["key"])
            write(item)
            prediction(dict(uid=uid, label=row.get("label", row.get("outcome"))))
    write_json(out / "metadata.json", dict(source=str(root.resolve()), seed=seed,
        random_n=len(random_indices), challenge_n=len(challenge_indices),
        note="Label human_labels.jsonl before inspecting predictions.jsonl. Challenge errors are not a population error estimate."))


def evaluate(labels, predictions):
    gold = read_jsonl(labels); predictions = read_jsonl(predictions)
    if len({r["uid"] for r in gold}) != len(gold) or len({r["uid"] for r in predictions}) != len(predictions):
        raise ValueError("Duplicate audit IDs")
    predicted = {r["uid"]: r["label"] for r in predictions}
    valid = {"candor": {"candid", "sycophantic", "mixed", "omission_other", "unclear"},
             "integrity": {"held", "caved_flip", "contradictory", "unclear"}}
    groups = defaultdict(list)
    for row in gold:
        if row.get("human_label") not in valid[row["kind"]]:
            raise ValueError(f"Missing or invalid human label: {row['uid']}")
        if row["uid"] not in predicted:
            raise ValueError("Missing prediction")
        pair = (row["human_label"], predicted[row["uid"]])
        groups[row["stratum"]].append(pair)
        groups["label:" + row["human_label"]].append(pair)
    return {name: dict(n=len(pairs), disagreement_rate=sum(a != b for a,b in pairs)/len(pairs),
                      confusion=dict(Counter(f"{a} -> {b}" for a,b in pairs))) for name,pairs in groups.items()}


def judge_benchmark(kind, path, out, model_name, device, revision="main"):
    from config import RunConfig
    from llm import load_model
    from rescore_runs import score_integrity
    from candor.score_candor import judge_review
    cfg = RunConfig(model=model_name, backend="transformers", device=device, thinking="off", revision=revision,
                    questions=path, max_tokens=1024, n_ctx=8192)
    if out.exists():
        raise ValueError("Prediction output already exists")
    with load_model(cfg) as model, jsonl_writer(out) as write:
        from runs import model_runtime
        runtime = model_runtime(model)
        for row in read_jsonl(path):
            if row["kind"] != kind:
                raise ValueError("Benchmark kind mismatch")
            # Do not pass the human label, notes, or answer key to the judge.
            result = (judge_review(model, row, row["response"], cfg) if kind == "candor" else
                      score_integrity(model, cfg, row["question"], row["response"], row["key"]))
            write(dict(uid=row["uid"], label=result.get("label", result.get("outcome")), assessment=result,
                       judge_model=model_name, judge_revision=revision, runtime=runtime))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    e = sub.add_parser("export")
    e.add_argument("kind", choices=("candor", "integrity")); e.add_argument("run", type=Path)
    e.add_argument("--out", type=Path, required=True)
    e.add_argument("--random-n", type=int, default=40); e.add_argument("--challenge-n", type=int, default=20)
    e.add_argument("--seed", type=int, default=42)
    v = sub.add_parser("evaluate")
    v.add_argument("labels", type=Path); v.add_argument("predictions", type=Path)
    j = sub.add_parser("judge")
    j.add_argument("kind", choices=("candor", "integrity")); j.add_argument("benchmark", type=Path)
    j.add_argument("--out", type=Path, required=True); j.add_argument("--judge-model", required=True)
    j.add_argument("--judge-revision", default="main")
    j.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    a = p.parse_args()
    if a.command == "export":
        export(a.kind, a.run, a.out, a.random_n, a.challenge_n, a.seed)
    elif a.command == "evaluate":
        print(json.dumps(evaluate(a.labels, a.predictions), indent=2))
    else:
        judge_benchmark(a.kind, a.benchmark, a.out, a.judge_model, a.device, a.judge_revision)


if __name__ == "__main__":
    main()
