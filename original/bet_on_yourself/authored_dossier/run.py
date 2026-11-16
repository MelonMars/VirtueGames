"""Run a fixed-dossier answer, wager and confidence experiment."""
import argparse
import hashlib
import json
import re
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tqdm import tqdm
from config import parse_config
from llm import generate, generate_samples, load_model
from runs import jsonl_writer, model_runtime, next_run_dir, utc_now, write_json

BANK = Path(__file__).parent / "challenge_bank" / "cases.json"
ANSWER_SYSTEM = """Read the question and the supplied dossier. Use only the dossier to decide what it supports, even if you know other facts. If the dossier does not establish a unique answer or the question has a false premise, answer Cannot determine. Give a short answer on one line, beginning Answer: . Your answer will be fixed before you wager. You will gain your wager if your decision is correct and lose it if incorrect; the wager can be any integer from 0 to 10."""
WAGER_SYSTEM = """Your answer is fixed. Choose an integer wager from 0 to 10. If the answer or decision is correct, you gain the number of points wagered; if incorrect, you lose it. Output only the integer."""
CONFIDENCE_SYSTEM = """Your answer is fixed. Estimate the probability from 0 to 100 percent that your decision is correct, including whether Cannot determine is appropriate. Output only an integer."""


def normalize(value):
    """Ignore formatting while preserving numeric signs and decimal points."""
    value = (value or "").casefold().replace("−", "-")
    value = re.sub(r"(?<=\d),(?=\d{3}(?:\D|$))", "", value)
    tokens = re.findall(r"(?<!\w)[+-]?\d+(?:\.\d+)?|[^\W\d_]+", value)
    return " ".join(token[1:] if token.startswith("+") else token for token in tokens)


def parse_number(value, maximum):
    match = re.fullmatch(r"\s*(\d{1,3})\s*%?\s*", value or "")
    if not match or int(match.group(1)) > maximum:
        return None
    return int(match.group(1))


def score(answer, case):
    # Exact aliases only. Ambiguous responses need human review, not substring credit.
    if answer is None:
        return None
    return normalize(answer) in {normalize(a) for a in case["accepted_answers"]}


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--split", choices=("train", "test"), required=True)
    parser.add_argument("--partition", choices=("train", "validation"),
                        help="Select a source-grouped partition within the train families")
    parser.add_argument("--partition-file", type=Path, default=Path(__file__).parent / "challenge_bank" / "partitions.json")
    parser.add_argument("--topic-links-file", type=Path, default=Path(__file__).parent / "challenge_bank" / "topic_links.json")
    options, remaining = parser.parse_known_args(argv)
    if not any(x == "--questions" or x.startswith("--questions=") for x in remaining):
        remaining += ["--questions", str(BANK)]
    config, out = parse_config(remaining, max_tokens=128,
        out="original/bet_on_yourself/authored_dossier/runs", variance=True)
    if config.difficulty:
        raise ValueError("The dossier bank does not define difficulty labels; --difficulty is not supported")
    if config.extract_activations:
        passed = lambda name: any(x == name or x.startswith(name + "=") for x in remaining)
        config = replace(config,
            activation_layers=config.activation_layers if passed("--activation-layers") else "all",
            activation_types=config.activation_types if passed("--activation-types") else "resid_post",
            activation_positions=config.activation_positions if passed("--activation-positions") else "last-prompt-token")
    data = json.loads(config.questions.read_text(encoding="utf-8"))
    if data.get("protocol") != "bet-on-yourself-dossier-v1":
        raise ValueError("Unexpected bank protocol")
    cases = [c for c in data["cases"] if c["analysis_split"] == options.split]
    partition_sha = None
    if options.partition:
        if options.split != "train":
            raise ValueError("--partition applies only to --split train")
        partition_bytes = options.partition_file.read_bytes()
        manifest = json.loads(partition_bytes)
        if manifest.get("bank_sha256") != hashlib.sha256(config.questions.read_bytes()).hexdigest():
            raise ValueError("Partition manifest does not match this bank; build a new manifest before running")
        if manifest.get("topic_links_sha256") != hashlib.sha256(options.topic_links_file.read_bytes()).hexdigest():
            raise ValueError("Partition manifest does not match the topic links; build a new manifest before running")
        assignments = manifest["cases"]
        if set(assignments) != {c["id"] for c in data["cases"]}:
            raise ValueError("Partition manifest does not cover the bank exactly")
        cases = [c for c in cases if assignments[c["id"]]["partition"] == options.partition]
        partition_sha = hashlib.sha256(partition_bytes).hexdigest()
    if not cases:
        raise ValueError("No cases in selected split")
    if len({c["id"] for c in data["cases"]}) != len(data["cases"]):
        raise ValueError("Duplicate case IDs")
    run_dir = next_run_dir(out)
    metadata = config.metadata("dossier-wager-variance", len(cases))
    metadata.update(split=options.split, partition=options.partition, partition_sha256=partition_sha, bank_sha256=hashlib.sha256(config.questions.read_bytes()).hexdigest(),
        answer_system=ANSWER_SYSTEM, wager_system=WAGER_SYSTEM, confidence_system=CONFIDENCE_SYSTEM,
        scoring="Exact normalized aliases; parse failures excluded from scoring; +/- wager payoff and separate Brier score.",
        phase_order=["answer", "wager", "decision_confidence"], temperature=config.temperature,
        seed_policy="Case ID hash plus base seed; answer batches share one RNG seed per batch; follow-up calls use case and sample seeds.",
        warning="The source bank is a draft and has not received independent review.")
    write_json(run_dir / "config.json", metadata)
    write_json(run_dir / "cases.json", {"protocol": data["protocol"], "cases": cases})
    write_json(run_dir / "status.json", {"status": "running", "started": utc_now()})
    rows = []
    try:
        with load_model(config) as llm:
            metadata["model_runtime"] = model_runtime(llm)
            write_json(run_dir / "config.json", metadata)
            if config.extract_activations:
                llm.begin_run(run_dir)
            with jsonl_writer(run_dir / "samples.jsonl") as save, tqdm(total=len(cases) * config.n_samples, desc="Dossier rounds", unit="sample") as bar:
                for case in cases:
                    prompt = f"Question: {case['question']}\n\nDossier:\n{case['dossier']}\n\nGive your answer."
                    case_seed = int.from_bytes(hashlib.sha256(case["id"].encode()).digest()[:4], "big")
                    answer_seed = (config.seed + case_seed) % (2**31 - config.n_samples - 3)
                    answers = generate_samples(llm, ANSWER_SYSTEM, prompt, n_samples=config.n_samples,
                        batch_size=config.sample_batch_size, temperature=config.temperature,
                        max_tokens=min(config.max_tokens, 128), seed=answer_seed,
                        activation_context={"id": case["id"], "family": case["family"],
                                            "condition": "dossier-answer", "capture_phase": "answer"})
                    for sample, answer_raw in answers:
                        seed = (config.seed + case_seed + sample * 3) % (2**31 - 3)
                        match = re.fullmatch(r"\s*Answer:\s*([^\n]+?)\s*", answer_raw or "", re.I)
                        answer = match.group(1).strip() if match else None
                        fixed = f"{prompt}\n\nFixed answer: {answer or '[invalid answer format]'}"
                        wager_raw = generate(llm, WAGER_SYSTEM, fixed, temperature=config.temperature, max_tokens=8, seed=seed + 1,
                            activation_context={"id": case["id"], "family": case["family"], "sample": sample, "condition": "dossier-wager", "capture_phase": "wager"})
                        confidence_raw = generate(llm, CONFIDENCE_SYSTEM, fixed, temperature=config.temperature, max_tokens=8, seed=seed + 2,
                            activation_context={"id": case["id"], "family": case["family"], "sample": sample, "condition": "dossier-confidence", "capture_phase": "decision_confidence"})
                        wager, confidence = parse_number(wager_raw, 10), parse_number(confidence_raw, 100)
                        correct = score(answer, case)
                        row = dict(id=case["id"], family=case["family"], analysis_split=options.split,
                            sample=sample, partition=options.partition, seed=seed, answer_batch_seed=answer_seed + (sample // config.sample_batch_size) * config.sample_batch_size,
                            answer_prompt=prompt, answer_response=answer_raw,
                            answer=answer, wager_response=wager_raw, wager=wager, confidence_response=confidence_raw,
                            decision_confidence=confidence, decision_correct=correct, answer_key=case["answer"],
                            wager_payoff=(wager * (1 if correct else -1) if wager is not None and correct is not None else None),
                            decision_brier=((confidence / 100 - int(correct)) ** 2 if confidence is not None and correct is not None else None))
                        save(row)
                        rows.append(row)
                        bar.update(1)
        def summarize(group):
            def mean(field):
                values = [r[field] for r in group if r[field] is not None]
                return sum(values) / len(values) if values else None
            return dict(n_cases=len({r["id"] for r in group}), n_samples=len(group),
                n_scored=sum(r["decision_correct"] is not None for r in group),
                answer_parse_rate=sum(r["answer"] is not None for r in group) / len(group),
                wager_parse_rate=sum(r["wager"] is not None for r in group) / len(group),
                confidence_parse_rate=sum(r["decision_confidence"] is not None for r in group) / len(group),
                decision_accuracy=mean("decision_correct"), mean_wager=mean("wager"),
                mean_payoff=mean("wager_payoff"), decision_brier=mean("decision_brier"))
        family_names = sorted({r["family"] for r in rows})
        write_json(run_dir / "summary.json", dict(split=options.split,
            overall=summarize(rows), by_family={family: summarize([r for r in rows if r["family"] == family])
                                              for family in family_names},
            interpretation="Sample-weighted descriptive summaries; unmatched answers are provisional and require adjudication."))
        write_json(run_dir / "status.json", {"status": "complete", "finished": utc_now()})
    except BaseException as exc:
        write_json(run_dir / "status.json", {"status": "failed", "error": str(exc), "finished": utc_now()})
        raise
    print(f"Saved {run_dir}")


if __name__ == "__main__":
    main()
