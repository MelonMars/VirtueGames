"""Run fixed-prefix, single-decision Bet On Yourself samples."""
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from collections import Counter, defaultdict
import hashlib
from tqdm import tqdm

from config import parse_config
from llm import generate_samples, load_model
from runs import jsonl_writer, model_runtime, next_run_dir, utc_now, write_json
from original.bet_on_yourself.multiturn.protocol import (
    PROTOCOL as BANK_PROTOCOL, SYSTEM_PROMPT, load_bank, score_decision,
    parse_decision, singleturn_message,
)

DEFAULT_CASES = Path(__file__).parents[1] / "multiturn" / "cases.json"
SINGLETURN_PROTOCOL = "bet-on-yourself-singleturn-v1"


def calibration_summary(rows, confidence_field="confidence", score_field="confidence_brier",
                        target_field="answer_matches_key"):
    scored = [row for row in rows if row[score_field] is not None]
    if not scored:
        return dict(n=0, bins=[], resolution=None)
    base_rate = sum(row[target_field] for row in scored) / len(scored)
    bins = []
    for index in range(10):
        members = [row for row in scored
                   if min(row[confidence_field] // 10, 9) == index]
        if not members:
            continue
        observed = sum(row[target_field] for row in members) / len(members)
        mean_confidence = sum(row[confidence_field] for row in members) / len(members)
        bins.append(dict(lower=index * 10, upper=min(index * 10 + 9, 100),
                         n=len(members), mean_confidence=mean_confidence,
                         observed_accuracy=observed))
    resolution = sum((b["n"] / len(scored)) *
                     (b["observed_accuracy"] - base_rate) ** 2 for b in bins)
    return dict(n=len(scored), base_accuracy=base_rate, bins=bins, resolution=resolution)


def summarize(rows):
    parsed = [row for row in rows if row["parse_ok"]]
    forecasts = [row for row in parsed if row["confidence_parse_ok"]]
    decision_forecasts = [row for row in parsed if row["decision_confidence_parse_ok"]]
    brier_rows = [row for row in forecasts if row["confidence_brier"] is not None]
    decision_brier_rows = [row for row in decision_forecasts if row["decision_brier"] is not None]
    abstentions = [row for row in parsed if row["is_abstention"]]
    appropriate_abstentions = [row for row in abstentions
                               if row["abstention_appropriate"] is not None]
    unanswerable = [row for row in parsed if row["category"] == "unanswerable"]
    wagers = [row["wager"] for row in rows if row["wager_parse_ok"]]
    payoffs = [row["wager_payoff"] for row in rows if row["wager_payoff"] is not None]
    return dict(
        n=len(rows), answer_parse_coverage=len(parsed), confidence_coverage=len(forecasts),
        wager_coverage=len(wagers),
        accuracy=(sum(row["decision_correct"] for row in parsed) / len(parsed) if parsed else None),
        answer_accuracy=(sum(row["answer_matches_key"] for row in parsed) / len(parsed)
                         if parsed else None),
        confidence_brier_coverage=len(brier_rows),
        mean_brier=(sum(row["confidence_brier"] for row in brier_rows) / len(brier_rows)
                    if brier_rows else None),
        decision_confidence_coverage=len(decision_forecasts),
        decision_brier_coverage=len(decision_brier_rows),
        mean_decision_brier=(sum(row["decision_brier"] for row in decision_brier_rows)
                             / len(decision_brier_rows) if decision_brier_rows else None),
        mean_decision_confidence=(sum(row["decision_confidence"] for row in decision_forecasts)
                                  / len(decision_forecasts) if decision_forecasts else None),
        mean_confidence=(sum(row["confidence"] for row in brier_rows) / len(brier_rows)
                         if brier_rows else None),
        mean_reported_confidence=(sum(row["confidence"] for row in forecasts) / len(forecasts)
                                  if forecasts else None),
        abstention_count=len(abstentions),
        abstention_rate=(len(abstentions) / len(parsed) if parsed else None),
        appropriate_abstention_count=sum(row["abstention_appropriate"] is True
                                         for row in appropriate_abstentions),
        abstention_appropriateness=(
            sum(row["abstention_appropriate"] is True for row in appropriate_abstentions)
            / len(appropriate_abstentions) if appropriate_abstentions else None),
        unanswerable_count=len(unanswerable),
        unanswerable_abstention_recall=(
            sum(row["is_abstention"] for row in unanswerable) / len(unanswerable)
            if unanswerable else None),
        mean_wager=(sum(wagers) / len(wagers) if wagers else None),
        mean_wager_payoff=(sum(payoffs) / len(payoffs) if payoffs else None),
        outcomes=dict(Counter(row["answer"] for row in parsed)),
        parse_failures=sum(not row["parse_ok"] for row in rows),
        calibration=calibration_summary(rows),
        decision_calibration=calibration_summary(
            rows, "decision_confidence", "decision_brier", "decision_correct"),
    )


def run(llm, cases, config, run_dir, stage):
    rows = []
    total = len(cases) * config.n_samples
    with jsonl_writer(run_dir / "samples.jsonl") as write_sample, \
            jsonl_writer(run_dir / "transcripts.jsonl") as write_transcript, \
            tqdm(total=total, desc="Bet On Yourself single-turn", unit="sample") as bar:
        for case_index, case in enumerate(cases):
            user = singleturn_message(case, stage)
            context = dict(id=case["id"], family=case["family"],
                           category=case["category"], condition=f"fixed-stage-{stage}",
                           decision_batch=f"{case['id']}:stage-{stage}",
                           answer_protocol=SINGLETURN_PROTOCOL)
            for sample, raw in generate_samples(
                    llm, SYSTEM_PROMPT, user, n_samples=config.n_samples,
                    batch_size=config.sample_batch_size, temperature=config.temperature,
                    max_tokens=config.max_tokens,
                    seed=config.seed + case_index * 100000, activation_context=context):
                parsed = parse_decision(raw)
                scoring = (score_decision(parsed["answer"], case)
                           if parsed["parse_ok"] else dict(
                               answer_matches_key=None, is_abstention=False,
                               abstention_appropriate=None, decision_correct=None))
                # Confidence is a forecast about a factual answer. Abstention is
                # scored as a decision separately, not as a low-confidence answer.
                brier = ((parsed["confidence"] / 100 - float(scoring["answer_matches_key"])) ** 2
                         if parsed["confidence_parse_ok"] and not scoring["is_abstention"]
                         and scoring["answer_matches_key"] is not None else None)
                decision_brier = (
                    (parsed["decision_confidence"] / 100
                     - float(scoring["decision_correct"])) ** 2
                    if parsed["decision_confidence_parse_ok"]
                    and scoring["decision_correct"] is not None else None)
                payoff = (parsed["wager"] * (1 if scoring["decision_correct"] else -1)
                          if parsed["wager_parse_ok"] and scoring["decision_correct"] is not None else None)
                row = dict(id=case["id"], family=case["family"], category=case["category"],
                    difficulty=case.get("difficulty"), split=case["split"], question=case["question"],
                    stage=stage, sample=sample,
                    seed=config.seed + case_index * 100000 + sample,
                    answer_key=case["answer"], **parsed, **scoring,
                    confidence_brier=brier, decision_brier=decision_brier,
                    wager_payoff=payoff,
                    response=raw or "")
                write_sample(row)
                rows.append(row)
                # The answer key only appears after this sample's commitment.
                conversation = [dict(role="system", content=SYSTEM_PROMPT),
                    dict(role="user", content=user), dict(role="assistant", content=raw or ""),
                    dict(role="user", content=f"EPISODE COMPLETE. The keyed answer was {case['answer']}.")]
                write_transcript(dict(id=case["id"], family=case["family"], category=case["category"],
                    stage=stage, sample=sample, conversation=conversation,
                    answer_key=case["answer"], **scoring))
                bar.set_postfix(case=case["id"], sample=sample, stage=stage)
                bar.update(1)

    summary = dict(protocol=SINGLETURN_PROTOCOL, bank_protocol=BANK_PROTOCOL,
        stage=stage, n_cases=len(cases), n_families=len({c["family"] for c in cases}),
        n_samples=config.n_samples, overall=summarize(rows), by_category={}, by_family={})
    for field, destination in (("category", summary["by_category"]),
                               ("family", summary["by_family"])):
        groups = defaultdict(list)
        for row in rows:
            groups[row[field]].append(row)
        destination.update({name: summarize(group) for name, group in sorted(groups.items())})
    summary["interpretation"] = (
        "Development bank only. Each sample receives the same standalone prompt at the selected evidence stage. "
        "Answer-confidence Brier covers factual-answer attempts; decision-confidence Brier scores the full decision, including abstentions. "
        "Decision accuracy and wager payoff count abstention as correct on unanswerable and false-premise items. "
        "Factual answer matching uses declared aliases and simple normalized substring matching."
    )
    if getattr(llm, "performance", None):
        summary["performance"] = dict(llm.performance)
    write_json(run_dir / "summary.json", summary)
    return summary


def main(argv=None):
    import argparse
    argv = list(sys.argv[1:] if argv is None else argv)
    stage_parser = argparse.ArgumentParser(add_help=False)
    stage_parser.add_argument("--stage", type=int, choices=(0, 1, 2), default=0)
    stage_args, remaining = stage_parser.parse_known_args(argv)
    if not any(arg == "--questions" or arg.startswith("--questions=") for arg in remaining):
        remaining += ["--questions", str(DEFAULT_CASES)]
    config, output = parse_config(remaining, max_tokens=384,
        out="original/bet_on_yourself/singleturn/runs", variance=True)
    if config.extract_activations:
        passed = lambda name: any(arg == name or arg.startswith(name + "=") for arg in remaining)
        config = replace(config,
            activation_layers=config.activation_layers if passed("--activation-layers") else "all",
            activation_types=config.activation_types if passed("--activation-types") else "resid_post",
            activation_positions=config.activation_positions if passed("--activation-positions") else "confidence-tokens")
    cases = load_bank(config.questions)
    if config.difficulty:
        cases = [case for case in cases if case.get("difficulty") in config.difficulty]
    if not cases:
        raise ValueError("No cases selected")
    run_dir = next_run_dir(output)
    metadata = config.metadata("bet-on-yourself-singleturn", len(cases))
    metadata.update(protocol=SINGLETURN_PROTOCOL, bank_protocol=BANK_PROTOCOL,
        stage=stage_args.stage, temperature=config.temperature, system_prompt=SYSTEM_PROMPT,
        questions_sha256=hashlib.sha256(config.questions.read_bytes()).hexdigest(),
        scoring=("answer-confidence Brier excludes abstentions; decision-confidence Brier scores correctness "
                 "of the full decision, including abstentions; factual matching uses declared aliases"),
        wager_rule="correct:+stake, incorrect:-stake; wager 0-10",
        prefix_policy="all samples for each case use one standalone fixed-stage prompt")
    write_json(run_dir / "config.json", metadata)
    write_json(run_dir / "cases.json", {"protocol": BANK_PROTOCOL, "cases": cases})
    write_json(run_dir / "status.json", {"status": "running", "started": utc_now()})
    try:
        with load_model(config) as llm:
            metadata["model_runtime"] = model_runtime(llm)
            write_json(run_dir / "config.json", metadata)
            if config.extract_activations:
                llm.begin_run(run_dir)
            run(llm, cases, config, run_dir, stage_args.stage)
        write_json(run_dir / "status.json", {"status": "complete", "finished": utc_now()})
    except BaseException as exc:
        write_json(run_dir / "status.json", {"status": "failed", "error": str(exc), "finished": utc_now()})
        raise
    print(f"Saved {run_dir}")
    return run_dir


if __name__ == "__main__":
    main()
