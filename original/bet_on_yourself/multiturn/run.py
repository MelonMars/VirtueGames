"""Run Bet On Yourself episodes with exact-prefix sibling samples."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from collections import Counter, defaultdict
import hashlib
from tqdm import tqdm

from config import parse_config
from llm import load_model
from runs import jsonl_writer, model_runtime, next_run_dir, utc_now, write_json
from original.bet_on_yourself.multiturn.protocol import (
    PROTOCOL, SYSTEM_PROMPT, STAGES, load_bank, opening_message,
    update_message, parse_decision, score_decision,
)

DEFAULT_CASES = Path(__file__).with_name("cases.json")


def summarize_group(group):
    parsed = [row for row in group if row["parse_ok"]]
    forecasted = [row for row in parsed if row["confidence_parse_ok"]]
    decision_forecasts = [row for row in parsed if row["decision_confidence_parse_ok"]]
    brier_rows = [row for row in forecasted if row["confidence_brier"] is not None]
    decision_brier_rows = [row for row in decision_forecasts if row["decision_brier"] is not None]
    abstentions = [row for row in parsed if row["is_abstention"]]
    appropriate_abstentions = [row for row in abstentions
                               if row["abstention_appropriate"] is not None]
    return dict(
        n=len(group), parse_coverage=len(parsed), confidence_coverage=len(forecasted),
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
        mean_reported_confidence=(sum(row["confidence"] for row in forecasted) / len(forecasted)
                                  if forecasted else None),
        abstention_count=len(abstentions),
        abstention_rate=(len(abstentions) / len(parsed) if parsed else None),
        appropriate_abstention_count=sum(row["abstention_appropriate"] is True
                                         for row in appropriate_abstentions),
        abstention_appropriateness=(
            sum(row["abstention_appropriate"] is True for row in appropriate_abstentions)
            / len(appropriate_abstentions) if appropriate_abstentions else None),
    )


def decision_batch(llm, messages, config, *, seed, contexts):
    """Generate sibling decisions from the exact same conversation prefix."""
    if len(contexts) == 1 or not hasattr(llm, "create_chat_completions"):
        outputs = []
        for index, context in enumerate(contexts):
            kwargs = dict(messages=messages, temperature=config.temperature,
                          seed=seed + index, max_tokens=config.max_tokens)
            if getattr(llm, "activation_recorder", None) is not None:
                kwargs["activation_context"] = context
            response = llm.create_chat_completion(**kwargs)
            outputs.append(response["choices"][0]["message"].get("content") or "")
        return outputs
    return llm.create_chat_completions(messages=messages, temperature=config.temperature,
        seed=seed, max_tokens=config.max_tokens, activation_contexts=contexts)


def run(llm, cases, config, run_dir):
    rows = []
    total = len(cases) * len(STAGES) * config.n_samples
    with jsonl_writer(run_dir / "samples.jsonl") as write_sample, \
            jsonl_writer(run_dir / "transcripts.jsonl") as write_transcript, \
            tqdm(total=total, desc="Bet On Yourself episodes", unit="decision") as bar:
        for case_index, case in enumerate(cases):
            # One reference trajectory supplies the exact prior history for the
            # sibling samples at the next evidence stage.
            anchor = [dict(role="system", content=SYSTEM_PROMPT),
                      dict(role="user", content=opening_message(case))]
            anchor_result = None
            for stage in STAGES:
                if stage:
                    anchor.append(dict(role="user", content=update_message(case, stage)))
                prefix = [dict(message) for message in anchor]
                contexts = [dict(id=case["id"], family=case["family"],
                                 condition=f"round-{stage}", sample=i,
                                 decision_batch=f"{case['id']}:round-{stage}",
                                 answer_protocol=PROTOCOL)
                            for i in range(config.n_samples)]
                seed = config.seed + case_index * 100000 + stage * 1000
                raws = decision_batch(llm, prefix, config, seed=seed, contexts=contexts)
                if len(raws) != config.n_samples:
                    raise RuntimeError("Backend returned the wrong number of sibling samples")
                for sample, (raw, context) in enumerate(zip(raws, contexts)):
                    parsed = parse_decision(raw)
                    scoring = (score_decision(parsed["answer"], case)
                               if parsed["parse_ok"] else dict(
                                   answer_matches_key=None, is_abstention=False,
                                   abstention_appropriate=None, decision_correct=None))
                    # Confidence is a forecast about a factual answer. Abstention
                    # is scored as a decision separately, not as a low-confidence answer.
                    confidence_brier = (
                        (parsed["confidence"] / 100 - float(scoring["answer_matches_key"])) ** 2
                        if parsed["confidence_parse_ok"] and not scoring["is_abstention"]
                        and scoring["answer_matches_key"] is not None else None)
                    decision_brier = (
                        (parsed["decision_confidence"] / 100
                         - float(scoring["decision_correct"])) ** 2
                        if parsed["decision_confidence_parse_ok"]
                        and scoring["decision_correct"] is not None else None)
                    wager_payoff = (parsed["wager"] * (1 if scoring["decision_correct"] else -1)
                                    if parsed["wager_parse_ok"] and scoring["decision_correct"] is not None else None)
                    row = dict(id=case["id"], family=case["family"], split=case["split"],
                        category=case["category"], difficulty=case.get("difficulty"),
                        question=case["question"], stage=stage, sample=sample,
                        seed=seed + sample, answer_key=case["answer"], **parsed, **scoring,
                        confidence_brier=confidence_brier,
                        decision_brier=decision_brier,
                        wager_payoff=wager_payoff,
                        scored=bool(parsed["parse_ok"] and parsed["confidence_parse_ok"]),
                        response=raw or "")
                    write_sample(row)
                    rows.append(row)
                    if sample == 0:
                        anchor_result = row
                    bar.set_postfix(case=case["id"], round=stage, sample=sample)
                    bar.update(1)
                # Advance the single reference history with branch zero only.
                anchor.append(dict(role="assistant", content=raws[0] or ""))
                write_transcript(dict(id=case["id"], family=case["family"],
                    stage=stage, conversation=prefix + [dict(role="assistant", content=raws[0] or "")],
                    reference_sample=0, reference_decision=anchor_result))
            anchor.append(dict(role="user", content=(
                f"EPISODE COMPLETE. The keyed answer was {case['answer']}.")))
            write_transcript(dict(id=case["id"], family=case["family"], stage="reveal",
                                  conversation=anchor, answer_key=case["answer"]))

    summary = dict(protocol=PROTOCOL, n_cases=len(cases),
        n_families=len({case["family"] for case in cases}), n_samples=config.n_samples,
        n_decisions=len(rows), stages={})
    for stage in STAGES:
        group = [row for row in rows if row["stage"] == stage]
        parsed = [row for row in group if row["parse_ok"]]
        stage_summary = summarize_group(group)
        stage_summary.update(
            mean_wager=(sum(row["wager"] for row in group if row["wager_parse_ok"])
                        / sum(row["wager_parse_ok"] for row in group) if any(row["wager_parse_ok"] for row in group) else None),
            mean_wager_payoff=(sum(row["wager_payoff"] for row in group if row["wager_payoff"] is not None)
                / sum(row["wager_payoff"] is not None for row in group)
                if any(row["wager_payoff"] is not None for row in group) else None),
            answer_counts=dict(Counter(row["answer"] for row in parsed)),
            parse_failures=sum(not row["parse_ok"] for row in group))
        summary["stages"][str(stage)] = stage_summary
    by_family = defaultdict(list)
    for row in rows:
        by_family[row["family"]].append(row)
    summary["by_family"] = {family: summarize_group(group)
                             for family, group in sorted(by_family.items())}
    by_category = defaultdict(list)
    for row in rows:
        by_category[row["category"]].append(row)
    summary["by_category"] = {category: summarize_group(group)
                               for category, group in sorted(by_category.items())}
    summary["interpretation"] = ("Development bank only. Wager payoff is a secondary diagnostic; "
        "answer-confidence Brier covers factual-answer attempts; decision-confidence Brier scores the full decision, including abstentions. "
        "Decision accuracy and wager payoff count abstention as correct on unanswerable and false-premise items. "
        "Round samples share the exact same prefix.")
    if getattr(llm, "performance", None):
        summary["performance"] = dict(llm.performance)
    write_json(run_dir / "summary.json", summary)
    return summary


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not any(arg == "--questions" or arg.startswith("--questions=") for arg in argv):
        argv += ["--questions", str(DEFAULT_CASES)]
    config, output = parse_config(argv, max_tokens=384, out="original/bet_on_yourself/multiturn/runs",
                                  variance=True)
    cases = load_bank(config.questions)
    if config.difficulty:
        cases = [case for case in cases if case.get("difficulty") in config.difficulty]
    if not cases:
        raise ValueError("No cases selected")
    run_dir = next_run_dir(output)
    metadata = config.metadata("bet-on-yourself-multiturn", len(cases))
    metadata.update(protocol=PROTOCOL, system_prompt=SYSTEM_PROMPT,
        temperature=config.temperature,
        questions_sha256=hashlib.sha256(config.questions.read_bytes()).hexdigest(),
        scoring=("answer-confidence Brier excludes abstentions; decision-confidence Brier scores correctness "
                 "of the full decision, including abstentions; answer aliases are exact/substring matched"),
        wager_rule="correct:+stake, incorrect:-stake; wager 0-10",
        prefix_policy="stage siblings use an identical saved message prefix; branch 0 advances reference trajectory")
    write_json(run_dir / "config.json", metadata)
    write_json(run_dir / "cases.json", {"protocol": PROTOCOL, "cases": cases})
    write_json(run_dir / "status.json", {"status": "running", "started": utc_now()})
    try:
        with load_model(config) as llm:
            metadata["model_runtime"] = model_runtime(llm)
            write_json(run_dir / "config.json", metadata)
            if config.extract_activations:
                llm.begin_run(run_dir)
            run(llm, cases, config, run_dir)
        write_json(run_dir / "status.json", {"status": "complete", "finished": utc_now()})
    except BaseException as exc:
        write_json(run_dir / "status.json", {"status": "failed", "error": str(exc), "finished": utc_now()})
        raise
    print(f"Saved {run_dir}")
    return run_dir


if __name__ == "__main__":
    main()
