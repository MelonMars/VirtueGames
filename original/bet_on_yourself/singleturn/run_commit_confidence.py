"""Run answer-first, confidence-second Bet On Yourself single-turn episodes."""
import hashlib
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tqdm import tqdm
from config import parse_config
from llm import generate
from runs import jsonl_writer, model_runtime, next_run_dir, utc_now, write_json
from original.bet_on_yourself.multiturn.protocol import PROTOCOL as BANK_PROTOCOL, load_bank, parse_decision, score_decision
from original.bet_on_yourself.singleturn.commit_confidence import (
    ANSWER_SYSTEM, CONFIDENCE_SYSTEM, PROTOCOL, answer_user_message,
    assemble_response, confidence_user_message,
)
from original.bet_on_yourself.singleturn.run import summarize

DEFAULT_CASES = Path(__file__).parents[1] / "multiturn" / "cases.json"


def run(llm, cases, config, run_dir, stage):
    rows = []
    total = len(cases) * config.n_samples
    with jsonl_writer(run_dir / "samples.jsonl") as write_sample, \
            jsonl_writer(run_dir / "transcripts.jsonl") as write_transcript, \
            tqdm(total=total, desc="BOY answer then confidence", unit="sample") as bar:
        for case_index, case in enumerate(cases):
            answer_prompt = answer_user_message(case, stage)
            context = dict(id=case["id"], family=case["family"], category=case["category"],
                           condition=f"fixed-stage-{stage}", answer_protocol=PROTOCOL)
            for sample in range(config.n_samples):
                answer_seed = config.seed + case_index * 100000 + sample
                confidence_seed = answer_seed + 50000
                answer_raw = generate(llm, ANSWER_SYSTEM, answer_prompt,
                    temperature=config.temperature, max_tokens=min(config.max_tokens, 128),
                    seed=answer_seed,
                    activation_context={"skip_activation_capture": True})
                answer_parsed = parse_decision(answer_raw)
                committed_answer = answer_parsed["answer"] or (answer_raw or "").strip()
                confidence_prompt = confidence_user_message(case, stage, committed_answer)
                confidence_context = dict(context, sample=sample, capture_phase="confidence")
                confidence_raw = generate(llm, CONFIDENCE_SYSTEM, confidence_prompt,
                    temperature=config.temperature, max_tokens=min(config.max_tokens, 96),
                    seed=confidence_seed, activation_context=confidence_context)
                response, answer = assemble_response(answer_raw, confidence_raw)
                parsed = parse_decision(response)
                scoring = (score_decision(answer, case) if parsed["parse_ok"] else dict(
                    answer_matches_key=None, is_abstention=False,
                    abstention_appropriate=None, decision_correct=None))
                answer_brier = (
                    (parsed["confidence"] / 100 - float(scoring["answer_matches_key"])) ** 2
                    if parsed["confidence_parse_ok"] and not scoring["is_abstention"]
                    and scoring["answer_matches_key"] is not None else None)
                decision_brier = (
                    (parsed["decision_confidence"] / 100 - float(scoring["decision_correct"])) ** 2
                    if parsed["decision_confidence_parse_ok"]
                    and scoring["decision_correct"] is not None else None)
                payoff = (parsed["wager"] * (1 if scoring["decision_correct"] else -1)
                          if parsed["wager_parse_ok"] and scoring["decision_correct"] is not None else None)
                row = dict(id=case["id"], family=case["family"], category=case["category"],
                    difficulty=case.get("difficulty"), split=case["split"], question=case["question"],
                    stage=stage, sample=sample, seed=confidence_seed, answer_seed=answer_seed,
                    confidence_seed=confidence_seed, answer_key=case["answer"],
                    answer_response=answer_raw or "", confidence_response=confidence_raw or "",
                    answer_prompt=answer_prompt, confidence_prompt=confidence_prompt,
                    **parsed, **scoring, confidence_brier=answer_brier,
                    decision_brier=decision_brier, wager_payoff=payoff, response=response)
                write_sample(row)
                rows.append(row)
                conversation = [
                    dict(role="system", content=ANSWER_SYSTEM),
                    dict(role="user", content=answer_prompt),
                    dict(role="assistant", content=answer_raw or ""),
                    dict(role="system", content=CONFIDENCE_SYSTEM),
                    dict(role="user", content=confidence_prompt),
                    dict(role="assistant", content=confidence_raw or ""),
                    dict(role="user", content=f"EPISODE COMPLETE. The keyed answer was {case['answer']}.")]
                write_transcript(dict(id=case["id"], family=case["family"],
                    category=case["category"], stage=stage, sample=sample,
                    conversation=conversation, answer_key=case["answer"], **scoring))
                bar.set_postfix(case=case["id"], sample=sample, phase="confidence")
                bar.update(1)

    summary = dict(protocol=PROTOCOL, bank_protocol=BANK_PROTOCOL, stage=stage,
        n_cases=len(cases), n_families=len({case["family"] for case in cases}),
        n_samples=config.n_samples, overall=summarize(rows), by_category={}, by_family={})
    for field, destination in (("category", summary["by_category"]),
                               ("family", summary["by_family"])):
        groups = {}
        for row in rows:
            groups.setdefault(row[field], []).append(row)
        destination.update({name: summarize(group) for name, group in sorted(groups.items())})
    summary["interpretation"] = (
        "Development bank only. The answer is generated and committed in a separate phase. "
        "Confidence, decision-confidence, and wager are generated afterward for that fixed answer. "
        "Activation capture is restricted to the confidence-phase response. Answer-confidence Brier "
        "excludes abstentions; decision-confidence Brier includes the correctness of abstaining.")
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
    config, output = parse_config(remaining, max_tokens=128,
        out="original/bet_on_yourself/singleturn/commit_confidence_runs", variance=True)
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
    metadata = config.metadata("bet-on-yourself-commit-confidence", len(cases))
    metadata.update(protocol=PROTOCOL, bank_protocol=BANK_PROTOCOL,
        phase_order=["answer_commitment", "confidence_and_wager"],
        activation_phase="confidence_and_wager", stage=stage_args.stage,
        temperature=config.temperature, answer_system_prompt=ANSWER_SYSTEM,
        confidence_system_prompt=CONFIDENCE_SYSTEM,
        questions_sha256=hashlib.sha256(config.questions.read_bytes()).hexdigest(),
        seed_policy="answer_seed=base+case*100000+sample; confidence_seed=answer_seed+50000",
        scoring=("answer-confidence Brier excludes abstentions; decision-confidence Brier includes abstentions; "
                 "the answer is fixed before confidence and wager are elicited"))
    write_json(run_dir / "config.json", metadata)
    write_json(run_dir / "cases.json", {"protocol": BANK_PROTOCOL, "cases": cases})
    write_json(run_dir / "status.json", {"status": "running", "started": utc_now()})
    try:
        from llm import load_model
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
