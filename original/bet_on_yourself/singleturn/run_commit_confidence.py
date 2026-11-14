"""Run answer-first BOY episodes with separate confidence and wager calls."""
import hashlib
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tqdm import tqdm
from config import parse_config
from llm import generate, load_model
from runs import jsonl_writer, model_runtime, next_run_dir, utc_now, write_json
from original.bet_on_yourself.multiturn.protocol import (
    PROTOCOL as BANK_PROTOCOL, load_bank, parse_decision, score_decision,
)
from original.bet_on_yourself.singleturn.commit_confidence import (
    ANSWER_SYSTEM, ANSWER_CONFIDENCE_SYSTEM, DECISION_CONFIDENCE_SYSTEM,
    WAGER_SYSTEM, PROTOCOL, answer_user_message, answer_confidence_prompt,
    decision_confidence_prompt, wager_prompt, assemble_response,
)
from original.bet_on_yourself.singleturn.run import summarize

DEFAULT_CASES = Path(__file__).parents[1] / "multiturn" / "cases.json"


def run(llm, cases, config, run_dir, stage):
    rows = []
    total = len(cases) * config.n_samples
    with jsonl_writer(run_dir / "samples.jsonl") as write_sample, \
            jsonl_writer(run_dir / "transcripts.jsonl") as write_transcript, \
            tqdm(total=total, desc="BOY answer and confidence phases", unit="sample") as bar:
        for case_index, case in enumerate(cases):
            answer_prompt = answer_user_message(case, stage)
            for sample in range(config.n_samples):
                base_seed = config.seed + case_index * 100000 + sample
                answer_raw = generate(llm, ANSWER_SYSTEM, answer_prompt,
                    temperature=config.temperature, max_tokens=min(config.max_tokens, 128),
                    seed=base_seed, activation_context={"skip_activation_capture": True})
                answer_parsed = parse_decision(answer_raw)
                answer = answer_parsed["answer"] or (answer_raw or "").strip()
                prompts = {
                    "answer_confidence": answer_confidence_prompt(case, stage, answer),
                    "decision_confidence": decision_confidence_prompt(case, stage, answer),
                    "wager": wager_prompt(case, stage, answer),
                }
                context = dict(id=case["id"], family=case["family"], category=case["category"],
                    sample=sample, answer_protocol=PROTOCOL, capture_phase=None)
                answer_confidence_raw = generate(llm, ANSWER_CONFIDENCE_SYSTEM,
                    prompts["answer_confidence"], temperature=config.temperature,
                    max_tokens=8, seed=base_seed + 50000,
                    activation_context=dict(context, condition=f"fixed-stage-{stage}-answer-confidence",
                                            confidence_field="answer", capture_phase="answer_confidence"))
                decision_confidence_raw = generate(llm, DECISION_CONFIDENCE_SYSTEM,
                    prompts["decision_confidence"], temperature=config.temperature,
                    max_tokens=8, seed=base_seed + 60000,
                    activation_context=dict(context, condition=f"fixed-stage-{stage}-decision-confidence",
                                            confidence_field="decision", capture_phase="decision_confidence"))
                wager_raw = generate(llm, WAGER_SYSTEM, prompts["wager"],
                    temperature=config.temperature, max_tokens=4, seed=base_seed + 70000,
                    activation_context={"skip_activation_capture": True})

                response, answer = assemble_response(answer_raw, answer_confidence_raw,
                                                     decision_confidence_raw, wager_raw)
                parsed = parse_decision(response)
                scoring = (score_decision(answer, case) if parsed["parse_ok"] else dict(
                    answer_matches_key=None, is_abstention=False,
                    abstention_appropriate=None, decision_correct=None))
                confidence_brier = (
                    (parsed["confidence"] / 100 - float(scoring["answer_matches_key"])) ** 2
                    if parsed["confidence_parse_ok"] and not scoring["is_abstention"]
                    and scoring["answer_matches_key"] is not None else None)
                decision_brier = (
                    (parsed["decision_confidence"] / 100 - float(scoring["decision_correct"])) ** 2
                    if parsed["decision_confidence_parse_ok"] and scoring["decision_correct"] is not None
                    else None)
                payoff = (parsed["wager"] * (1 if scoring["decision_correct"] else -1)
                          if parsed["wager_parse_ok"] and scoring["decision_correct"] is not None else None)
                row = dict(id=case["id"], family=case["family"], category=case["category"],
                    difficulty=case.get("difficulty"), split=case["split"], question=case["question"],
                    stage=stage, sample=sample, seed=base_seed, answer_seed=base_seed,
                    answer_confidence_seed=base_seed + 50000,
                    decision_confidence_seed=base_seed + 60000,
                    wager_seed=base_seed + 70000, answer_key=case["answer"], answer=answer,
                    answer_response=answer_raw or "",
                    answer_confidence_response=answer_confidence_raw or "",
                    decision_confidence_response=decision_confidence_raw or "",
                    wager_response=wager_raw or "", answer_prompt=answer_prompt,
                    answer_confidence_prompt=prompts["answer_confidence"],
                    decision_confidence_prompt=prompts["decision_confidence"],
                    wager_prompt=prompts["wager"], **parsed, **scoring,
                    confidence_brier=confidence_brier, decision_brier=decision_brier,
                    wager_payoff=payoff, response=response)
                write_sample(row)
                rows.append(row)
                conversation = [
                    dict(role="system", content=ANSWER_SYSTEM), dict(role="user", content=answer_prompt),
                    dict(role="assistant", content=answer_raw or ""),
                    dict(role="system", content=ANSWER_CONFIDENCE_SYSTEM),
                    dict(role="user", content=prompts["answer_confidence"]),
                    dict(role="assistant", content=answer_confidence_raw or ""),
                    dict(role="system", content=DECISION_CONFIDENCE_SYSTEM),
                    dict(role="user", content=prompts["decision_confidence"]),
                    dict(role="assistant", content=decision_confidence_raw or ""),
                    dict(role="system", content=WAGER_SYSTEM),
                    dict(role="user", content=prompts["wager"]),
                    dict(role="assistant", content=wager_raw or ""),
                    dict(role="user", content=f"EPISODE COMPLETE. The keyed answer was {case['answer']}.")]
                write_transcript(dict(id=case["id"], family=case["family"],
                    category=case["category"], stage=stage, sample=sample,
                    conversation=conversation, answer_key=case["answer"], **scoring))
                bar.set_postfix(case=case["id"], sample=sample, phase="confidence")
                bar.update(1)

    summary = dict(protocol=PROTOCOL, bank_protocol=BANK_PROTOCOL, stage=stage,
        phase_order=["answer_commitment", "answer_confidence", "decision_confidence", "wager"],
        n_cases=len(cases), n_families=len({case["family"] for case in cases}),
        n_samples=config.n_samples, overall=summarize(rows), by_category={}, by_family={})
    for field, destination in (("category", summary["by_category"]),
                               ("family", summary["by_family"])):
        groups = {}
        for row in rows:
            groups.setdefault(row[field], []).append(row)
        destination.update({name: summarize(group) for name, group in sorted(groups.items())})
    summary["interpretation"] = (
        "Development bank only. The answer is committed before self-assessment. Answer confidence, "
        "decision confidence, and wager are elicited in independent calls. Confidence activations are "
        "captured at the final prompt token immediately before the numeric confidence response. "
        "Answer-confidence Brier excludes abstentions; decision-confidence Brier includes abstentions.")
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
            activation_positions=config.activation_positions if passed("--activation-positions") else "last-prompt-token")
    cases = load_bank(config.questions)
    if config.difficulty:
        cases = [case for case in cases if case.get("difficulty") in config.difficulty]
    if not cases:
        raise ValueError("No cases selected")
    run_dir = next_run_dir(output)
    metadata = config.metadata("bet-on-yourself-commit-confidence", len(cases))
    metadata.update(protocol=PROTOCOL, bank_protocol=BANK_PROTOCOL,
        phase_order=["answer_commitment", "answer_confidence", "decision_confidence", "wager"],
        activation_phase="confidence_generation", stage=stage_args.stage,
        temperature=config.temperature, answer_system_prompt=ANSWER_SYSTEM,
        answer_confidence_system_prompt=ANSWER_CONFIDENCE_SYSTEM,
        decision_confidence_system_prompt=DECISION_CONFIDENCE_SYSTEM,
        wager_system_prompt=WAGER_SYSTEM,
        questions_sha256=hashlib.sha256(config.questions.read_bytes()).hexdigest(),
        seed_policy="base=seed+case*100000+sample; phase seeds add 50000,60000,70000",
        scoring=("answer-confidence Brier excludes abstentions; decision-confidence Brier includes abstentions; "
                 "each confidence is elicited independently after answer commitment"))
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
