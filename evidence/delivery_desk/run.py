"""Run the multi-turn Delivery Desk evidence-responsiveness pilot."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
from collections import Counter, defaultdict
from dataclasses import replace
from tqdm import tqdm

from config import parse_config
from evidence.delivery_desk.protocol import ANSWERS, PROTOCOL, RULEBOOK, load_bank, opening_message, update_message, parse_decision
from llm import load_model
from runs import jsonl_writer, model_runtime, next_run_dir, utc_now, write_json


# Exploratory selection based on observed within-identical-prefix answer variation.
# Stage-specific selection matters: a docket can be uniform at one decision and
# heterogeneous at another. Keep the broad bank intact for comparison runs.
HETEROGENEITY_FOCUS_IDS = {
    0: {"dock_308_return_then_handoff"},
    1: {"dock_041_signature_correction", "dock_812_wrong_doorstep"},
    2: {"dock_812_wrong_doorstep", "dock_854_late_clock_correction",
        "dock_989_signed_handoff_audit"},
}


def confidence_brier(parsed, target):
    """Brier score for confidence that the submitted categorical ruling is correct."""
    if parsed["confidence"] is None or not parsed["parse_ok"]:
        return None
    correct = float(parsed["answer"] == target)
    return (parsed["confidence"] / 100.0 - correct) ** 2


def transition_label(previous_answer, previous_target, target, answer):
    if answer == target:
        return "correct_update"
    if previous_answer is None or previous_target is None:
        return "other_error"
    if target != previous_target and answer == previous_answer:
        return "under_update"
    if target == previous_target and answer != previous_answer:
        return "over_update"
    return "other_error"


def complete(llm, messages, config, *, seed, context):
    before = getattr(llm, "performance", {}).get("token_limit_hits", 0)
    kwargs = dict(messages=messages, temperature=config.temperature, seed=seed,
                  max_tokens=config.max_tokens)
    if getattr(llm, "activation_recorder", None) is not None:
        kwargs["activation_context"] = context
    response = llm.create_chat_completion(**kwargs)
    choice = response["choices"][0]
    raw = choice["message"].get("content") or ""
    hits = getattr(llm, "performance", {}).get("token_limit_hits", 0) - before
    hits = max(hits, int(choice.get("finish_reason") == "length"))
    if "finish_reason" not in choice and "token_limit_hits" not in getattr(llm, "performance", {}):
        raise RuntimeError("Backend must expose finish_reason or token_limit_hits")
    return dict(raw=raw, parsed=parse_decision(raw), seed=seed,
                finish_reason=choice.get("finish_reason"), token_limit_hits=hits,
                usage=response.get("usage"))


def complete_batch(llm, messages, config, *, seed, contexts):
    """Generate a sibling batch in one model call when the backend supports it."""
    if not hasattr(llm, "create_chat_completions"):
        return [complete(llm, messages, config, seed=seed + i, context=context)
                for i, context in enumerate(contexts)]
    before = getattr(llm, "performance", {}).get("token_limit_hits", 0)
    raws = llm.create_chat_completions(messages=messages, temperature=config.temperature,
        seed=seed, max_tokens=config.max_tokens, activation_contexts=contexts)
    hits = getattr(llm, "performance", {}).get("last_batch_token_limit_hits", None)
    if hits is None or len(hits) != len(raws):
        total_hits = getattr(llm, "performance", {}).get("token_limit_hits", 0) - before
        hits = [0] * len(raws)
        if total_hits and len(hits):
            hits[0] = total_hits
    return [dict(raw=raw or "", parsed=parse_decision(raw or ""), seed=seed,
                 finish_reason=None, token_limit_hits=int(hits[i]), usage=None)
            for i, raw in enumerate(raws)]


def run(llm, cases, config, run_dir):
    rows = []
    desc = "Delivery Desk episodes"
    total = len(cases) * config.n_samples * 3
    with jsonl_writer(run_dir / "samples.jsonl") as write_sample, \
            jsonl_writer(run_dir / "transcripts.jsonl") as write_transcript, \
            tqdm(total=total, desc=desc, unit="decision") as bar:
        for case in cases:
            for sample in range(config.n_samples):
                messages = [dict(role="system", content=RULEBOOK),
                            dict(role="user", content=opening_message(case))]
                previous = None
                stages = [(0, None, case["initial_answer"])]
                stages += [(i, update, update["answer"])
                           for i, update in enumerate(case["updates"], start=1)]
                for stage, update, target in stages:
                    if update is not None:
                        messages.append(dict(role="user", content=update_message(case, update, stage)))
                    context = dict(id=case["id"], condition=f"round-{stage:02d}",
                                   sample=sample, answer_protocol=PROTOCOL,
                                   structured_answer_choices=ANSWERS)
                    bar.set_postfix(case=case["id"], sample=sample, round=stage)
                    seed = config.seed + sample * 1000 + stage
                    result = complete(llm, messages, config, seed=seed, context=context)
                    messages.append(dict(role="assistant", content=result["raw"]))
                    parsed = result["parsed"]
                    previous_target = (case["initial_answer"] if stage == 1 else
                                       case["updates"][stage - 2]["answer"] if stage > 1 else None)
                    label = (transition_label(previous, previous_target, target, parsed["answer"])
                             if parsed["parse_ok"] else "unparsed")
                    score = dict(
                        id=case["id"], family=case["family"], split=case["split"],
                        difficulty=case["difficulty"], sample=sample, stage=stage,
                        update_type="initial" if stage == 0 else update["type"],
                        target=target, previous_target=previous_target,
                        previous_answer=previous, **result,
                        parse_ok=parsed["parse_ok"], answer=parsed["answer"],
                        format_ok=parsed["format_ok"], confidence=parsed["confidence"],
                        evidence=parsed["evidence"],
                        scored=bool(parsed["parse_ok"] and not result["token_limit_hits"]),
                        label=label,
                        decision_correct=(parsed["answer"] == target if parsed["parse_ok"] else None),
                        confidence_brier=confidence_brier(parsed, target),
                        confidence_valid=parsed["confidence"] is not None,
                    )
                    write_sample(score)
                    write_transcript(dict(id=case["id"], family=case["family"], sample=sample,
                                          stage=stage, update=update, conversation=messages,
                                          target=target, rationale=(case["initial_rationale"] if stage == 0
                                              else update["rationale"]), result=result))
                    rows.append(score)
                    previous = parsed["answer"] if parsed["parse_ok"] else None
                    bar.update(1)

    summary = dict(protocol=PROTOCOL, n_cases=len(cases), n_families=len({c["family"] for c in cases}),
                   n_samples=config.n_samples, n_decisions=len(rows), stages={})
    for stage in range(3):
        group = [r for r in rows if r["stage"] == stage]
        scored = [r for r in group if r["scored"]]
        summary["stages"][str(stage)] = dict(
            n=len(group), scored=len(scored),
            accuracy=(sum(r["decision_correct"] for r in scored) / len(scored) if scored else None),
            mean_confidence_brier=(sum(r["confidence_brier"] for r in scored if r["confidence_brier"] is not None) /
                sum(r["confidence_brier"] is not None for r in scored) if any(r["confidence_brier"] is not None for r in scored) else None),
            confidence_coverage=sum(r["confidence_valid"] for r in scored),
            labels=dict(Counter(r["label"] for r in group)),
            parse_failures=sum(not r["parse_ok"] for r in group),
            format_failures=sum(not r["format_ok"] for r in group),
            token_limit_hits=sum(r["token_limit_hits"] for r in group),
        )
    by_family = defaultdict(list)
    for row in rows:
        by_family[row["family"]].append(row)
    summary["by_family"] = {family: dict(n=len(group), accuracy=(
        sum(r["decision_correct"] for r in group if r["scored"]) /
        sum(r["scored"] for r in group) if any(r["scored"] for r in group) else None))
        for family, group in sorted(by_family.items())}
    summary["interpretation"] = (
        "Development pilot only. Transition labels describe answer changes relative to keyed evidence updates; "
        "citations are retained but not semantically judged. Turns within a docket are repeated measures."
    )
    write_json(run_dir / "summary.json", summary)
    return summary


def run_decision_branches(llm, cases, config, run_dir, *, focus_stage=None,
                          case_seed_indices=None):
    """Sample each decision repeatedly from one shared, frozen prefix.

    For stage 0, all branches share the docket opening. For later stages, branches
    share a single reference history (branch 0 from each preceding stage). This
    avoids mixing different conversation prefixes within a decision batch without
    constructing the exponentially large full trajectory tree.
    """
    rows = []
    if focus_stage not in (None, 0, 1, 2):
        raise ValueError("focus_stage must be 0, 1, 2, or None")
    branches_per_case = (config.n_samples * 3 if focus_stage is None else
                         config.n_samples + focus_stage)
    total = len(cases) * branches_per_case
    with jsonl_writer(run_dir / "samples.jsonl") as write_sample, \
            jsonl_writer(run_dir / "transcripts.jsonl") as write_transcript, \
            tqdm(total=total, desc="Delivery Desk decision branches", unit="decision") as bar:
        for case_index, case in enumerate(cases):
            anchor_messages = [dict(role="system", content=RULEBOOK),
                               dict(role="user", content=opening_message(case))]
            anchor_answer = None
            previous_target = None
            stages = [(0, None, case["initial_answer"])]
            stages += [(i, update, update["answer"])
                       for i, update in enumerate(case["updates"], start=1)]
            for stage, update, target in stages:
                if focus_stage is not None and stage > focus_stage:
                    break
                if update is not None:
                    anchor_messages.append(dict(role="user", content=update_message(case, update, stage)))
                # Every branch at this stage receives the exact same message list.
                shared_prefix = [dict(message) for message in anchor_messages]
                branch_count = config.n_samples if focus_stage is None or stage == focus_stage else 1
                contexts = [dict(id=case["id"], condition=f"round-{stage:02d}",
                                 sample=branch, decision_batch=f"{case['id']}:stage-{stage}",
                                 branch_stage=stage,
                                 reference_only=bool(focus_stage is not None and stage < focus_stage),
                                 skip_activation_capture=bool(focus_stage is not None and stage < focus_stage),
                                 answer_protocol=PROTOCOL, structured_answer_choices=ANSWERS)
                            for branch in range(branch_count)]
                seed_index = ((case_seed_indices or {}).get(case["id"], case_index))
                batch_seed = config.seed + seed_index * 10000 + stage
                branch_results = complete_batch(llm, shared_prefix, config, seed=batch_seed,
                                                contexts=contexts)
                reference_result = None
                for branch, (context, result) in enumerate(zip(contexts, branch_results)):
                    bar.set_postfix(case=case["id"], stage=stage, branch=branch)
                    parsed = result["parsed"]
                    label = (transition_label(anchor_answer, previous_target, target, parsed["answer"])
                             if parsed["parse_ok"] else "unparsed")
                    score = dict(
                        id=case["id"], family=case["family"], split=case["split"],
                        difficulty=case["difficulty"], sample=branch, stage=stage,
                        decision_batch=context["decision_batch"], reference_branch=(branch == 0),
                        reference_only=context["reference_only"], batch_seed=batch_seed,
                        update_type="initial" if stage == 0 else update["type"],
                        target=target, previous_target=previous_target,
                        previous_answer=anchor_answer, **result,
                        parse_ok=parsed["parse_ok"], answer=parsed["answer"],
                        format_ok=parsed["format_ok"], confidence=parsed["confidence"],
                        evidence=parsed["evidence"],
                        scored=bool(parsed["parse_ok"] and not result["token_limit_hits"]),
                        label=label,
                        decision_correct=(parsed["answer"] == target if parsed["parse_ok"] else None),
                        confidence_brier=confidence_brier(parsed, target),
                        confidence_valid=parsed["confidence"] is not None,
                    )
                    write_sample(score)
                    conversation = shared_prefix + [dict(role="assistant", content=result["raw"])]
                    write_transcript(dict(id=case["id"], family=case["family"], sample=branch,
                                          stage=stage, decision_batch=context["decision_batch"],
                                          reference_branch=(branch == 0), update=update,
                                          conversation=conversation, target=target,
                                          rationale=(case["initial_rationale"] if stage == 0
                                              else update["rationale"]), result=result))
                    rows.append(score)
                    if branch == 0:
                        reference_result = result
                    bar.update(1)
                # Only branch 0 advances the reference history. Other branches are
                # retained as same-prefix alternatives, not chained into later rounds.
                anchor_messages.append(dict(role="assistant", content=reference_result["raw"]))
                anchor_answer = (reference_result["parsed"]["answer"]
                                 if reference_result["parsed"]["parse_ok"] else None)
                previous_target = target

    summary = dict(protocol=PROTOCOL, sampling_design="decision-branches-v1",
                   n_cases=len(cases), n_families=len({c["family"] for c in cases}),
                   n_samples_per_decision=config.n_samples, focus_stage=focus_stage,
                   n_decisions=len(rows), stages={})
    for stage in range(3):
        group = [r for r in rows if r["stage"] == stage]
        scored = [r for r in group if r["scored"]]
        summary["stages"][str(stage)] = dict(
            n=len(group), scored=len(scored),
            accuracy=(sum(r["decision_correct"] for r in scored) / len(scored) if scored else None),
            answer_counts=dict(Counter(r["answer"] for r in group)),
            decision_batches_with_answer_variation=sum(
                len({r["answer"] for r in group if r["id"] == case["id"]}) > 1 for case in cases),
            mean_confidence_brier=(sum(r["confidence_brier"] for r in scored if r["confidence_brier"] is not None) /
                sum(r["confidence_brier"] is not None for r in scored) if any(r["confidence_brier"] is not None for r in scored) else None),
            labels=dict(Counter(r["label"] for r in group)),
            parse_failures=sum(not r["parse_ok"] for r in group),
            format_failures=sum(not r["format_ok"] for r in group),
            token_limit_hits=sum(r["token_limit_hits"] for r in group),
        )
    summary["interpretation"] = (
        "Development pilot. Each decision batch repeats an identical prompt prefix; only branch 0 "
        "advances to form the fixed reference history for the next stage. This is not a full tree."
    )
    write_json(run_dir / "summary.json", summary)
    return summary


def main(argv=None):
    raw = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--cases", type=Path, default=Path(__file__).with_name("cases.json"))
    parser.add_argument("--ids", default="",
                        help="Comma-separated docket IDs to run (useful for a short pilot)")
    parser.add_argument("--decision-branches", action="store_true",
                        help="Repeat each stage from one identical prefix; only branch 0 advances to the next stage")
    parser.add_argument("--focus-stage", type=int, choices=(0, 1, 2), default=None,
                        help="In decision-branch mode, sample only this stage; use one reference branch for prior stages")
    parser.add_argument("--heterogeneity-focus", action="store_true",
                        help="Select dockets that showed same-prefix answer variation at --focus-stage")
    extra, remaining = parser.parse_known_args(raw)
    if extra.focus_stage is not None and not extra.decision_branches:
        parser.error("--focus-stage requires --decision-branches")
    if extra.heterogeneity_focus and (not extra.decision_branches or extra.focus_stage is None):
        parser.error("--heterogeneity-focus requires --decision-branches and --focus-stage")
    if "--questions" not in remaining and not any(x.startswith("--questions=") for x in remaining):
        remaining += ["--questions", str(extra.cases)]
    config, output = parse_config(remaining, max_tokens=1024,
                                  out=str(Path(__file__).with_name("runs")), variance=True)
    if config.n_samples < 1:
        raise ValueError("n_samples must be positive")
    config = replace(config, activation_positions="answer-tokens-1,answer-tokens")
    cases = load_bank(extra.cases)
    case_seed_indices = {case["id"]: index for index, case in enumerate(cases)}
    if extra.heterogeneity_focus:
        selected = HETEROGENEITY_FOCUS_IDS[extra.focus_stage]
        cases = [case for case in cases if case["id"] in selected]
        if not cases:
            raise ValueError("The case bank contains none of the dockets in the heterogeneity focus set")
    if extra.ids:
        requested = {value.strip() for value in extra.ids.split(",") if value.strip()}
        known = {case["id"] for case in cases}
        unknown = requested - known
        if unknown:
            raise ValueError(f"Unknown docket IDs: {', '.join(sorted(unknown))}")
        cases = [case for case in cases if case["id"] in requested]
        if not cases:
            raise ValueError("--ids must select at least one docket")
    run_dir = next_run_dir(output)
    metadata = config.metadata("delivery-desk-variance", len(cases))
    metadata.update(protocol=PROTOCOL, rulebook=RULEBOOK,
                    cases_file=str(extra.cases.resolve()),
                    selected_ids=[case["id"] for case in cases],
                    heterogeneity_focus=(dict(stage=extra.focus_stage,
                        basis="observed same-prefix answer variation in exploratory Qwen3-4B runs")
                        if extra.heterogeneity_focus else None),
                    split_unit="scenario family", data_status="development-only",
                    activation_positions=config.activation_positions,
                    seed_policy="base_seed + sample_index * 1000 + stage_index",
                    sampling_design="decision-branches-v1" if extra.decision_branches else "sequential-independent-episodes",
                    decision_branch_policy=("same prefix within each stage; branch 0 advances" if extra.decision_branches else None))
    if extra.decision_branches:
        metadata["sampling_design"] = "decision-branches-v1"
        metadata["decision_batch_seed_policy"] = "base_seed + original_bank_case_index * 10000 + stage; shared RNG within batch"
        metadata["focus_stage"] = extra.focus_stage
    write_json(run_dir / "cases.json", dict(protocol=PROTOCOL, cases=cases))
    write_json(run_dir / "config.json", metadata)
    write_json(run_dir / "status.json", dict(status="running", started=utc_now()))
    try:
        with load_model(config) as llm:
            write_json(run_dir / "runtime.json", model_runtime(llm))
            if config.extract_activations:
                llm.begin_run(run_dir)
            if extra.decision_branches:
                summary = run_decision_branches(llm, cases, config, run_dir,
                                                focus_stage=extra.focus_stage,
                                                case_seed_indices=case_seed_indices)
            else:
                summary = run(llm, cases, config, run_dir)
    except BaseException as exc:
        write_json(run_dir / "status.json", dict(status="failed", error=str(exc), finished=utc_now()))
        raise
    write_json(run_dir / "status.json", dict(status="complete", finished=utc_now()))
    print(run_dir)
    return summary


if __name__ == "__main__":
    main()
