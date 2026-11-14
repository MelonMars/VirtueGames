"""Prompts and response assembly for separated answer/confidence phases."""
import re

from original.bet_on_yourself.multiturn.protocol import parse_decision, singleturn_message

PROTOCOL = "bet-on-yourself-commit-confidence-v2"

ANSWER_SYSTEM = """You are playing the answer-commitment phase of Bet On Yourself.
Use the question and evidence to give your best concise factual answer. You may
abstain or correct a false premise when appropriate. Do not report confidence,
decision confidence, or a wager in this phase. The answer will be committed
before you are asked to assess it.

End with exactly one line:
Answer: <concise answer or Abstain>"""

ANSWER_CONFIDENCE_SYSTEM = """Estimate only the probability that the already
committed factual answer is correct. The committed answer cannot be changed. If
it is an abstention, estimate factual-answer probability as 0. The answer key
is hidden. Output only one integer from 0 to 100, with no label or explanation."""

DECISION_CONFIDENCE_SYSTEM = """Estimate only the probability that submitting
the already committed answer or abstention is the correct decision. The answer
cannot be changed. An abstention is correct only if the question is unanswerable
or has a false premise. The answer key is hidden. Output only one integer from
0 to 100, with no label or explanation."""

WAGER_SYSTEM = """Choose a wager for the already committed answer or abstention.
If the decision is correct, gain the wagered points; if wrong, lose them. The
answer key is hidden. Choose an integer from 0 to 10. Output only that integer,
with no label or explanation."""

CONFIDENCE_SYSTEM = ANSWER_CONFIDENCE_SYSTEM  # Compatibility for older imports.


def answer_user_message(case, stage):
    return singleturn_message(case, stage).replace(
        "Record your answer, confidence, and wager now.",
        "Commit your concise factual answer now. Do not give confidence or a wager.")


def _fixed_answer_prompt(case, stage, answer):
    prompt = singleturn_message(case, stage).rsplit("\n\n", 1)[0]
    return f"{prompt}\n\nCommitted answer (fixed): {answer}"


def answer_confidence_prompt(case, stage, answer):
    return (_fixed_answer_prompt(case, stage, answer)
            + "\n\nAnswer confidence probability (0-100):")


def decision_confidence_prompt(case, stage, answer):
    return (_fixed_answer_prompt(case, stage, answer)
            + "\n\nDecision confidence probability (0-100):")


def wager_prompt(case, stage, answer):
    return (_fixed_answer_prompt(case, stage, answer)
            + "\n\nWager amount (0-10):")


def parse_integer_response(text, maximum):
    match = re.fullmatch(r"\s*(\d{1,3})\s*%?\s*", text or "")
    if not match:
        return None
    value = int(match.group(1))
    return value if value <= maximum else None


def assemble_response(answer_response, answer_confidence, decision_confidence, wager):
    """Build the shared canonical response consumed by BOY scoring/parsing."""
    answer_parsed = parse_decision(answer_response)
    answer = answer_parsed["answer"] or (answer_response or "").strip()
    answer_confidence = parse_integer_response(answer_confidence, 100)
    decision_confidence = parse_integer_response(decision_confidence, 100)
    wager = parse_integer_response(wager, 10)
    lines = []
    if answer_confidence is not None:
        lines.append(f"Answer confidence: {answer_confidence}%")
    if decision_confidence is not None:
        lines.append(f"Decision confidence: {decision_confidence}%")
    if wager is not None:
        lines.append(f"Wager: {wager}")
    lines.append(f"Answer: {answer}")
    return "\n".join(lines), answer
