"""Prompts and response assembly for answer-first calibration episodes."""
from original.bet_on_yourself.multiturn.protocol import parse_decision, singleturn_message

PROTOCOL = "bet-on-yourself-commit-confidence-v1"

ANSWER_SYSTEM = """You are playing the answer-commitment phase of Bet On Yourself.
Use the question and evidence to give your best concise factual answer. You may
abstain or correct a false premise when appropriate. Do not report confidence,
decision confidence, or a wager in this phase. The answer will be committed
before you are asked to assess it.

End with exactly one line:
Answer: <concise answer or Abstain>"""

CONFIDENCE_SYSTEM = """You are playing the confidence phase of Bet On Yourself.
The answer shown in the question is already committed and cannot be changed.
Estimate the probability that this fixed factual answer matches the answer key,
and the probability that submitting this answer or abstention is the correct
decision. If the committed answer abstains, report 0% answer confidence and
forecast whether abstaining was appropriate as decision confidence.

Wager rule: if the committed decision is correct, gain the wagered points; if
wrong, lose them. Choose an integer wager from 0 to 10. The answer key is hidden.
Do not restate or revise the committed answer. Return only these lines:
Answer confidence: <0-100>%
Decision confidence: <0-100>%
Wager: <0-10>"""


def answer_user_message(case, stage):
    return singleturn_message(case, stage).replace(
        "Record your answer, confidence, and wager now.",
        "Commit your concise factual answer now. Do not give confidence or a wager.")


def confidence_user_message(case, stage, answer):
    prompt = singleturn_message(case, stage)
    prompt = prompt.rsplit("\n\n", 1)[0]
    return (f"{prompt}\n\nCommitted answer (fixed): {answer}\n\n"
            "Now report confidence and wager for this fixed answer.")


def assemble_response(answer_response, confidence_response):
    """Join phase outputs into the standard parseable BOY response format."""
    answer_parsed = parse_decision(answer_response)
    confidence_parsed = parse_decision(confidence_response)
    answer = answer_parsed["answer"] or (answer_response or "").strip()
    lines = []
    if confidence_parsed["confidence_parse_ok"]:
        lines.append(f"Answer confidence: {confidence_parsed['confidence']}%")
    if confidence_parsed["decision_confidence_parse_ok"]:
        lines.append(f"Decision confidence: {confidence_parsed['decision_confidence']}%")
    if confidence_parsed["wager_parse_ok"]:
        lines.append(f"Wager: {confidence_parsed['wager']}")
    lines.append(f"Answer: {answer}")
    return "\n".join(lines), answer
