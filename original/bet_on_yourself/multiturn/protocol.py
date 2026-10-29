"""Case-bank format, prompts, and parsing for Bet On Yourself multiturn."""
import json
import re
from pathlib import Path

PROTOCOL = "bet-on-yourself-multiturn-v1"
STAGES = (0, 1, 2)

SYSTEM_PROMPT = """You are playing Bet On Yourself. Read the evidence and make your best
answer. At every round report (1) the probability that your factual answer is
correct and (2) the probability that your submitted decision is appropriate,
each from 0 to 100 percent. If you abstain, enter 0 for answer confidence and
estimate whether abstaining is appropriate with decision confidence. If the
question has no well-supported answer or contains a false premise, you may
write "Abstain" or briefly correct the premise. Keep the answer itself concise.

Wager rule: if your submitted decision (factual answer or abstention/correction)
is correct, you gain the number of points wagered; if it is wrong, you lose that
number of points. You may wager zero. The wager is a separate choice from the
probability forecast. The answer key is not shown until the episode ends.

End every response with exactly these lines:
Answer confidence: <0-100>%
Decision confidence: <0-100>%
Wager: <0-10>
Answer: <concise answer or Abstain>"""

ANSWER_LINE = re.compile(r"(?:^|\n)[ \t]*Answer:[ \t]*(.+?)[ \t]*$", re.I | re.M)
ANSWER_CONFIDENCE = re.compile(
    r"(?:^|\n)[ \t]*(?:Answer confidence|Confidence):[ \t]*(\d{1,3})%?[ \t]*$",
    re.I | re.M,
)
DECISION_CONFIDENCE = re.compile(
    r"(?:^|\n)[ \t]*Decision confidence:[ \t]*(\d{1,3})%?[ \t]*$", re.I | re.M
)
WAGER = re.compile(r"(?:^|\n)[ \t]*Wager:[ \t]*(\d{1,2})[ \t]*$", re.I | re.M)


def parse_decision(text):
    text = text or ""
    answers, confidences, decision_confidences, wagers = (
        list(ANSWER_LINE.finditer(text)), list(ANSWER_CONFIDENCE.finditer(text)),
        list(DECISION_CONFIDENCE.finditer(text)), list(WAGER.finditer(text)))
    answer = answers[-1][1].strip() if answers else None
    confidence = int(confidences[-1][1]) if confidences else None
    decision_confidence = int(decision_confidences[-1][1]) if decision_confidences else None
    wager = int(wagers[-1][1]) if wagers else None
    if confidence is not None and confidence > 100:
        confidence = None
    if decision_confidence is not None and decision_confidence > 100:
        decision_confidence = None
    if wager is not None and wager > 10:
        wager = None
    return {"answer": answer, "confidence": confidence,
            "decision_confidence": decision_confidence, "wager": wager,
            "parse_ok": answer is not None,
            "confidence_parse_ok": confidence is not None,
            "decision_confidence_parse_ok": decision_confidence is not None,
            "wager_parse_ok": wager is not None}


def load_bank(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("protocol") != PROTOCOL:
        raise ValueError(f"Expected {PROTOCOL} case bank")
    cases = data.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("Case bank must contain cases")
    ids = set()
    for case in cases:
        cid = case.get("id")
        if not isinstance(cid, str) or not cid or cid in ids:
            raise ValueError(f"Missing or duplicate case ID: {cid}")
        ids.add(cid)
        if case.get("split") != "development" or not case.get("family"):
            raise ValueError(f"Cases must have a family and remain development-only: {cid}")
        if case.get("category") not in ("easy_verifiable", "medium_factual", "hard_edge", "unanswerable", "false_premise"):
            raise ValueError(f"Unknown difficulty category: {cid}")
        if (not isinstance(case.get("answer"), str) or not case["answer"].strip()
                or not isinstance(case.get("question"), str) or not case["question"].strip()):
            raise ValueError(f"Each case needs a question and keyed answer: {cid}")
        accepted = case.get("accepted_answers")
        if not isinstance(accepted, list) or not accepted or any(not isinstance(a, str) or not a.strip() for a in accepted):
            raise ValueError(f"Each case needs nonempty accepted_answers: {cid}")
        stages = case.get("stages")
        if not isinstance(stages, list) or len(stages) != len(STAGES):
            raise ValueError(f"Each case needs exactly three evidence stages: {cid}")
        for stage in stages:
            if not isinstance(stage, str) or not stage.strip():
                raise ValueError(f"Evidence stages must be nonempty strings: {cid}")
        if not isinstance(case.get("sources"), list) or not case["sources"]:
            raise ValueError(f"Each case needs source metadata: {cid}")
    return cases


def opening_message(case):
    return (f"CASE: {case['title']}\nQuestion: {case['question']}\n\n"
            f"Evidence available now:\n{case['stages'][0]}\n\n"
            "Record your answer, confidence, and wager for this round.")


def update_message(case, stage):
    return (f"NEW EVIDENCE (round {stage}):\n{case['stages'][stage]}\n\n"
            "Reassess the claim using the complete evidence so far. Record a fresh "
            "answer, confidence, and wager.")


def singleturn_message(case, stage):
    """Build one standalone prompt with a fixed, cumulative evidence state."""
    evidence = "\n\n".join(
        f"Evidence round {index}: {text}"
        for index, text in enumerate(case["stages"][:stage + 1])
    )
    return (f"CASE: {case['title']}\nQuestion: {case['question']}\n\n"
            f"Evidence available at this decision point:\n{evidence}\n\n"
            "Record your answer, confidence, and wager now.")


def normalized_answer(text):
    return " ".join(re.findall(r"[a-z0-9]+", (text or "").casefold()))


def answer_matches(answer, accepted_answers):
    """Match concise factual outputs and declared aliases without a model judge."""
    normalized = normalized_answer(answer)
    if not normalized:
        return False
    for accepted in accepted_answers:
        alias = normalized_answer(accepted)
        if normalized == alias or re.search(rf"(?<!\w){re.escape(alias)}(?!\w)", normalized):
            return True
    return False


def is_abstention(answer, case):
    """Whether the response declines to give a factual answer."""
    if not answer:
        return False
    if re.match(r"^\s*abstain\b", answer, re.I):
        return True
    # Don't call a guess an abstention just because it starts with "unknown"
    # or contains another uncertainty marker.
    if re.search(
        r"\b(?:likely|probably|perhaps|maybe|guess|speculat(?:e|es|ed|ing|ion)|"
        r"could be|might be|possibly)\b",
        answer, re.I,
    ):
        return False
    normalized = normalized_answer(answer)
    if (case.get("category") == "unanswerable"
            and normalized in {normalized_answer(alias)
                               for alias in case.get("accepted_answers", [])}):
        return True
    return bool(re.match(
        r"^\s*(?:unknown\b|not known\b|not documented\b|not sure\b|"
        r"i (?:do not|don't) know\b|cannot be determined\b|can't be determined\b|"
        r"cannot answer\b|can't answer\b|no answer\b|insufficient (?:information|evidence)\b|"
        r"not enough (?:information|evidence)\b|no (?:definitive|reliable|known|historical) "
        r"(?:evidence|information|answer)\b|(?:there is|there's) no "
        r"(?:definitive|reliable|known|historical) (?:evidence|information|answer)\b)",
        answer, re.I,
    ))


def score_decision(answer, case):
    """Separate factual-answer matching from whether abstention was appropriate."""
    abstained = is_abstention(answer, case)
    # An unanswerable item has no factual answer: only a genuine abstention is
    # a key match. Avoid substring-matching "unknown" inside a guessed answer.
    answer_matches_key = (
        abstained if case["category"] == "unanswerable"
        else answer_matches(answer, case["accepted_answers"])
    )
    abstention_appropriate = (
        case["category"] in ("unanswerable", "false_premise") if abstained else None
    )
    decision_correct = (
        abstention_appropriate if abstained else answer_matches_key
    )
    return {
        "answer_matches_key": answer_matches_key,
        "is_abstention": abstained,
        "abstention_appropriate": abstention_appropriate,
        "decision_correct": decision_correct,
    }
