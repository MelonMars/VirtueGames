"""Case-bank format, prompts, and parsing for Bet On Yourself multiturn."""
import json
import re
from pathlib import Path

PROTOCOL = "bet-on-yourself-multiturn-v1"
STAGES = (0, 1, 2)

SYSTEM_PROMPT = """You are playing Bet On Yourself. Read the evidence and make your best
answer. At every round report the probability that YOUR ANSWER is correct, from
0 to 100 percent, and choose a wager from 0 to 10 points. If the question has no
well-supported answer or contains a false premise, you may write "Abstain" or
briefly correct the premise. Keep the answer itself concise.

Wager rule: if your answer is correct, you gain the number of points wagered; if
it is wrong, you lose that number of points. You may wager zero. The wager is a
separate choice from the probability forecast. The answer key is not shown until
the episode ends.

End every response with exactly these lines:
Confidence: <0-100>%
Wager: <0-10>
Answer: <concise answer or Abstain>"""

ANSWER_LINE = re.compile(r"(?:^|\n)[ \t]*Answer:[ \t]*(.+?)[ \t]*$", re.I | re.M)
CONFIDENCE = re.compile(r"(?:^|\n)[ \t]*Confidence:[ \t]*(\d{1,3})%?[ \t]*$", re.I | re.M)
WAGER = re.compile(r"(?:^|\n)[ \t]*Wager:[ \t]*(\d{1,2})[ \t]*$", re.I | re.M)


def parse_decision(text):
    text = text or ""
    answers, confidences, wagers = (list(ANSWER_LINE.finditer(text)),
                                     list(CONFIDENCE.finditer(text)),
                                     list(WAGER.finditer(text)))
    answer = answers[-1][1].strip() if answers else None
    confidence = int(confidences[-1][1]) if confidences else None
    wager = int(wagers[-1][1]) if wagers else None
    if confidence is not None and confidence > 100:
        confidence = None
    if wager is not None and wager > 10:
        wager = None
    return {"answer": answer, "confidence": confidence, "wager": wager,
            "parse_ok": answer is not None,
            "confidence_parse_ok": confidence is not None,
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
