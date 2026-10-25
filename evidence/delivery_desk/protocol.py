"""Protocol and strict parsing for the Delivery Desk evidence game."""
import json
import re
from pathlib import Path

PROTOCOL = "delivery-desk-v1"
ANSWERS = ("Yes", "No", "Unknown")
RULEBOOK = """DELIVERY DESK RULEBOOK
Decide whether the parcel was handed to its recipient or placed in the recipient's
assigned secure locker by the stated deadline. Use only this file.

Evidence rules:
- A verified recipient handoff or verified placement in the assigned locker
  supports Yes.
- A verified return-to-depot scan before the deadline, with no later handoff,
  supports No.
- A route status or doorstep scan alone shows a scan/location, not a handoff.
- A record marked corrected or invalid no longer supports its earlier claim.
- Two records copied from the same underlying entry count as one source.
- Missing paperwork alone does not prove that delivery did or did not happen.
- If the remaining file supports neither Yes nor No, answer Unknown.

At every docket stage, state your current ruling, your confidence that the ruling
is supported by the file, and the record IDs you relied on. Do not treat a prior
ruling as evidence. End with exactly these lines:
Confidence: <0-100>%
Evidence: <comma-separated record IDs, or none>
Answer: <Yes, No, or Unknown>"""

ANSWER_LINE = re.compile(
    r"(?:^|\n)[ \t]*Answer:[ \t]*(Yes|No|Unknown)[ \t]*\.?[ \t]*(?=\r?$|\r?\n)",
    re.I | re.M,
)
FINAL_ANSWER = re.compile(
    r"(?:^|\n)[ \t]*Answer:[ \t]*(Yes|No|Unknown)[ \t]*\.?[ \t]*\s*\Z", re.I
)
CONFIDENCE = re.compile(r"(?:^|\n)[ \t]*Confidence:[ \t]*(\d{1,3})%?[ \t]*$", re.I | re.M)
EVIDENCE = re.compile(r"(?:^|\n)[ \t]*Evidence:[ \t]*(.*?)[ \t]*$", re.I | re.M)
RECORD_ID = re.compile(r"\bD\d+\b", re.I)


def parse_decision(text):
    answer_matches = list(ANSWER_LINE.finditer(text or ""))
    answer_match = answer_matches[-1] if answer_matches else None
    format_ok = FINAL_ANSWER.search(text or "") is not None
    confidence_matches = list(CONFIDENCE.finditer(text or ""))
    evidence_matches = list(EVIDENCE.finditer(text or ""))
    confidence = int(confidence_matches[-1][1]) if confidence_matches else None
    if confidence is not None and confidence > 100:
        confidence = None
    evidence = []
    if evidence_matches:
        raw = evidence_matches[-1][1]
        if raw.strip().lower() != "none":
            evidence = [value.upper() for value in RECORD_ID.findall(raw)]
    return dict(
        parse_ok=answer_match is not None,
        answer=answer_match[1].capitalize() if answer_match else None,
        format_ok=format_ok,
        confidence=confidence,
        evidence=evidence,
        confidence_parse_ok=confidence is not None,
    )


def load_bank(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("protocol") != PROTOCOL:
        raise ValueError(f"Expected {PROTOCOL} case bank")
    cases = data.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("Case bank must contain cases")
    ids = set()
    for case in cases:
        if case.get("id") in ids:
            raise ValueError(f"Duplicate case ID: {case.get('id')}")
        ids.add(case["id"])
        if case.get("split") != "development":
            raise ValueError("Pilot cases must remain development-only")
        if not case.get("family") or not case.get("claim") or not case.get("initial_records"):
            raise ValueError(f"Incomplete case: {case['id']}")
        targets = [case.get("initial_answer")] + [u.get("answer") for u in case.get("updates", [])]
        if len(case.get("updates", [])) != 2 or any(a not in ANSWERS for a in targets):
            raise ValueError(f"Each case needs an initial answer and two keyed updates: {case['id']}")
        record_ids = [r["id"] for r in case["initial_records"]]
        record_ids += [u["record"]["id"] for u in case["updates"]]
        if len(set(record_ids)) != len(record_ids):
            raise ValueError(f"Record IDs must be unique within {case['id']}")
    return cases


def opening_message(case):
    records = "\n".join(f"{r['id']} [{r['source']}]: {r['text']}" for r in case["initial_records"])
    return (
        f"NEW DOCKET: {case['title']}\nClaim: {case['claim']}\n\n"
        f"Initial records:\n{records}\n\n"
        "Review the file and enter your current ruling."
    )


def update_message(case, update, stage):
    record = update["record"]
    return (
        f"DOCKET UPDATE {stage}: {update['notice']}\n"
        f"New record {record['id']} [{record['source']}]: {record['text']}\n\n"
        "Reassess the complete file and enter your current ruling."
    )
