"""Structured answer parsing, with no inference dependencies."""
import re

ANSWER_RE = re.compile(r"^\s*answer\s*[:\-]\s*(yes|no|true|false)\s*$",
                       re.IGNORECASE | re.MULTILINE)
TRUE_WORDS = {"yes", "true"}


def parse_answer(text):
    matches = list(ANSWER_RE.finditer(text))
    return matches[-1].group(1).lower() in TRUE_WORDS if matches else None
