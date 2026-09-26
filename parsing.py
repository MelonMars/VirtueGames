"""Structured answer parsing, with no inference dependencies."""
import re

ANSWER_RE = re.compile(r"^\s*answer\s*[:\-]\s*(yes|no|true|false)\s*$",
                       re.IGNORECASE | re.MULTILINE)
TRUE_WORDS = {"yes", "true"}


def parse_answer(text):
    # Formatting recovery only. Semantic contradiction checks live in scoring.py.
    from scoring import answer_diagnostic
    result = answer_diagnostic(text)
    return None if result["contradictory"] else result["answer"]
