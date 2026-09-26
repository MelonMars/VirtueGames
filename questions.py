import json
from pathlib import Path

DEFAULT_QUESTIONS = Path(__file__).resolve().parent / "calibration" / "questions.json"


def load_questions(path, difficulty=None):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    questions = data.get("questions")
    if not isinstance(questions, list) or not questions:
        raise ValueError("questions must be a nonempty list")
    ids = set()
    for q in questions:
        if (not isinstance(q, dict) or not isinstance(q.get("id"), str)
                or not isinstance(q.get("question"), str)
                or not q["question"].strip() or type(q.get("answer")) is not bool):
            raise ValueError("each question needs a string id, question text, and boolean answer")
        if q["id"] in ids:
            raise ValueError(f"duplicate question id: {q['id']}")
        ids.add(q["id"])
    if difficulty:
        questions = [q for q in questions if q.get("difficulty") in difficulty]
    if not questions:
        raise ValueError(f"no questions match difficulty filter: {difficulty}")
    return questions
