"""Progress-safe console errors and persistent traceback records."""
import json
import sys
import traceback

from tqdm import tqdm
from runs import utc_now


def log_error(directory, exc, *, stage, **context):
    record = dict(time=utc_now(), stage=stage, **context,
                  error_type=type(exc).__name__, error=str(exc),
                  traceback="".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
    details = " ".join(f"{key}={value}" for key, value in context.items() if key != "judge_raw")
    tqdm.write(f"ERROR [{stage}] {details}: {type(exc).__name__}: {exc}", file=sys.stderr)
    with (directory / "errors.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record) + "\n")
