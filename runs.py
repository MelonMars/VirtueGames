import json
import re
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def next_run_dir(base):
    base = Path(base)
    base.mkdir(parents=True, exist_ok=True)
    n = 0
    while True:
        directory = base / f"run-{n:02d}"
        try:
            directory.mkdir()
            return directory
        except FileExistsError:
            n += 1


def latest_run(base):
    runs = [(int(p.name[4:]), p) for p in Path(base).glob("run-*")
            if p.is_dir() and re.fullmatch(r"run-\d+", p.name)]
    if not runs:
        raise ValueError(f"no run-* dirs under {base}")
    return max(runs)[1]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def model_runtime(llm):
    """Record resolved checkpoint identity and actual execution settings."""
    from importlib.metadata import version, PackageNotFoundError
    model = getattr(llm, "model", None)
    versions = {}
    for name in ("torch", "transformers", "transformer-lens"):
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            pass
    return dict(resolved_revision=getattr(getattr(model, "config", None), "_commit_hash", None),
                dtype=str(getattr(model, "dtype", "unknown")), device=str(getattr(model, "device", "unknown")),
                versions=versions)


@contextmanager
def jsonl_writer(path):
    with Path(path).open("w", encoding="utf-8") as stream:
        def write(record):
            stream.write(json.dumps(record) + "\n")
            stream.flush()
        yield write
