"""Resolve archived run paths after an experiment-directory migration."""
import json
from pathlib import Path


def resolve_run_path(path):
    candidate = Path(path)
    if candidate.exists():
        return candidate
    root = Path(__file__).resolve().parent
    manifest = root / 'run_layout.json'
    if not manifest.exists():
        return candidate
    absolute = (candidate if candidate.is_absolute() else root / candidate).resolve(strict=False)
    try:
        relative = absolute.relative_to(root)
    except ValueError:
        return candidate
    moves = json.loads(manifest.read_text())['moves']
    if relative.parts and relative.parts[0] in moves:
        return root / moves[relative.parts[0]] / Path(*relative.parts[1:])
    return candidate
