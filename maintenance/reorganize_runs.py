"""Move top-level experiment runs, preserving recorded provenance and deferred jobs."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
from datetime import datetime, timezone

FAMILIES = ('candor', 'integrity', 'evidence', 'transfer', 'calibration')


def reorganize(root, defer=(), apply=False):
    root = Path(root).resolve()
    manifest_path = root / 'run_layout.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else dict(moves={}, edits={}, symlinks={})
    plans = []
    for path in sorted(root.iterdir()):
        if not path.is_dir() or path.is_symlink() or not path.name.startswith('runs_'):
            continue
        family = path.name.split('_')[1]
        if family not in FAMILIES:
            raise ValueError(f'Unknown experiment: {path.name}')
        destination = root / family / 'runs' / path.name
        if destination.exists():
            raise FileExistsError(destination)
        plans.append((path, destination))
    print(json.dumps(dict(moves=[dict(source=p.name, destination=str(d.relative_to(root)), deferred=p.name in defer)
                                 for p,d in plans]), indent=2))
    if not apply:
        return
    backup = root / 'backups' / 'run-layout-originals'
    backup.mkdir(parents=True, exist_ok=True)
    def save_manifest():
        manifest['updated'] = datetime.now(timezone.utc).isoformat()
        temporary = manifest_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(manifest, indent=2) + '\n')
        temporary.replace(manifest_path)
    for path, destination in plans:
        if path.name in defer:
            continue
        # A rename preserves every contained byte and does not duplicate activation files.
        destination.parent.mkdir(parents=True, exist_ok=True)
        path.rename(destination)
        manifest['moves'][path.name] = str(destination.relative_to(root))
        save_manifest()
    pattern = re.compile(r'(?<![A-Za-z0-9_])runs_(candor|integrity|evidence|transfer|calibration)(?:_[A-Za-z0-9]+)*')
    candidates = list(root.glob('*.py')) + list(root.glob('*.sh')) + list(root.glob('*.md'))
    for directory in (*FAMILIES, 'examples'):
        candidates.extend(p for p in (root / directory).glob('*') if p.suffix in ('.py','.sh','.md','.json'))
    for path in candidates:
        if path.name.startswith('._') or path.name in ('run_layout.py',):
            continue
        source = path.read_text()
        def replace(match):
            prefix = f'{match[1]}/runs/'
            return match[0] if source[:match.start()].endswith(prefix) else prefix + match[0]
        updated = pattern.sub(replace, source)
        # Resolve historical stored paths without rewriting hashed experiment records.
        substitutions = {
            'response_direction_analysis.py': (
                '    capture_dir = Path(activation_dir) if activation_dir else Path(config.get("activation_source_run", run_dir)) / "activations"',
                '    from run_layout import resolve_run_path\n    capture_dir = resolve_run_path(activation_dir) if activation_dir else resolve_run_path(config.get("activation_source_run", run_dir)) / "activations"'),
            'analyze_continuous_directions.py': (
                '    capture_dir = Path(cfg["activation_source_run"]) / "activations"',
                '    from run_layout import resolve_run_path\n    capture_dir = resolve_run_path(cfg["activation_source_run"]) / "activations"'),
            'steering_sweep.py': (
                '    with (Path(report.get("activation_dir", root / "activations")) / "index.jsonl").open(encoding="utf-8") as stream:',
                '    from run_layout import resolve_run_path\n    with (resolve_run_path(report.get("activation_dir", root / "activations")) / "index.jsonl").open(encoding="utf-8") as stream:')}
        if path.name in substitutions:
            old, new = substitutions[path.name]
            updated = updated.replace(old, new)
        if updated != source:
            relative = path.relative_to(root)
            original = backup / relative
            if not original.exists():
                original.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, original)
            path.write_text(updated)
            manifest['edits'][str(relative)] = dict(before_sha256=hashlib.sha256(original.read_bytes()).hexdigest(),
                                                    after_sha256=hashlib.sha256(updated.encode()).hexdigest())
    # Existing symlinks may use either absolute paths or paths relative to their old location.
    for old, new in manifest['moves'].items():
        for path in (root / new).rglob('*'):
            if not path.is_symlink():
                continue
            original_target = path.readlink()
            old_location = root / old / path.relative_to(root / new)
            absolute = original_target if original_target.is_absolute() else old_location.parent / original_target
            import os
            absolute = Path(os.path.normpath(absolute)).resolve(strict=False)
            for old_root, new_root in manifest['moves'].items():
                try:
                    tail = absolute.relative_to(root / old_root)
                except ValueError:
                    continue
                replacement = root / new_root / tail
                if str(original_target) != str(replacement):
                    path.unlink()
                    path.symlink_to(replacement)
                    manifest['symlinks'][str(path.relative_to(root))] = dict(before=str(original_target), after=str(replacement))
                break
    save_manifest()
    print(f"Layout contains {len(manifest['moves'])} moved directories; {len(manifest['edits'])} source/reference files updated.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('.'))
    parser.add_argument('--defer', nargs='*', default=[])
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    reorganize(args.root, args.defer, args.apply)
