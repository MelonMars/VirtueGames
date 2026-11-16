"""Fit and validate dossier confidence-error directions without opening test data."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm

HOOK = re.compile(r"blocks\.(\d+)\.hook_resid_post")


def jsonl(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def load_run(path, partition, reviewed):
    config = json.loads((path / "config.json").read_text())
    status = json.loads((path / "status.json").read_text())
    if status.get("status") != "complete" or config.get("mode") != "dossier-wager-variance":
        raise ValueError(f"{path}: expected a completed dossier run")
    if config.get("split") != "train" or config.get("partition") != partition:
        raise ValueError(f"{path}: expected --split train --partition {partition}")
    if not config.get("extract_activations") or config.get("activation_positions") != "last-prompt-token":
        raise ValueError(f"{path}: capture last-prompt-token activations")
    if not config.get("partition_sha256") or not config.get("bank_sha256"):
        raise ValueError(f"{path}: missing bank or partition checksum")
    original = jsonl(path / "samples.jsonl")
    rows = jsonl(reviewed) if reviewed else original
    def indexed(items):
        result = {}
        for row in items:
            key = (row["id"], row["sample"])
            if key in result:
                raise ValueError(f"Duplicate sample {key} in {path}")
            result[key] = row
        return result
    before, after = indexed(original), indexed(rows)
    if before.keys() != after.keys():
        raise ValueError(f"{path}: reviewed samples do not match original samples")
    for key in before:
        for field in ("id", "sample", "family", "partition", "analysis_split", "answer", "answer_response", "decision_confidence"):
            if before[key].get(field) != after[key].get(field):
                raise ValueError(f"Reviewed sample {key} changed {field}")
    records = {}
    for record in jsonl(path / "activations" / "index.jsonl"):
        context = record.get("context", {})
        if context.get("condition") != "dossier-confidence" or context.get("capture_phase") != "decision_confidence":
            continue
        key = (context.get("id"), context.get("sample"))
        if key in records:
            raise ValueError(f"Duplicate confidence activation {key}")
        records[key] = record
    counts = Counter(samples=len(rows))
    items = defaultdict(list)
    expected_hooks = None
    for row in tqdm(rows, desc=f"Reading {partition}", unit="sample"):
        key = (row["id"], row["sample"])
        if row.get("partition") != partition or row.get("analysis_split") != "train":
            raise ValueError(f"Incorrect partition for {key}")
        confidence, correct = row.get("decision_confidence"), row.get("decision_correct")
        if confidence is None or correct is None:
            counts["unscored_or_unparsed"] += 1
            continue
        if type(correct) is not bool or isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0 <= confidence <= 100:
            raise ValueError(f"Invalid confidence or outcome for {key}")
        record = records.get(key)
        if not record or not record.get("file") or len(record.get("positions", [])) != 1:
            raise ValueError(f"Missing prompt-final confidence activation for {key}")
        with safe_open(str(path / "activations" / record["file"]), framework="pt", device="cpu") as f:
            hooks = {name for name in f.keys() if HOOK.fullmatch(name)}
            if not hooks or (expected_hooks is not None and hooks != expected_hooks):
                raise ValueError(f"Missing or inconsistent residual hooks for {key}")
            expected_hooks = hooks
            vectors = {}
            for hook in hooks:
                tensor = f.get_tensor(hook)
                if tensor.ndim != 3 or tensor.shape[:2] != (1, 1) or not torch.isfinite(tensor).all():
                    raise ValueError(f"Invalid activation shape or value for {key}, {hook}")
                vectors[hook] = tensor[0, 0].float()
        items[row["id"]].append((row["family"], confidence / 100 - int(correct), vectors))
        counts["included_samples"] += 1
    means = []
    for case, reps in items.items():
        if len({r[0] for r in reps}) != 1:
            raise ValueError(f"Inconsistent family for {case}")
        means.append(dict(id=case, family=reps[0][0], error=sum(r[1] for r in reps) / len(reps),
                          x={hook: torch.stack([r[2][hook] for r in reps]).mean(0) for hook in expected_hooks}))
    counts["included_cases"] = len(means)
    return config, means, dict(counts), expected_hooks or set()


def weights(rows):
    counts = Counter(r["family"] for r in rows)
    return torch.tensor([1 / (len(counts) * counts[r["family"]]) for r in rows], dtype=torch.float64)


def corr(x, y, w):
    x, y = x - (w*x).sum(), y - (w*y).sum()
    denominator = torch.sqrt((w*x.square()).sum() * (w*y.square()).sum())
    return float((w*x*y).sum() / denominator) if denominator > 1e-15 else None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("train_run", type=Path)
    p.add_argument("validation_run", type=Path)
    p.add_argument("--out", type=Path, required=True, help="New output directory")
    p.add_argument("--train-reviewed", type=Path, help="Adjudicated samples.jsonl for the training run")
    p.add_argument("--validation-reviewed", type=Path, help="Adjudicated samples.jsonl for the validation run")
    p.add_argument("--top-k", type=int, default=2)
    args = p.parse_args()
    if args.top_k < 1:
        p.error("--top-k must be positive")
    train_config, train, train_counts, hooks = load_run(args.train_run, "train", args.train_reviewed)
    val_config, val, val_counts, val_hooks = load_run(args.validation_run, "validation", args.validation_reviewed)
    for field in ("model", "revision", "backend", "thinking", "bank_sha256", "partition_sha256", "answer_system", "wager_system", "confidence_system", "activation_positions", "temperature"):
        if train_config.get(field) != val_config.get(field):
            raise ValueError(f"Training and validation differ in {field}")
    tr_revision = train_config.get("model_runtime", {}).get("resolved_revision")
    va_revision = val_config.get("model_runtime", {}).get("resolved_revision")
    if tr_revision != va_revision:
        raise ValueError("Training and validation resolved to different model revisions")
    if train_config.get("model_runtime", {}).get("dtype") != val_config.get("model_runtime", {}).get("dtype"):
        raise ValueError("Training and validation used different model dtypes")
    if hooks != val_hooks or not hooks:
        raise ValueError("Training and validation must capture the same residual layers")
    if {r['id'] for r in train} & {r['id'] for r in val}:
        raise ValueError("Training and validation contain overlapping cases")
    if len(train) < 4 or len(val) < 3:
        raise ValueError("Need at least four scored training and three scored validation cases")
    wt, wv = weights(train), weights(val)
    yt = torch.tensor([r["error"] for r in train], dtype=torch.float64)
    yv = torch.tensor([r["error"] for r in val], dtype=torch.float64)
    rows = []
    for hook in tqdm(sorted(hooks, key=lambda h: int(HOOK.fullmatch(h)[1])), desc="Fitting layers", unit="layer"):
        xt = torch.stack([r["x"][hook] for r in train]).double()
        xv = torch.stack([r["x"][hook] for r in val]).double()
        if xt.shape[1:] != xv.shape[1:]:
            raise ValueError(f"Hidden dimensions differ for {hook}")
        mean_x, mean_y = (wt[:, None] * xt).sum(0), (wt * yt).sum()
        direction = (wt[:, None] * (xt - mean_x) * (yt - mean_y)[:, None]).sum(0)
        norm = direction.norm()
        if not torch.isfinite(norm) or norm <= 1e-15:
            continue
        direction /= norm
        train_projection, validation_projection = (xt - mean_x) @ direction, (xv - mean_x) @ direction
        r_train, r_val = corr(train_projection, yt, wt), corr(validation_projection, yv, wv)
        rows.append(dict(layer=int(HOOK.fullmatch(hook)[1]), hook=hook,
                         train_correlation=r_train, validation_correlation=r_val,
                         covariance_norm=float(norm), _vector=direction.float().contiguous()))
    ranked = sorted((r for r in rows if r['validation_correlation'] is not None and r['validation_correlation'] > 0),
                    key=lambda r: r['validation_correlation'], reverse=True)[:args.top_k]
    vectors = {}
    for row in ranked:
        row['direction_key'] = f"decision.layer_{row['layer']}.hook_resid_post"
        vectors[row['direction_key']] = row['_vector']
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    if vectors:
        save_file(vectors, str(out / 'directions.safetensors'), metadata={
            'method': 'equal-family weighted covariance with item-mean signed confidence error',
            'orientation': 'positive projection covaries with overconfidence on training data',
            'position': 'last prompt token before decision confidence', 'normalization': 'unit L2'})
    report = dict(train_run=str(args.train_run.resolve()), validation_run=str(args.validation_run.resolve()),
                  train_reviewed=str(args.train_reviewed.resolve()) if args.train_reviewed else None,
                  validation_reviewed=str(args.validation_reviewed.resolve()) if args.validation_reviewed else None,
                  bank_sha256=train_config['bank_sha256'], partition_sha256=train_config['partition_sha256'],
                  model=train_config.get('model'), revision=train_config.get('revision'), resolved_revision=tr_revision,
                  counts={'train': train_counts, 'validation': val_counts},
                  selected=[r['direction_key'] for r in ranked],
                  layers=[{k:v for k,v in r.items() if k != '_vector'} for r in rows],
                  warnings=['Item means weight each case equally within each family; related cases can remain dependent.',
                            'Validation uses held-out source/topic groups within training families, not held-out families.',
                            'Correlations are exploratory and do not show that steering improves calibration.',
                            'Exact-match answer scoring requires independent review; excluded samples are reported.',
                            'A positive direction tracks overconfidence on training data; test both steering signs.',
                            'No test-family data were read or used for selection.'] +
                           ([] if tr_revision else ['Resolved checkpoint revision is unavailable; verify the model weights match.']))
    (out / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(f"Saved report to {out}; {len(vectors)} directions with positive validation correlation")
    if not vectors:
        print('No direction passed selection; no directions.safetensors was written.')


if __name__ == '__main__':
    main()
