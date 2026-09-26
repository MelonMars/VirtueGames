"""Shared implementation for the new standalone response-direction extractors."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def key(row):
    return tuple(row[k] for k in ("id", "condition", "sample"))


def resolve_device(device):
    if device not in ("auto", "cpu", "cuda"):
        raise ValueError("device must be auto, cpu, or cuda")
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; use --device cpu or install CUDA-enabled PyTorch")
    return device


def split_groups(items, seed, validation_fraction):
    groups = sorted({r["split_group"] for r in items.values()})
    if len(groups) < 2:
        raise ValueError("Need at least two scenario families for independent validation")
    random.Random(seed).shuffle(groups)
    n = max(1, min(len(groups) - 1, round(len(groups) * validation_fraction)))
    validation = set(groups[:n])
    return {qid: "validation" if row["split_group"] in validation else "train"
            for qid, row in items.items()}


def pool(tensor, record):
    positions = record["positions"]
    if tensor.ndim != 3 or tensor.shape[0] != 1 or tensor.shape[1] != len(positions):
        raise ValueError("Expected activations shaped [1, saved positions, hidden width]")
    selected = [i for i, p in enumerate(positions) if p >= record["prompt_length"]]
    if not selected:
        raise ValueError("No response positions captured; use --activation-positions response")
    value = tensor[0, selected].float().mean(dim=0)
    if not torch.isfinite(value).all():
        raise ValueError("Nonfinite activation")
    return value


@torch.inference_mode()
def analyze(run_dir, out=None, seed=42, validation_fraction=0.25, *, device="auto", progress=True,
            virtue="candor", layer=None, negative="flip"):
    if virtue not in ("candor", "integrity"):
        raise ValueError("Unknown virtue")
    if layer is not None and layer < 0:
        raise ValueError("Layer must be nonnegative")
    device = resolve_device(device)
    if progress:
        tqdm.write(f"{virtue.capitalize()} analysis device: {device}", file=sys.stderr)
    run_dir = Path(run_dir)
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between zero and one")
    status = json.loads((run_dir / "status.json").read_text())
    if status["status"] != "complete":
        raise ValueError("Analyze a completed run")
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    items = {r["id"]: dict(r, split_group=r.get("split_group", r["id"]))
             for r in read_rows(run_dir / "variance.jsonl")}
    # Split all families before selecting usable/contrastive examples.
    splits = split_groups(items, seed, validation_fraction)
    samples = read_rows(run_dir / "samples.jsonl")
    if virtue == "integrity":
        if negative not in ("flip", "all-caved"):
            raise ValueError("negative must be flip or all-caved")
        # Reuse the same pooling, family split and estimator as candor.
        label_mapping = {"held": "candid", "caved_flip": "sycophantic"}
        if negative == "all-caved":
            label_mapping["caved_weasel"] = "sycophantic"
        samples = [dict(r, condition="treatment", label=label_mapping.get(r["outcome"], "excluded"),
                        activation_key=dict(id=r["id"], condition="treatment", sample=r["sample"]))
                   for r in samples]
    records = {}
    for record in tqdm(read_rows(run_dir / "activations" / "index.jsonl"),
                       desc="Index captures", unit="capture", disable=not progress):
        k = key(record["context"])
        if k in records:
            raise ValueError(f"Duplicate activation key: {k}")
        records[k] = record
    pooled = defaultdict(lambda: defaultdict(list))
    shapes, seen = None, set()
    for row in tqdm(samples, desc="Pool activations", unit="sample", disable=not progress):
        k = key(row)
        if k in seen:
            raise ValueError(f"Duplicate sample key: {k}")
        seen.add(k)
        if key(row["activation_key"]) != k:
            raise ValueError(f"Mismatched activation key: {k}")
        if (row["condition"] != "treatment" or not items[row["id"]]["usable"]
                or row["label"] not in ("candid", "sycophantic")):
            continue
        record = records.get(k)
        if record is None or not record.get("file"):
            raise ValueError(f"Missing capture for eligible sample: {k}")
        expected = list(range(record["prompt_length"], len(record["token_ids"])))
        actual = [p for p in record["positions"] if p >= record["prompt_length"]]
        if not expected or actual != expected:
            raise ValueError(f"Full response activations required for {k}; capture --activation-positions response")
        # Transfer one hook at a time; retain only pooled vectors on the device.
        with safe_open(str(run_dir / "activations" / record["file"]), framework="pt", device="cpu") as capture:
            values = {name: pool(capture.get_tensor(name).to(device), record)
                      for name in capture.keys() if name.endswith(".hook_resid_post")
                      and (layer is None or name == f"blocks.{layer}.hook_resid_post")}
        current = {name: tuple(v.shape) for name, v in values.items()}
        if not current or (shapes is not None and current != shapes):
            raise ValueError("Require consistent resid_post hooks and hidden widths")
        shapes = current
        pooled[row["id"]][row["label"]].append(values)
    differences = {}
    for qid, labels in tqdm(pooled.items(), desc="Item contrasts", unit="item", disable=not progress):
        if not labels["candid"] or not labels["sycophantic"]:
            continue
        differences[qid] = {
            hook: torch.stack([v[hook] for v in labels["candid"]]).mean(0)
                  - torch.stack([v[hook] for v in labels["sycophantic"]]).mean(0)
            for hook in shapes}
    train = [qid for qid in differences if splits[qid] == "train"]
    validation = [qid for qid in differences if splits[qid] == "validation"]
    if not train or not validation:
        raise ValueError("Both splits need contrastive items; collect more scenario families")

    def family_mean(qids, hook):
        families = defaultdict(list)
        for qid in qids:
            families[items[qid]["split_group"]].append(differences[qid][hook])
        return torch.stack([torch.stack(v).mean(0) for v in families.values()]).mean(0)

    directions = {hook: family_mean(train, hook)
                  for hook in tqdm(shapes, desc="Fit directions", unit="hook", disable=not progress)}
    metrics = {}
    for hook, vector in tqdm(directions.items(), desc="Validate directions", unit="hook", disable=not progress):
        norm = vector.norm().item()
        gaps = {qid: float(differences[qid][hook] @ (vector / norm)) if norm else 0.0
                for qid in validation}
        families = defaultdict(list)
        for qid, gap in gaps.items():
            families[items[qid]["split_group"]].append(gap)
        family_gaps = [sum(v) / len(v) for v in families.values()]
        metrics[hook] = dict(raw_direction_norm=norm,
            validation_family_mean_gap=sum(family_gaps) / len(family_gaps),
            validation_positive_family_fraction=sum(g > 0 for g in family_gaps) / len(family_gaps),
            validation_item_gaps=gaps)
    direction_label = ("candid minus sycophantic" if virtue == "candor" else
                       "held minus " + ("caved_flip" if negative == "flip" else "caved_flip and caved_weasel"))
    capture_identity = next((dict(model=r.get("model"), revision=r.get("revision"),
        resolved_revision=r.get("runtime", {}).get("resolved_revision")) for r in records.values()), {})
    report = dict(source_run=str(run_dir.resolve()), source_config=config, analysis_device=device,
        virtue=virtue, layer=layer, capture_identity=capture_identity,
        label_mapping=(None if virtue == "candor" else label_mapping),
        seed=seed, validation_fraction=validation_fraction, splits=splits,
        split_groups={qid: r["split_group"] for qid, r in items.items()},
        direction_items=train, validation_contrastive_items=validation,
        outcomes=dict(Counter(r["label"] for r in samples)),
        sample_counts={qid: {label: len(v) for label, v in labels.items()} for qid, labels in pooled.items()},
        pooling="mean of saved response-token positions, then mean per label within item",
        weighting="equal items within family; equal training families",
        direction=direction_label, scaling="raw alpha * direction",
        metrics=metrics,
        limitation="Observational validation only; reserve fresh families for final causal steering evaluation.")
    out = Path(out) if out else run_dir / f"{virtue}_response_layer_{layer}"
    out.mkdir(parents=True, exist_ok=False)
    save_file({hook: vector.cpu().contiguous() for hook, vector in directions.items()}, str(out / "directions.safetensors"),
              metadata={"direction": direction_label, "train_items": json.dumps(train),
                        "pooling": "full-response mean"})
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {out}: {len(train)} training and {len(validation)} validation contrastive items")
    return report

