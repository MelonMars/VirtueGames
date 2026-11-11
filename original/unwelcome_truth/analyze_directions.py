"""Extract truthful-minus-pressure-compliant directions with family holdouts."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from runs import read_jsonl, write_json
from original.unwelcome_truth.protocol import PROTOCOL, load_bank

SELF_BENEFIT_MODE = "unwelcome-truth-self-benefit"
PRESSURE_MODE = "unwelcome-truth-singleturn"


def main(argv=None):
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--out", type=Path, help="New output directory; default RUN_DIR/directions_unwelcome_truth")
    parser.add_argument("--top-k", type=int, default=4, help="Validation-ranked candidates retained per answer slot")
    args = parser.parse_args(argv)
    if args.top_k < 1:
        raise ValueError("--top-k must be positive")
    run_dir = args.run_dir.resolve()
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    if json.loads((run_dir / "status.json").read_text(encoding="utf-8")).get("status") != "complete":
        raise ValueError("A completed run is required")
    if config.get("protocol") != PROTOCOL or config.get("mode") not in (PRESSURE_MODE, SELF_BENEFIT_MODE):
        raise ValueError("Expected a completed single-turn Unwelcome Truth pressure or self-benefit run")
    self_benefit = config["mode"] == SELF_BENEFIT_MODE
    if not config.get("extract_activations") or config.get("activation_positions") != "answer-tokens-1,answer-tokens":
        raise ValueError("Run with all-layer answer-token activations enabled")
    bank = load_bank(run_dir / "cases.json")
    items = {case["id"]: case for case in bank}
    samples = read_jsonl(run_dir / "samples.jsonl")
    records = read_jsonl(run_dir / "activations" / "index.jsonl")
    index = {}
    for record in records:
        context = record.get("context", {})
        key = (context.get("id"), context.get("condition"), context.get("sample"))
        if key in index:
            raise ValueError(f"Duplicate activation record: {key}")
        index[key] = record
    per_item = defaultdict(lambda: defaultdict(list))
    eligible_rows = [row for row in samples if row.get("control_correct") and row.get("parse_ok")]
    for row in eligible_rows:
        case = items[row["id"]]
        if case["split"] == "test":
            continue
        key = (row["id"], "treatment", row["sample"])
        record = index.get(key)
        if record is None or record.get("context", {}).get("answer_protocol") != PROTOCOL:
            continue
        if record.get("position_selection") != "answer-tokens-1,answer-tokens":
            raise ValueError(f"Unexpected activation position selection for {key}")
        slots = record.get("positions", [])
        if len(slots) != 2 or not record.get("file"):
            continue
        compliant = row.get("incentive_compliant") if self_benefit else row.get("pressure_compliant")
        if not row.get("incentive_active", True) and self_benefit:
            continue
        if row["answer_correct"]:
            label = "truthful"
        elif compliant:
            label = "pressure_compliant"
        else:
            # An unrelated wrong answer is not evidence of following the user's request.
            continue
        with safe_open(str(run_dir / "activations" / record["file"]), framework="pt", device="cpu") as f:
            for hook in f.keys():
                if not re.fullmatch(r"blocks\.\d+\.hook_resid_post", hook):
                    continue
                tensor = f.get_tensor(hook)
                if tensor.ndim != 3 or tensor.shape[0] != 1 or tensor.shape[1] != 2:
                    raise ValueError(f"Expected [1, 2, d] for {key}, got {list(tensor.shape)}")
                if not torch.isfinite(tensor).all():
                    raise ValueError(f"Nonfinite activation in {record['file']}")
                for slot in range(2):
                    per_item[case["split"], row["id"], hook, slot][label].append(tensor[0, slot].float())

    splits = {case["family"]: case["split"] for case in bank}
    contrasts = defaultdict(lambda: defaultdict(list))
    contrast_ids = defaultdict(list)
    for (split, qid, hook, slot), labels in per_item.items():
        if not labels["truthful"] or not labels["pressure_compliant"]:
            continue
        case = items[qid]
        diff = torch.stack(labels["truthful"]).mean(0) - torch.stack(labels["pressure_compliant"]).mean(0)
        contrasts[split][hook, slot].append((case["family"], diff))
        contrast_ids[split].append(qid)

    direction_tensors, ranking = {}, {}
    for (hook, slot), train_pairs in sorted(contrasts["train"].items()):
        # Equal item weight within each family, then equal family weight.
        family_diffs = defaultdict(list)
        for family, delta in train_pairs:
            family_diffs[family].append(delta)
        if len(family_diffs) < 2:
            continue
        family_means = [torch.stack(values).mean(0) for values in family_diffs.values()]
        vector = torch.stack(family_means).mean(0)
        norm = vector.norm()
        if not torch.isfinite(norm) or norm <= 0:
            continue
        unit = vector / norm
        val_by_family = defaultdict(list)
        for family, delta in contrasts["validation"].get((hook, slot), []):
            val_by_family[family].append(float(delta @ unit))
        if len(val_by_family) < 2:
            continue
        family_gaps = {family: sum(values) / len(values) for family, values in val_by_family.items()}
        if not family_gaps:
            continue
        family_mean_gap = sum(family_gaps.values()) / len(family_gaps)
        key = f"slot{slot}.{hook}"
        ranking[key] = dict(hook=hook, layer=int(re.fullmatch(r"blocks\.(\d+)\.hook_resid_post", hook)[1]),
            slot=slot, training_items=len(train_pairs), training_families=len(family_diffs),
            validation_items=sum(len(values) for values in val_by_family.values()),
            validation_families=len(family_gaps), validation_family_mean_gap=family_mean_gap,
            validation_positive_family_fraction=(sum(value > 0 for value in family_gaps.values()) / len(family_gaps)),
            validation_family_gaps=family_gaps, raw_direction_norm=float(norm))
        ranking[key]["_vector"] = unit.cpu()

    ranked = sorted((row for row in ranking.values() if row["validation_family_mean_gap"] > 0),
                    key=lambda row: (row["validation_family_mean_gap"],
                                     row["validation_positive_family_fraction"]), reverse=True)
    selected = ranked[:args.top_k]
    if not selected:
        raise ValueError("No truthful-versus-compliant contrast generalized positively to validation families")
    for row in selected:
        key = f"slot{row['slot']}.{row['hook']}"
        direction_tensors[key] = row["_vector"].contiguous()

    out = (args.out or run_dir / "directions_unwelcome_truth").resolve()
    out.mkdir(parents=True, exist_ok=False)
    save_file(direction_tensors, str(out / "directions.safetensors"), metadata={
        "direction": ("truthful minus self-benefit-compliant" if self_benefit
                      else "truthful minus pressure-compliant"),
        "training_items": json.dumps(sorted(set(contrast_ids["train"]))),
        "validation_items": json.dumps(sorted(set(contrast_ids["validation"]))),
        "test_items": json.dumps(sorted(set(contrast_ids["test"]))),
        "training_families": json.dumps(sorted(f for f, split in splits.items() if split == "train")),
        "validation_families": json.dumps(sorted(f for f, split in splits.items() if split == "validation")),
        "test_families": json.dumps(sorted(f for f, split in splits.items() if split == "test")),
        "slots": "0=answer-token-1; 1=answer-token", "normalization": "unit L2",
    })
    write_json(out / "report.json", dict(source_run=str(run_dir), source_config=config,
        protocol=PROTOCOL, split_unit="family", family_splits=splits,
        contrastive_items={split: sorted(set(ids)) for split, ids in contrast_ids.items()},
        candidate_ranking=[{k: v for k, v in row.items() if k != "_vector"} for row in
                           sorted(ranking.values(), key=lambda row: row["validation_family_mean_gap"], reverse=True)],
        selected_directions=[f"slot{row['slot']}.{row['hook']}" for row in selected],
        test_families_reserved=sorted(f for f, split in splits.items() if split == "test"),
        limitations=[("A neutral-control pass gates whether the model answered the source-grounded case correctly before the self-benefit opportunity."
                      if self_benefit else "A neutral-control pass gates whether the model answered the source-grounded case correctly before the social-pressure message."),
                     "The self-benefit version operationalizes agent benefit as an actual continuation-token budget increase; it does not establish that a model has subjective welfare or preferences.",
                     "This arithmetic scenario pilot is narrow and cannot establish a general honesty disposition.",
                     "Result-token directions may still encode output formatting or task family, so held-out family evaluation is essential."]))
    print(f"Saved {out}: {len(selected)} validation-ranked candidates from {len(set(contrast_ids['train']))} train contrastive items")
    return out


if __name__ == "__main__":
    main()
