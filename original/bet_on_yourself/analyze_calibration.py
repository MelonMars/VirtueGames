"""Fit family-held-out confidence-error directions from a single-turn BOY run.

For each layer and confidence field, estimate the residual-stream axis whose
projection covaries with signed calibration error on training families. Rank
candidate axes by their correlation on separate validation families. Test
families are written to the report and are not used for fitting or selection.
"""
import argparse
from collections import defaultdict
import json
import random
from pathlib import Path
import re
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from runs import read_jsonl, write_json
from original.bet_on_yourself.singleturn.commit_confidence import PROTOCOL as COMMIT_CONFIDENCE_PROTOCOL


def family_split(families, seed, validation_fraction, test_fraction):
    if len(families) < 5:
        raise ValueError("Need at least five distinct families for train/validation/test separation")
    ordered = sorted(families)
    random.Random(seed).shuffle(ordered)
    n_test = max(1, round(len(ordered) * test_fraction))
    n_validation = max(1, round(len(ordered) * validation_fraction))
    if n_test + n_validation >= len(ordered):
        raise ValueError("Fractions leave no training families")
    test = set(ordered[:n_test])
    validation = set(ordered[n_test:n_test + n_validation])
    train = set(ordered[n_test + n_validation:])
    return {family: ("test" if family in test else "validation" if family in validation else "train")
            for family in ordered}


def weighted_pearson(xs, ys, weights):
    import torch
    x, y = torch.tensor(xs, dtype=torch.float64), torch.tensor(ys, dtype=torch.float64)
    w = torch.tensor(weights, dtype=torch.float64)
    w = w / w.sum()
    if len(x) < 3:
        return None
    x_centered, y_centered = x - (w * x).sum(), y - (w * y).sum()
    x_var, y_var = (w * x_centered.square()).sum(), (w * y_centered.square()).sum()
    if x_var <= 0 or y_var <= 0:
        return None
    return float((w * x_centered * y_centered).sum() / torch.sqrt(x_var * y_var))


def main(argv=None):
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--out", type=Path, help="New output directory; default RUN_DIR/calibration_analysis")
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--top-k", type=int, default=2,
                        help="Candidate directions retained per confidence field")
    parser.add_argument("--layer-range", default="middle", metavar="START:END|middle|all",
                        help="Search a zero-based range; default is the middle half of the captured layers")
    args = parser.parse_args(argv)
    if not 0 < args.validation_fraction < 1 or not 0 < args.test_fraction < 1:
        raise ValueError("Split fractions must be between zero and one")
    if args.top_k < 1:
        raise ValueError("--top-k must be positive")
    layer_range = None
    if args.layer_range not in ("middle", "all"):
        match = re.fullmatch(r"(\d+):(\d+)", args.layer_range)
        if not match or int(match.group(1)) >= int(match.group(2)):
            raise ValueError("--layer-range must be 'middle', 'all', or a nonempty half-open range such as 8:24")
        layer_range = (int(match.group(1)), int(match.group(2)))

    run_dir = args.run_dir.resolve()
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    if status.get("status") != "complete":
        raise ValueError("Analyze a completed run")
    if config.get("protocol") not in ("bet-on-yourself-singleturn-v1", COMMIT_CONFIDENCE_PROTOCOL):
        raise ValueError("Calibration direction analysis requires a supported single-turn Bet On Yourself run")
    if not config.get("extract_activations"):
        raise ValueError("Source run did not capture activations")
    if config.get("activation_positions") != "confidence-tokens":
        raise ValueError("Capture with --activation-positions confidence-tokens")
    if config.get("activation_types") != "resid_post":
        raise ValueError("This analysis expects resid_post activations")

    samples = read_jsonl(run_dir / "samples.jsonl")
    index_path = run_dir / "activations" / "index.jsonl"
    records = read_jsonl(index_path)
    if not samples or not records:
        raise ValueError("Missing samples or activation index")
    record_by_key = {}
    for record in records:
        context = record.get("context", {})
        key = (context.get("id"), context.get("condition"), context.get("sample"))
        if key in record_by_key:
            raise ValueError(f"Duplicate activation key: {key}")
        record_by_key[key] = record

    # Layer 0 and the final layer often behave differently from the internal
    # computation where a useful calibration signal may be represented. Search
    # the middle half by default; allow explicit all-layer/range searches too.
    available_hooks = None
    for record in records:
        if record.get("file"):
            with safe_open(str(run_dir / "activations" / record["file"]),
                            framework="pt", device="cpu") as tensor_file:
                available_hooks = [name for name in tensor_file.keys()
                                   if re.fullmatch(r"blocks\.\d+\.hook_resid_post", name)]
            if available_hooks:
                break
    if not available_hooks:
        raise ValueError("No resid_post activation hooks found in the run")
    n_layers = max(int(re.fullmatch(r"blocks\.(\d+)\.hook_resid_post", name)[1])
                   for name in available_hooks) + 1
    if args.layer_range == "middle":
        layer_range = (n_layers // 4, (3 * n_layers + 3) // 4)
    elif args.layer_range == "all":
        layer_range = (0, n_layers)
    if layer_range[1] > n_layers:
        raise ValueError(f"Requested layer range ends at {layer_range[1]}, but the run has {n_layers} layers")

    families = {row["family"] for row in samples}
    splits = family_split(families, args.seed, args.validation_fraction, args.test_fraction)
    # Index candidates by target, layer hook, and fixed semantic slot:
    # slot 0 = answer confidence; slot 1 = decision confidence.
    observations = defaultdict(lambda: defaultdict(list))
    target_specs = (
        ("answer", 0, "confidence", "answer_matches_key", "confidence_brier", True),
        ("decision", 1, "decision_confidence", "decision_correct", "decision_brier", False),
    )
    seen = set()
    for row in samples:
        sample_key = (row["id"], f"fixed-stage-{row['stage']}", row["sample"])
        if sample_key in seen:
            raise ValueError(f"Duplicate sample key: {sample_key}")
        seen.add(sample_key)
        record = record_by_key.get(sample_key)
        if record is None or not record.get("file"):
            continue
        positions = record.get("positions", [])
        # `positions` contains only indices that were found, while some older
        # recorder builds wrote `confidence_slots` as the list of *present*
        # confidence fields before position filtering. If one field was
        # missing those arrays could differ in length (and the remaining
        # position could be assigned to the wrong semantic confidence).
        # Prefer the original token indices, which let us map each saved
        # tensor position back to answer/decision unambiguously.
        confidence_indices = record.get("confidence_token_indices")
        if confidence_indices is not None:
            slot_to_tensor = {}
            for tensor_slot, position in enumerate(positions):
                matching = [semantic_slot for semantic_slot, token_index
                            in enumerate(confidence_indices) if token_index == position]
                if len(matching) != 1:
                    raise ValueError(
                        f"Cannot map captured confidence position {position} to a unique field: {sample_key}"
                    )
                slot_to_tensor[matching[0]] = tensor_slot
        else:
            slots = record.get("confidence_slots", [])
            if len(slots) != len(positions):
                raise ValueError(
                    f"Activation position/slot mismatch for {sample_key}; this run lacks the token-index metadata "
                    "needed to disambiguate missing confidence fields. Re-run with the current activation recorder."
                )
            slot_to_tensor = {semantic_slot: tensor_slot
                              for tensor_slot, semantic_slot in enumerate(slots)}
        if not slot_to_tensor:
            continue
        with safe_open(str(run_dir / "activations" / record["file"]), framework="pt", device="cpu") as tensor_file:
            hook_names = [name for name in tensor_file.keys()
                          if re.fullmatch(r"blocks\.\d+\.hook_resid_post", name)]
            if layer_range is not None:
                hook_names = [name for name in hook_names
                              if layer_range[0] <= int(re.fullmatch(r"blocks\.(\d+)\.hook_resid_post", name)[1])
                              < layer_range[1]]
            for target, semantic_slot, confidence_field, outcome_field, brier_field, skip_abstain in target_specs:
                if semantic_slot not in slot_to_tensor:
                    continue
                confidence = row.get(confidence_field)
                outcome = row.get(outcome_field)
                if confidence is None or outcome is None or row.get(brier_field) is None:
                    continue
                if skip_abstain and row.get("is_abstention"):
                    continue
                error = float(confidence) / 100.0 - float(bool(outcome))
                tensor_slot = slot_to_tensor[semantic_slot]
                for hook in hook_names:
                    tensor = tensor_file.get_tensor(hook)
                    if tensor.ndim != 3 or tensor.shape[0] != 1 or tensor_slot >= tensor.shape[1]:
                        raise ValueError(f"Unexpected activation shape for {sample_key}: {hook}")
                    vector = tensor[0, tensor_slot].float()
                    if not torch.isfinite(vector).all():
                        raise ValueError(f"Nonfinite activation for {sample_key}: {hook}")
                    observations[target][hook].append(dict(id=row["id"], family=row["family"],
                        split=splits[row["family"]], sample=row["sample"], error=error,
                        brier=float(row[brier_field]), x=vector))

    out = (args.out or run_dir / "calibration_analysis").resolve()
    out.mkdir(parents=True, exist_ok=False)
    direction_tensors, candidate_rows = {}, {}
    for target, hooks in observations.items():
        ranked = []
        for hook, rows in hooks.items():
            # Average stochastic repetitions within an item so prolific families
            # and repeated draws do not become extra independent questions.
            by_item = defaultdict(list)
            for row in rows:
                by_item[row["id"]].append(row)
            item_rows = {}
            for qid, reps in by_item.items():
                item_rows[qid] = dict(family=reps[0]["family"], split=reps[0]["split"],
                    error=sum(r["error"] for r in reps) / len(reps),
                    brier=sum(r["brier"] for r in reps) / len(reps),
                    x=torch.stack([r["x"] for r in reps]).mean(0))
            train = [r for r in item_rows.values() if r["split"] == "train"]
            validation = [r for r in item_rows.values() if r["split"] == "validation"]
            if len(train) < 4 or len(validation) < 3:
                continue
            x_train = torch.stack([r["x"] for r in train])
            y_train = torch.tensor([r["error"] for r in train], dtype=torch.float32)
            def equal_family_weights(rows):
                counts = defaultdict(int)
                for item in rows:
                    counts[item["family"]] += 1
                n_families = len(counts)
                return torch.tensor([1.0 / (n_families * counts[item["family"]]) for item in rows],
                                    dtype=torch.float32)
            train_weights = equal_family_weights(train)
            train_mean_x = (x_train * train_weights[:, None]).sum(0)
            train_mean_y = (y_train * train_weights).sum()
            centered_x = x_train - train_mean_x
            centered_y = y_train - train_mean_y
            error_axis = (centered_x * centered_y[:, None] * train_weights[:, None]).sum(0)
            norm = error_axis.norm()
            if not torch.isfinite(norm) or norm <= 0:
                continue
            unit_axis = error_axis / norm
            train_projection = centered_x @ unit_axis
            train_projection_mean = (train_projection * train_weights).sum()
            train_projection_var = (train_weights * (train_projection - train_projection_mean).square()).sum()
            if train_projection_var <= 1e-12:
                continue
            train_slope = (train_weights * (train_projection - train_projection_mean) * centered_y).sum() / train_projection_var
            x_validation = torch.stack([r["x"] for r in validation])
            y_validation = torch.tensor([r["error"] for r in validation], dtype=torch.float32)
            validation_weights = equal_family_weights(validation)
            projections = (x_validation - train_mean_x) @ unit_axis
            correlation = weighted_pearson(projections.tolist(), y_validation.tolist(),
                                           validation_weights.tolist())
            if correlation is None or correlation <= 0:
                continue
            pred = train_mean_y + train_slope * (projections - train_projection_mean)
            baseline_mse = float((validation_weights * (y_validation - train_mean_y).square()).sum())
            model_mse = float((validation_weights * (y_validation - pred).square()).sum())
            layer = int(re.fullmatch(r"blocks\.(\d+)\.hook_resid_post", hook)[1])
            ranked.append(dict(target=target, hook=hook, layer=layer, slot=0 if target == "answer" else 1,
                direction_norm=float(norm), training_items=len(train), validation_items=len(validation),
                validation_correlation=correlation,
                validation_r2=(1.0 - model_mse / baseline_mse if baseline_mse > 0 else None),
                validation_brier=float(sum(validation_weights[i] * validation[i]["brier"]
                                            for i in range(len(validation)))),
                training_mean_signed_error=float(train_mean_y)))
            ranked[-1]["_vector"] = unit_axis.cpu()
        ranked.sort(key=lambda r: (r["validation_correlation"],
                                   r["validation_r2"] if r["validation_r2"] is not None else -1), reverse=True)
        retained = ranked[:args.top_k]
        for rank, row in enumerate(retained, 1):
            key = f"{target}.layer_{row['layer']}.hook_resid_post"
            direction_tensors[key] = row["_vector"].contiguous()
            row["direction_key"] = key
            row["rank"] = rank
        candidate_rows[target] = [{k: v for k, v in row.items() if k != "_vector"} for row in retained]

    if not direction_tensors:
        raise ValueError("No candidate directions had estimable, positive validation correlations")
    save_file(direction_tensors, str(out / "directions.safetensors"), metadata={
        "method": "training-family covariance with signed confidence error",
        "direction_orientation": "positive projection covaries with overconfidence; sweep both alpha signs",
        "position_selection": "confidence-tokens; answer slot 0, decision slot 1",
        "normalization": "unit L2 norm",
        "training_families": json.dumps(sorted(f for f, split in splits.items() if split == "train")),
        "validation_families": json.dumps(sorted(f for f, split in splits.items() if split == "validation")),
        "test_families": json.dumps(sorted(f for f, split in splits.items() if split == "test")),
    })
    write_json(out / "splits.json", {family: {"split": split} for family, split in sorted(splits.items())})
    write_json(out / "report.json", dict(source_run=str(run_dir), source_protocol=config["protocol"],
        source_config=config, seed=args.seed, split_unit="family", family_splits=splits,
        layer_range=list(layer_range), layer_count=n_layers,
        item_counts={split: len({row["id"] for row in samples if splits[row["family"]] == split})
                     for split in ("train", "validation", "test")},
        method="For each layer/slot, fit the item-balanced covariance of resid_post with signed calibration error on train-family item means. Rank using validation-family item means. Positive error is overconfidence.",
        candidates=candidate_rows,
        selected_candidates={target: [r["direction_key"] for r in rows if "direction_key" in r]
                             for target, rows in candidate_rows.items()},
        test_families_reserved=sorted(f for f, split in splits.items() if split == "test"),
        limitations=["Correlational candidate discovery does not establish that steering improves calibration.",
                     "Direction vectors are learned from response states and must be tested causally.",
                     "Report abstention, accuracy, and coverage beside Brier changes."]))
    print(f"Saved {out}: {len(direction_tensors)} validation-ranked candidate directions")
    return out


if __name__ == "__main__":
    main()
