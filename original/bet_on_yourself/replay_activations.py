"""Replay saved Bet On Yourself transcripts to capture their model activations.

This makes activation extraction possible for completed runs that saved transcripts
but were originally generated without --extract-activations.
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from tqdm import tqdm

from runs import read_jsonl, write_json
from transformers_backend import TransformersModel


def load_config(run_dir, args):
    saved = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    if saved.get("backend") != "transformers":
        raise ValueError("Activation replay requires a Transformers run and its exact model checkpoint")
    runtime = saved.get("model_runtime", {})
    saved_dtype = str(runtime.get("dtype", saved.get("dtype", "auto"))).replace("torch.", "")
    if saved_dtype not in ("auto", "float32", "float16", "bfloat16"):
        saved_dtype = "auto"
    return SimpleNamespace(
        model=saved["model"], revision=runtime.get("resolved_revision") or saved.get("revision", "main"),
        device=args.device or saved.get("device", "auto"),
        dtype=args.dtype or saved_dtype,
        thinking=saved.get("thinking", "auto"), n_ctx=saved.get("n_ctx", 4096),
        extract_activations=True, activation_layers=args.layers,
        activation_types=args.types, activation_positions=args.positions,
        runtime=saved.get("runtime", "local"), backend="transformers",
    )


def activation_context(transcript, protocol):
    stage = int(transcript.get("stage", 0))
    context = dict(id=transcript["id"], family=transcript["family"],
                   category=transcript.get("category"), sample=int(transcript["sample"]),
                   decision_batch=f"{transcript['id']}:{'stage' if protocol.endswith('singleturn-v1') else 'round'}-{stage}",
                   answer_protocol=protocol)
    context["condition"] = (f"fixed-stage-{stage}" if protocol.endswith("singleturn-v1")
                            else f"round-{stage}")
    return context


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Completed run with config.json and transcripts.jsonl")
    parser.add_argument("--out", type=Path, help="New output directory (default: RUN_DIR/activation-replay)")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--dtype", choices=("auto", "float32", "float16", "bfloat16"))
    parser.add_argument("--layers", default="all", help="Layer indices, last, or all")
    parser.add_argument("--types", default="resid_post", help="Activation hooks to capture")
    parser.add_argument("--positions", default="confidence-tokens",
                        help="Capture positions; confidence-tokens saves both reported confidence values")
    args = parser.parse_args(argv)
    run_dir = args.run_dir.resolve()
    transcripts_path = run_dir / "transcripts.jsonl"
    if not transcripts_path.is_file():
        raise FileNotFoundError(f"Missing {transcripts_path}")
    output = (args.out or run_dir / "activation-replay").resolve()
    output.mkdir(parents=True, exist_ok=False)
    transcripts = read_jsonl(transcripts_path)
    if not transcripts:
        raise ValueError("No transcripts found")
    saved = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    protocol = saved.get("protocol")
    if protocol not in ("bet-on-yourself-singleturn-v1", "bet-on-yourself-multiturn-v1"):
        raise ValueError(f"Unsupported run protocol: {protocol}")
    config = load_config(run_dir, args)
    model = TransformersModel(config)
    try:
        model.begin_run(output)
        failures = []
        # Single-turn runs save one transcript per sample; multi-turn runs save
        # only the branch-0 history per stage. Reuse each saved prefix for every
        # sample at that exact case and round.
        prefixes = {}
        for transcript in transcripts:
            if not isinstance(transcript.get("stage"), int):
                continue
            conversation = transcript.get("conversation") or []
            target = next((i for i in range(len(conversation) - 1, -1, -1)
                           if conversation[i].get("role") == "assistant"), None)
            if target is not None:
                prefixes[(transcript["id"], int(transcript["stage"]))] = conversation[:target]
        samples_path = run_dir / "samples.jsonl"
        samples = read_jsonl(samples_path) if samples_path.exists() else []
        if not samples:
            raise ValueError("No samples.jsonl responses found to replay")
        for index, row in enumerate(tqdm(samples, desc="Replay activations", unit="response")):
            stage = int(row.get("stage", 0))
            messages = prefixes.get((row["id"], stage))
            if messages is None:
                failures.append(dict(index=index, id=row.get("id"), stage=stage,
                                     reason="missing_saved_prompt_prefix"))
                continue
            response = row.get("response") or ""
            if not response.strip():
                failures.append(dict(index=index, id=row.get("id"), sample=row.get("sample"), reason="empty_response"))
                continue
            context = activation_context(row, protocol)
            try:
                model.replay_response(messages, response, context=context,
                                      temperature=float(saved.get("temperature", 0.0)),
                                      seed=row.get("seed"))
            except (ValueError, RuntimeError) as exc:
                failures.append(dict(index=index, id=row.get("id"), sample=row.get("sample"),
                                     reason=str(exc)))
        write_json(output / "replay_manifest.json", dict(source_run=str(run_dir),
            source_protocol=protocol, records=len(samples), captured=len(samples) - len(failures), failed_records=failures,
            model=config.model, revision=config.revision, layers=args.layers,
            types=args.types, positions=args.positions,
            note="Teacher-forced replay uses the saved transcript text and recorded model revision."))
        if failures:
            print(f"Captured {len(samples) - len(failures)} of {len(samples)} responses; see replay_manifest.json")
    finally:
        model.close()
    print(f"Saved replayed activations under {output / 'activations'}")
    return output


if __name__ == "__main__":
    main()
