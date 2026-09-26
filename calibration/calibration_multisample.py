"""calibration-variance experiment. Execute this file directly."""
import sys
from pathlib import Path

# Direct script execution needs the repository root for shared helpers.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import statistics
import time
from datetime import datetime, timezone
from tqdm import tqdm

from llm import generate
from runs import jsonl_writer, write_json
from calibration.run_calibration import SYSTEM_PROMPT, STAKE, parse_response


def run(llm, questions, args, run_dir):
    samples_path = run_dir / "samples.jsonl"
    variance_path = run_dir / "variance.jsonl"
    agg = []
    t0 = time.time()
    total = len(questions) * args.n_samples
    with jsonl_writer(samples_path) as write_sample:
        with tqdm(total=total, desc="variance", unit="samp") as bar:
            for q in questions:
                key = q["answer"]
                samples = []
                for i in range(args.n_samples):
                    raw = generate(llm, SYSTEM_PROMPT, q["question"],
                        temperature=args.temperature, seed=args.seed + i,
                        reset=True, max_tokens=args.max_tokens,
                        activation_context={"id": q["id"], "condition": "calibration", "sample": i})
                    answer, conf = parse_response(raw)
                    parse_ok = answer is not None and conf is not None
                    correct = (answer == key) if answer is not None else None
                    s = {
                        "id": q["id"], "sample": i, "response_raw": raw,
                        "answer_parsed": answer, "confidence": conf,
                        "correct": correct, "parse_ok": parse_ok,
                    }
                    write_sample(s)
                    samples.append(s)
                    bar.update(1)

                ok = [s for s in samples if s["parse_ok"]]
                n_ok = len(ok)
                n_yes = sum(1 for s in ok if s["answer_parsed"] is True)
                n_no = sum(1 for s in ok if s["answer_parsed"] is False)
                confs = [s["confidence"] for s in ok]
                corrects = [s["correct"] for s in ok]

                answer_flip_rate = (min(n_yes, n_no) / n_ok) if n_ok else None
                conf_mean = statistics.mean(confs) if confs else None
                conf_std = statistics.pstdev(confs) if len(confs) > 1 else 0.0
                conf_min = min(confs) if confs else None
                conf_max = max(confs) if confs else None
                accuracy = (sum(1 for c in corrects if c) / n_ok) if n_ok else None

                answer_varies = n_ok > 0 and n_yes > 0 and n_no > 0
                conf_varies = conf_std >= args.conf_std_threshold
                contrastive = bool(answer_varies or conf_varies)

                a = {
                    "id": q["id"], "topic": q.get("topic"),
                    "difficulty": q.get("difficulty"), "key": key,
                    "n_samples": args.n_samples, "n_parsed": n_ok,
                    "n_yes": n_yes, "n_no": n_no,
                    "answer_flip_rate": answer_flip_rate,
                    "conf_mean": conf_mean, "conf_std": round(conf_std, 2),
                    "conf_min": conf_min, "conf_max": conf_max,
                    "accuracy": accuracy,
                    "answer_varies": answer_varies, "conf_varies": conf_varies,
                    "contrastive": contrastive,
                }
                agg.append(a)

    with jsonl_writer(variance_path) as write_variance:
        for a in agg:
            write_variance(a)

    elapsed = time.time() - t0
    usable = [a for a in agg if a["n_parsed"] > 0]
    n_contrastive = sum(1 for a in agg if a["contrastive"])
    n_answer_varies = sum(1 for a in agg if a["answer_varies"])
    n_saturated = sum(1 for a in usable if not a["contrastive"])
    mean_conf_std = (statistics.mean(a["conf_std"] for a in usable)
                     if usable else None)

    summary = {
        "n_questions": len(agg),
        "n_usable": len(usable),
        "n_contrastive": n_contrastive,
        "n_answer_flips": n_answer_varies,
        "n_saturated": n_saturated,
        "mean_conf_std": round(mean_conf_std, 2) if mean_conf_std is not None else None,
        "temperature": args.temperature,
        "n_samples": args.n_samples,
        "elapsed_sec": round(elapsed, 1),
        "finished": datetime.now(timezone.utc).isoformat(),
    }
    write_json(run_dir / "summary.json", summary)

    print(f"\nrun dir       : {run_dir}", file=sys.stderr)
    print(f"questions     : {len(agg)}  usable {len(usable)}", file=sys.stderr)
    print(f"contrastive   : {n_contrastive}  (answer flips: {n_answer_varies})", file=sys.stderr)
    print(f"saturated     : {n_saturated}  (no variation -> weak for extraction)", file=sys.stderr)
    print(f"mean conf std : {summary['mean_conf_std']}%", file=sys.stderr)
    print(f"\ncontrastive ids:\n  " +
          ", ".join(a["id"] for a in agg if a["contrastive"]), file=sys.stderr)

    return summary


def main(argv=None):
    from config import parse_config
    from llm import load_model
    from questions import load_questions
    from runs import next_run_dir, utc_now

    config, output = parse_config(argv, max_tokens=2048, out="runs_variance",
                                  variance=True, calibration=True)
    questions = load_questions(config.questions, config.difficulty)
    print(f"Loading model: {config.model}", file=sys.stderr)
    with load_model(config) as llm:
        run_dir = next_run_dir(output)
        metadata = config.metadata("calibration-variance", len(questions))
        metadata["system_prompt"] = SYSTEM_PROMPT
        metadata["stake"] = STAKE
        write_json(run_dir / "config.json", metadata)
        write_json(run_dir / "status.json", {"status": "running", "started": utc_now()})
        try:
            if config.extract_activations:
                llm.begin_run(run_dir)
            summary = run(llm, questions, config, run_dir)
        except BaseException as exc:
            write_json(run_dir / "status.json", {
                "status": "failed", "error": str(exc), "finished": utc_now()})
            raise
        write_json(run_dir / "status.json", {"status": "complete", "finished": utc_now()})
        return summary


if __name__ == "__main__":
    main()
