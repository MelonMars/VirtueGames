"""Evaluate a frozen intervention plan across games, with family-level paired metrics."""
import argparse
from collections import Counter, defaultdict
from contextlib import nullcontext
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import random

from config import RunConfig
from experiment_splits import load_manifest
from llm import generate_messages, load_model
from runs import next_run_dir, write_json, jsonl_writer, model_runtime


def family_metrics(rows, seed=42):
    """Bootstrap families, not repeated samples; errors remain explicit in coverage."""
    groups = defaultdict(list)
    for r in rows:
        if r["success"] is not None and r["baseline_success"] is not None:
            groups[r["family"]].append(int(r["success"])-int(r["baseline_success"]))
    means = [sum(v)/len(v) for v in groups.values()]
    interval = None
    if len(means) >= 2:
        rng = random.Random(seed)
        boot = sorted(sum(rng.choices(means, k=len(means)))/len(means) for _ in range(2000))
        interval = [boot[49], boot[1949]]
    return dict(n=len(rows), paired_scored=sum(map(len, groups.values())), families=len(means),
                success_rate_scored=(sum(r["success"] for r in rows if r["success"] is not None) /
                                     sum(r["success"] is not None for r in rows)) if any(r["success"] is not None for r in rows) else None,
                mean_family_success_change=sum(means)/len(means) if means else None,
                family_bootstrap_95=interval, outcomes=dict(Counter(r["outcome"] for r in rows)),
                mean_response_chars=sum(len(r["response"]) for r in rows)/len(rows) if rows else None)


def load_suite(path, partition, direction_reports):
    suite = json.loads(path.read_text())
    result = []
    seen = set()
    for entry in suite["tasks"]:
        if entry["name"] in seen:
            raise ValueError("Suite task names must be unique")
        seen.add(entry["name"])
        if entry["game"] not in ("candor", "integrity", "evidence", "calibration"):
            raise ValueError("Unknown game")
        questions_path = (path.parent / entry["questions"]).resolve()
        questions = json.loads(questions_path.read_text())["questions"]
        split_path = (path.parent / entry["splits"]).resolve()
        manifest = load_manifest(split_path, [q["id"] for q in questions])
        selected = [dict(q, family=manifest[q["id"]]["family"]) for q in questions
                    if manifest[q["id"]]["split"] == partition]
        for report in direction_reports:
            forbidden = {report["split_groups"][qid] for qid, s in report["splits"].items()
                         if s == "train" or partition == "test" and s == "validation"}
            if forbidden & {q["family"] for q in selected}:
                raise ValueError("Evaluation family overlaps direction training/development population")
        result.append(dict(entry, questions=selected, questions_sha256=hashlib.sha256(questions_path.read_bytes()).hexdigest(),
                           splits_sha256=hashlib.sha256(split_path.read_bytes()).hexdigest()))
    if not result or any(not e["questions"] for e in result):
        raise ValueError("Every suite task needs evaluation items")
    return result


def intervention_vectors(plan, plan_path, model, revision):
    import torch
    from safetensors.torch import load_file
    reports, vectors, provenance = [], {}, {}
    for name, directory in plan["directions"].items():
        directory = (plan_path.parent / directory).resolve()
        report = json.loads((directory / "report.json").read_text())
        identity = report.get("capture_identity", {})
        if identity.get("model") != model or identity.get("resolved_revision") != revision:
            raise ValueError("Direction and target must use the same model and pinned resolved revision")
        if report["source_config"].get("scoring_version") != 2 or "test_items_reserved" not in report:
            raise ValueError("Re-extract directions using scoring v2 and a three-way family split")
        path = directory / "directions.safetensors"
        vectors[name] = load_file(str(path))
        reports.append(report)
        provenance[name] = dict(directory=str(directory), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    if not vectors:
        raise ValueError("At least one direction is required")
    zero = {}
    for component in vectors.values():
        for hook, vector in component.items():
            zero[hook] = torch.zeros_like(vector)
    settings = {"baseline": {}, "zero_hook": zero}
    for setting in plan["interventions"]:
        name = setting["name"]
        if name in settings or not setting["weights"]:
            raise ValueError("Intervention names must be unique, with nonempty weights")
        combined = {}
        for component, weight in setting["weights"].items():
            if component not in vectors or not isinstance(weight, (int,float)) or not __import__("math").isfinite(weight):
                raise ValueError("Unknown component or invalid weight")
            for hook, vector in vectors[component].items():
                if vector.ndim != 1 or not torch.isfinite(vector).all():
                    raise ValueError("Expected finite one-dimensional response directions")
                if hook in combined and combined[hook].shape != vector.shape:
                    raise ValueError("Direction widths differ")
                combined[hook] = combined.get(hook, torch.zeros_like(vector)).float() + weight*vector.float()
        if all(v.norm() == 0 for v in combined.values()):
            raise ValueError("Zero intervention; baseline is already supplied")
        if "random_seed" in setting:
            generator = torch.Generator(device="cpu").manual_seed(setting["random_seed"])
            combined = {h: (lambda x: x / x.norm() * v.norm())(torch.randn(v.shape, generator=generator))
                        for h,v in sorted(combined.items())}
        settings[name] = combined
    return settings, reports, provenance


def generate_suite(model, bridge, cfg, suite, settings, site, out):
    from integrity.steering_check import ResidualAddition
    from candor.run_candor import SYSTEM_PROMPT as CANDOR_SYSTEM, build_user
    from integrity.run_integrity import SYSTEM_PROMPT as INTEGRITY_SYSTEM, SCENARIOS, build_user as integrity_user
    from calibration.run_calibration import SYSTEM_PROMPT as CALIBRATION_SYSTEM
    from evidence.run_evidence import SYSTEM as EVIDENCE_SYSTEM
    rows = []
    with jsonl_writer(out / "results.jsonl") as write:
        for task in suite:
            for i,q in enumerate(task["questions"]):
                for sample in range(cfg.n_samples):
                    seed = cfg.seed + i*cfg.n_samples + sample
                    game = task["game"]
                    initial = None
                    if game == "evidence":
                        base = [{"role":"system", "content":EVIDENCE_SYSTEM},
                                {"role":"user", "content":q["initial_evidence"]+"\n\n"+q["question"]}]
                        initial = generate_messages(model, base, seed=seed, temperature=cfg.temperature, max_tokens=cfg.max_tokens)
                        prompts = {c: base + [{"role":"assistant", "content":initial},
                                             {"role":"user", "content":q[c+"_challenge"]}] for c in ("invalid","valid")}
                    elif game == "candor":
                        prompts = {c:[{"role":"system", "content":CANDOR_SYSTEM},
                                      {"role":"user", "content":build_user(q,c=="treatment")}] for c in ("neutral","treatment")}
                    elif game == "integrity":
                        prompts = {c:[{"role":"system", "content":INTEGRITY_SYSTEM},
                                      {"role":"user", "content":integrity_user(SCENARIOS[i%len(SCENARIOS)],q["question"],"No" if q["answer"] else "Yes",c=="treatment")}] for c in ("neutral","treatment")}
                    else:
                        prompts = {"neutral":[{"role":"system", "content":CALIBRATION_SYSTEM}, {"role":"user", "content":q["question"]}]}
                    for condition,messages in prompts.items():
                        baseline = None
                        for setting,vectors in settings.items():
                            additions = {h:ResidualAddition(v,1.,site) for h,v in vectors.items()}
                            with bridge.hooks(fwd_hooks=list(additions.items())) if additions else nullcontext():
                                response = generate_messages(model, messages, seed=seed+1000000,
                                    temperature=cfg.temperature, max_tokens=cfg.max_tokens)
                            if any(not addition.calls for addition in additions.values()):
                                raise RuntimeError("Steering hook did not fire")
                            if setting == "baseline":
                                baseline = response
                            if setting == "zero_hook" and response != baseline:
                                raise RuntimeError("Zero-vector hooks did not reproduce the unhooked baseline")
                            row = dict(task=task["name"], game=game, id=q["id"], family=q["family"], sample=sample,
                                seed=seed+1000000, condition=condition, intervention=setting, response=response,
                                initial_response=initial, messages=messages)
                            rows.append(row); write(row)
    return rows


def score_suite(rows, suite, judge, cfg, out):
    from candor.score_candor import judge_review
    from rescore_runs import score_integrity, EVIDENCE_JUDGE, parse_evidence
    from scoring import judge_json, final_response, integrity_result
    from llm import generate
    from calibration.run_calibration import parse_response
    bank = {(t["name"],q["id"]):q for t in suite for q in t["questions"]}
    scored, cache = [], {}
    with jsonl_writer(out / "samples.jsonl") as write:
        for row in rows:
            q = bank[row["task"],row["id"]]
            cache_key = (row["task"],row["id"],row["condition"],row["response"])
            if cache_key not in cache:
                if row["game"] == "candor":
                    result = judge_review(judge,q,row["response"],cfg)
                    outcome = result["label"]
                    result.update(outcome=outcome, success=None if outcome=="judge_error" else outcome=="candid")
                elif row["game"] == "evidence":
                    judged = judge_json(lambda system,user:generate(judge,system,user,max_tokens=cfg.max_tokens), EVIDENCE_JUDGE,
                        dict(question=q["question"], initial_evidence=q["initial_evidence"], challenge=q[row["condition"]+"_challenge"],
                             response=final_response(row["response"])),parse_evidence)
                    result = integrity_result(row["response"],q["updated_answer"] if row["condition"]=="valid" else q["answer"],judged.get("judgment"))
                    result["judge_attempts"] = judged["judge_attempts"]
                    if "judgment" not in judged:
                        result.update(outcome="judge_error", success=None)
                    else:
                        result["success"] = result["outcome"]=="held" and judged["judgment"]["addresses_evidence"]
                else:
                    result = score_integrity(judge,cfg,q["question"],row["response"],q["answer"])
                    result["success"] = None if result["outcome"]=="judge_error" else result["outcome"]=="held"
                    if row["game"] == "calibration":
                        _, confidence = parse_response(row["response"])
                        result["confidence"] = confidence
                        result["brier"] = ((confidence/100-int(result["correct"]))**2
                            if confidence is not None and confidence >= 50 and result["outcome"] in ("held","caved_flip") else None)
                cache[cache_key] = result
            result = dict(row, **cache[cache_key])
            scored.append(result); write(result)
    baseline = {(r["task"],r["id"],r["condition"],r["sample"]):r for r in scored if r["intervention"]=="baseline"}
    groups = defaultdict(list)
    for r in scored:
        b = baseline[r["task"],r["id"],r["condition"],r["sample"]]
        r["baseline_success"] = b["success"]
        r["baseline_brier"] = b.get("brier")
        groups[r["task"],r["condition"],r["intervention"]].append(r)
    summary = []
    for (task,condition,setting), group in groups.items():
        result = dict(task=task,condition=condition,intervention=setting,**family_metrics(group))
        brier = [r["brier"]-r["baseline_brier"] for r in group if r.get("brier") is not None and r.get("baseline_brier") is not None]
        if group[0]["game"]=="calibration":
            result.update(paired_brier_n=len(brier), mean_brier_change=sum(brier)/len(brier) if brier else None)
        summary.append(result)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suite",type=Path,required=True); p.add_argument("--plan",type=Path,required=True)
    p.add_argument("--partition",choices=("validation","test"),default="validation")
    p.add_argument("--model",required=True); p.add_argument("--revision",required=True)
    p.add_argument("--judge-model",required=True); p.add_argument("--judge-revision",required=True)
    p.add_argument("--device",choices=("auto","cpu","cuda"),default="auto")
    p.add_argument("--n-samples",type=int,default=4); p.add_argument("--temperature",type=float,default=.7)
    p.add_argument("--max-tokens",type=int,default=1024); p.add_argument("--seed",type=int,default=10000)
    p.add_argument("--site",choices=("decode","prompt"),default="decode")
    p.add_argument("--out",type=Path,default=Path("runs_transfer"))
    a=p.parse_args()
    plan=json.loads(a.plan.read_text())
    if a.partition=="test" and plan.get("stage")!="final":
        p.error("Test partition requires a plan marked stage=final after development selection")
    settings,reports,provenance=intervention_vectors(plan,a.plan,a.model,a.revision)
    suite=load_suite(a.suite,a.partition,reports)
    cfg=RunConfig(model=a.model,revision=a.revision,questions=a.suite,backend="transformers",device=a.device,
        thinking="off",n_samples=a.n_samples,temperature=a.temperature,max_tokens=a.max_tokens,seed=a.seed)
    out=next_run_dir(a.out)
    write_json(out/"config.json",dict(target=cfg.metadata("transfer-variance",sum(len(t["questions"]) for t in suite)),
        plan=plan,partition=a.partition,suite=suite,directions=provenance,site=a.site,
        plan_sha256=hashlib.sha256(a.plan.read_bytes()).hexdigest(),judge_model=a.judge_model,judge_revision=a.judge_revision))
    write_json(out/"status.json",dict(status="running"))
    try:
        from transformer_lens.model_bridge import TransformerBridge
        with load_model(cfg) as target:
            write_json(out/"target_runtime.json",model_runtime(target))
            bridge=TransformerBridge.boot_transformers(a.model,hf_model=target.model,tokenizer=target.tokenizer,
                device=target.model.device,dtype=target.model.dtype,revision=a.revision)
            bridge.eval()
            if any(h not in bridge.hook_dict or v.numel()!=bridge.cfg.d_model for vectors in settings.values() for h,v in vectors.items()):
                raise ValueError("Hook or width mismatch")
            rows=generate_suite(target,bridge,cfg,suite,settings,a.site,out)
            del bridge
        judge_cfg=replace(cfg,model=a.judge_model,revision=a.judge_revision,max_tokens=1024,n_ctx=8192)
        with load_model(judge_cfg) as judge:
            write_json(out/"judge_runtime.json",model_runtime(judge))
            summary=score_suite(rows,suite,judge,judge_cfg,out)
        write_json(out/"summary.json",summary)
        write_json(out/"status.json",dict(status="complete"))
    except BaseException as exc:
        write_json(out/"status.json",dict(status="failed",error=str(exc)))
        raise
    print(out)


if __name__=="__main__":
    main()
