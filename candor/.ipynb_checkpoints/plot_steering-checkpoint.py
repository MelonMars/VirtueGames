"""Plot a completed candor sweep and quantify paired changes by scenario family."""
import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

VALID = {"candid", "sycophantic", "omission_other"}
LABELS = ["candid", "sycophantic", "omission_other", "unclear", "judge_error"]


def pair_key(row):
    return row["id"], row["condition"], row["sample"]


def pair_rows(rows):
    baselines, groups, seen = {}, defaultdict(list), set()
    for row in rows:
        if row["label"] not in LABELS or not np.isfinite(row["alpha"]):
            raise ValueError("Unknown outcome or nonfinite strength")
        key = (*pair_key(row), row["hook"], row["alpha"])
        if key in seen:
            raise ValueError(f"Duplicate response: {key}")
        seen.add(key)
        if row["hook"] is None:
            if pair_key(row) in baselines:
                raise ValueError("Duplicate baseline")
            baselines[pair_key(row)] = row
    for row in rows:
        if row["hook"] is None:
            continue
        base = baselines.get(pair_key(row))
        if base is None or any(row[k] != base[k] for k in ("seed", "split_group")):
            raise ValueError("Missing or mismatched baseline")
        if row["alpha"] == 0 and (row["response"] != base["response"] or row["label"] != base["label"]):
            raise ValueError("Zero-strength response or label differs from baseline")
        groups[row["condition"], row["hook"], row["alpha"]].append(dict(row, baseline_label=base["label"]))
    if not groups:
        raise ValueError("No steering responses")
    for (condition, _, _), group in groups.items():
        expected = {k for k in baselines if k[1] == condition}
        if {pair_key(r) for r in group} != expected:
            raise ValueError("Intervention has incomplete paired coverage")
    return groups


def estimate(rows, n_bootstrap=2000, seed=42):
    counts = Counter(r["label"] for r in rows)
    pairs = [r for r in rows if r["label"] in VALID and r["baseline_label"] in VALID]
    # Samples -> item means -> family means; variants cannot dominate a family.
    items = defaultdict(list)
    for r in pairs:
        items[r["split_group"], r["id"]].append(
            int(r["label"] == "candid") - int(r["baseline_label"] == "candid"))
    families = defaultdict(list)
    for (family, _), values in items.items():
        families[family].append(float(np.mean(values)))
    values = np.array([np.mean(families[g]) for g in sorted(families)])
    interval = [None, None]
    if len(values) >= 2 and n_bootstrap:
        rng = np.random.default_rng(seed)
        # Batch bootstrap draws to bound memory for large family banks.
        draws = []
        for start in range(0, n_bootstrap, 256):
            draws.extend(rng.choice(values, (min(256, n_bootstrap-start), len(values))).mean(axis=1))
        interval = np.quantile(draws, [.025, .975]).tolist()
    n_scored = sum(counts[k] for k in VALID)
    baseline_candid = sum(r["baseline_label"] == "candid" for r in pairs)
    baseline_sycophantic = sum(r["baseline_label"] == "sycophantic" for r in pairs)
    recovered = sum(r["baseline_label"] == "sycophantic" and r["label"] == "candid" for r in pairs)
    regressed = sum(r["baseline_label"] == "candid" and r["label"] != "candid" for r in pairs)
    return dict(n=len(rows), n_scored=n_scored, paired_n=len(pairs),
        paired_excluded=len(rows)-len(pairs), n_families=len(values),
        outcomes={k: counts[k] for k in LABELS},
        candor_rate=counts["candid"]/n_scored if n_scored else None,
        paired_baseline_candor_rate=baseline_candid/len(pairs) if pairs else None,
        paired_steered_candor_rate=sum(r["label"] == "candid" for r in pairs)/len(pairs) if pairs else None,
        family_candor_change=float(values.mean()) if len(values) else None,
        ci_low=interval[0], ci_high=interval[1],
        recovery_rate=recovered/baseline_sycophantic if baseline_sycophantic else None,
        regression_rate=regressed/baseline_candid if baseline_candid else None,
        baseline_sycophantic=baseline_sycophantic, baseline_candid=baseline_candid,
        recovered=recovered, regressed=regressed,
        transitions={f"{a}->{b}": sum(r["baseline_label"] == a and r["label"] == b for r in rows)
                     for a in LABELS for b in LABELS})


def plot(metrics, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    conditions = sorted({r["condition"] for r in metrics})
    hooks = sorted({r["hook"] for r in metrics}, key=lambda h: int(h.split('.')[1]))
    alphas = sorted({r["alpha"] for r in metrics})
    fig, axes = plt.subplots(1, len(conditions), figsize=(7*len(conditions), max(4, len(hooks)*.28)), squeeze=False, constrained_layout=True)
    for ax, condition in zip(axes[0], conditions):
        table = {(r["hook"], r["alpha"]): r for r in metrics if r["condition"] == condition}
        grid = np.array([[table.get((h,a), {}).get("family_candor_change") for a in alphas] for h in hooks], dtype=float)*100
        im = ax.imshow(grid, aspect="auto", cmap="RdBu", vmin=-100, vmax=100)
        ax.set(xticks=range(len(alphas)), xticklabels=alphas, yticks=range(len(hooks)),
               yticklabels=["Layer " + h.split('.')[1] for h in hooks], xlabel="Raw direction multiplier", title=condition.capitalize())
        fig.colorbar(im, ax=ax, label="Candor change vs baseline (percentage points)")
    fig.savefig(out / "candor_heatmap.png", dpi=180)
    plt.close(fig)
    # One readable page per layer rather than dozens of overlapping curves.
    from matplotlib.backends.backend_pdf import PdfPages
    with PdfPages(out / "layer_sweeps.pdf") as pdf:
        for hook in hooks:
            fig, axes = plt.subplots(2, len(conditions), figsize=(7*len(conditions), 8), squeeze=False, constrained_layout=True)
            for col, condition in enumerate(conditions):
                rows = sorted([r for r in metrics if r["hook"] == hook and r["condition"] == condition], key=lambda r:r["alpha"])
                x = [r["alpha"] for r in rows]
                num = lambda key: np.array([r[key] if r[key] is not None else np.nan for r in rows])*100
                ax = axes[0, col]
                ax.plot(x, num("family_candor_change"), marker="o", color="#176b87")
                ax.fill_between(x, num("ci_low"), num("ci_high"), alpha=.2, color="#176b87", label="Pointwise 95% family bootstrap")
                ax.axhline(0, color="gray", linestyle="--")
                ax.set(title=condition.capitalize(), ylabel="Paired candor change (pp)", xlabel="Raw direction multiplier", ylim=(-105,105))
                ax.legend(fontsize=8)
                ax = axes[1, col]
                for label in LABELS:
                    ax.plot(x, [100*r["outcomes"][label]/r["n"] for r in rows], marker=".", label=label)
                ax.set(xlabel="Raw direction multiplier", ylabel="Share of all responses (%)", ylim=(-2,102))
                ax.legend(fontsize=8)
            fig.suptitle(f"Layer {hook.split('.')[1]}: candor steering")
            pdf.savefig(fig)
            fig.savefig(out / f"layer_{hook.split('.')[1]}.png", dpi=160)
            plt.close(fig)


def analyze(run_dir, out=None, n_bootstrap=2000, seed=42):
    import matplotlib  # Check plotting dependency before writing any artifacts.
    root = Path(run_dir)
    if n_bootstrap < 0:
        raise ValueError("bootstrap must be nonnegative")
    if json.loads((root / "status.json").read_text())["status"] != "complete":
        raise ValueError("Plot a completed sweep")
    rows = [json.loads(line) for line in (root / "samples.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    groups = pair_rows(rows)
    metrics = [dict(condition=c, hook=h, alpha=a, **estimate(group, n_bootstrap, seed))
               for (c,h,a), group in tqdm(sorted(groups.items()), desc="Analyze sweep", unit="setting")]
    candidates = [r for r in metrics if r["condition"] == "treatment" and r["alpha"] != 0 and r["family_candor_change"] is not None]
    candidates.sort(key=lambda r: (-r["family_candor_change"], abs(r["alpha"]), r["hook"]))
    best = candidates[0] if candidates else None
    lines = ["# Candor steering analysis", "", f"Analyzed {len(metrics)} settings from `{root.resolve()}`.", ""]
    if best:
        lines += [f"Highest observed treatment gain: **{best['hook']}, alpha={best['alpha']:g}**, "
                  f"{100*best['family_candor_change']:+.1f} percentage points in family-balanced candor.",
                  f"Valid pairs: {best['paired_n']}/{best['n']}; represented families: {best['n_families']}.", ""]
        if best["ci_low"] is not None:
            lines += [f"Pointwise 95% family bootstrap interval: [{100*best['ci_low']:+.1f}, {100*best['ci_high']:+.1f}] pp.", ""]
        lines += [f"Recovered {best['recovered']}/{best['baseline_sycophantic']} baseline sycophantic responses; "
                  f"regressed on {best['regressed']}/{best['baseline_candid']} baseline candid responses.", ""]
        control = next((r for r in metrics if r["condition"] == "control" and r["hook"] == best["hook"] and r["alpha"] == best["alpha"]), None)
        if control and control["family_candor_change"] is not None:
            lines += [f"At the same setting, neutral-prompt candor changed by {100*control['family_candor_change']:+.1f} pp.", ""]
        if best["family_candor_change"] <= 0:
            lines += ["No tested nonzero setting improved the family-balanced treatment point estimate.", ""]
    else:
        lines += ["No treatment setting has enough valid pairs to rank.", ""]
    lines += ["## Interpretation", "",
        "Changes compare each response against the same prompt and seed at baseline. Samples are averaged within items, then items within families, then families equally. Unclear and judge-error pairs are excluded; outcome charts retain every response. Different exclusion patterns can affect comparisons.", "",
        "Intervals resample scenario families and are pointwise, not corrected for testing many settings. The highest observed setting is a development candidate, not evidence of a final optimized effect. Few families give unreliable intervals; no interval is reported with fewer than two. These intervals do not measure judge error or new-generation uncertainty within a family.", "",
        "Positive treatment change with little neutral regression supports a useful intervention on these cases. Inspect recovery, regression, omission, and judge-error rates together. Compare positive and negative strengths to assess whether the effect follows the learned direction. Confirm any selected setting on fresh scenario families. This flawed-work bank cannot establish whether steering causes false criticism of sound work.", ""]
    out = Path(out) if out else root / "steering_plots"
    out.mkdir(parents=True, exist_ok=False)
    (out / "metrics.json").write_text(json.dumps(dict(source=str(root.resolve()), bootstrap=n_bootstrap, seed=seed, settings=metrics), indent=2)+"\n", encoding="utf-8")
    flat = [{k:v for k,v in r.items() if not isinstance(v, dict)} for r in metrics]
    with (out / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat[0])); writer.writeheader(); writer.writerows(flat)
    (out / "analysis.md").write_text("\n".join(lines), encoding="utf-8")
    plot(metrics, out)
    print(f"Saved plots and analysis: {out}")
    return metrics


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir", type=Path, help="candor_steering/run-XX directory")
    p.add_argument("--out", type=Path)
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    analyze(args.run_dir, args.out, args.bootstrap, args.seed)
