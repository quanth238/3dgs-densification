#!/usr/bin/env python3
import argparse
import csv
import math
import os
from collections import defaultdict

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False


def parse_summary(path):
    vals = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if ":" not in line:
                continue
            k, v = line.strip().split(":", 1)
            vals[k.strip()] = v.strip()
    return vals


def fnum(x):
    try:
        return float(x)
    except Exception:
        return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", required=True, type=str)
    ap.add_argument("--out_csv", default="", type=str)
    ap.add_argument("--out_plot", default="", type=str)
    args = ap.parse_args()

    base_dir = args.base_dir
    out_csv = args.out_csv or os.path.join(base_dir, "new_atom_sweep_summary.csv")
    out_plot = args.out_plot or os.path.join(base_dir, "new_atom_sweep_summary.png")

    rows = []
    for root, _, files in os.walk(base_dir):
        if "summary.txt" not in files:
            continue
        path = os.path.join(root, "summary.txt")
        vals = parse_summary(path)
        if not vals:
            continue
        rows.append({
            "iteration": vals.get("iteration", ""),
            "view_index": vals.get("view_index", ""),
            "alpha0": vals.get("alpha0", ""),
            "epsilon": vals.get("epsilon", ""),
            "scale_factor": vals.get("scale_factor", ""),
            "depth_multipliers": vals.get("depth_multipliers", ""),
            "num_candidates": vals.get("num_candidates", ""),
            "num_samples": vals.get("num_samples", ""),
            "depth_source": vals.get("depth_source", ""),
            "pcd_coverage": vals.get("pcd_coverage", ""),
            "visible_ratio": vals.get("visible_ratio", ""),
            "r2": vals.get("r2", ""),
            "pearson": vals.get("pearson", ""),
            "spearman": vals.get("spearman", ""),
            "topk_overlap": vals.get("topk_overlap", ""),
        })

    # Write CSV with sections.
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("# per_run\n")
        writer = csv.writer(f)
        writer.writerow([
            "iteration","view_index","alpha0","epsilon","scale_factor","depth_multipliers",
            "num_candidates","num_samples","depth_source","pcd_coverage",
            "visible_ratio","r2","pearson","spearman","topk_overlap"
        ])
        for r in rows:
            writer.writerow([
                r["iteration"], r["view_index"], r["alpha0"], r["epsilon"], r["scale_factor"], r["depth_multipliers"],
                r["num_candidates"], r["num_samples"], r["depth_source"], r["pcd_coverage"],
                r["visible_ratio"], r["r2"], r["pearson"], r["spearman"], r["topk_overlap"]
            ])

        # Aggregate by iteration
        f.write("\n# by_iteration\n")
        writer.writerow([
            "iteration","mean_r2","std_r2","mean_pearson","std_pearson",
            "mean_spearman","std_spearman","mean_visible","std_visible",
            "mean_topk","std_topk"
        ])
        by_iter = defaultdict(list)
        for r in rows:
            by_iter[r["iteration"]].append(r)

        for it in sorted(by_iter.keys(), key=lambda x: int(x) if str(x).lstrip("-").isdigit() else 1_000_000):
            items = by_iter[it]
            r2 = [fnum(x["r2"]) for x in items]
            pear = [fnum(x["pearson"]) for x in items]
            spear = [fnum(x["spearman"]) for x in items]
            vis = [fnum(x["visible_ratio"]) for x in items]
            topk = [fnum(x["topk_overlap"]) for x in items]

            def mstd(a):
                a = [v for v in a if not math.isnan(v)]
                if not a:
                    return float("nan"), float("nan")
                return float(np.mean(a)), float(np.std(a))

            mr2, sr2 = mstd(r2)
            mp, sp = mstd(pear)
            ms, ss = mstd(spear)
            mv, sv = mstd(vis)
            mt, st = mstd(topk)
            writer.writerow([it, mr2, sr2, mp, sp, ms, ss, mv, sv, mt, st])

    if HAS_MPL:
        # Build plot for iteration aggregates
        iters = []
        stats = []
        for it, items in sorted(by_iter.items(), key=lambda x: int(x[0]) if str(x[0]).lstrip("-").isdigit() else 1_000_000):
            if not str(it).lstrip("-").isdigit():
                continue
            r2 = [fnum(x["r2"]) for x in items]
            pear = [fnum(x["pearson"]) for x in items]
            spear = [fnum(x["spearman"]) for x in items]
            vis = [fnum(x["visible_ratio"]) for x in items]
            def mstd(a):
                a = [v for v in a if not math.isnan(v)]
                if not a:
                    return float("nan"), float("nan")
                return float(np.mean(a)), float(np.std(a))
            stats.append((mstd(r2), mstd(pear), mstd(spear), mstd(vis)))
            iters.append(int(it))

        if iters:
            fig, axs = plt.subplots(2, 2, figsize=(10, 8))
            titles = ["R^2 vs iteration", "Pearson r vs iteration", "Spearman ρ vs iteration", "Visible ratio vs iteration"]
            for ax, title, idx in zip(axs.flatten(), titles, range(4)):
                means = [s[idx][0] for s in stats]
                stds = [s[idx][1] for s in stats]
                ax.errorbar(iters, means, yerr=stds, marker="o")
                ax.set_title(title)
                ax.set_xlabel("iteration")
            fig.tight_layout()
            fig.savefig(out_plot, dpi=150)

    print(f"Wrote {out_csv}")
    if HAS_MPL:
        print(f"Wrote {out_plot}")


if __name__ == "__main__":
    main()
