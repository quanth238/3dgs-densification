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


def fnum(x):
    try:
        return float(x)
    except Exception:
        return float("nan")


def parse_summary(path):
    per_run = None
    by_k = []
    section = None
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("#"):
            if "per_run" in line:
                section = "per_run"
                # next line is header, next line is row
                if i + 2 < len(lines):
                    header_line = lines[i + 1]
                    row_line = lines[i + 2]
                    reader = csv.reader([header_line, row_line])
                    header = next(reader)
                    values = next(reader)
                    per_run = dict(zip(header, values))
                    i += 3
                    continue
            elif "by_K" in line:
                section = "by_k"
                i += 1
                # consume header line
                if i < len(lines) and lines[i].startswith("K,"):
                    i += 1
                # consume data lines until next section
                while i < len(lines) and not lines[i].startswith("#"):
                    row = next(csv.reader([lines[i]]))
                    if len(row) >= 5:
                        by_k.append({
                            "K": row[0],
                            "delta_mean": row[1],
                            "delta_std": row[2],
                            "min_g_mean": row[3],
                            "min_g_std": row[4],
                        })
                    i += 1
                continue
            else:
                section = None
        i += 1
    return per_run, by_k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", required=True, type=str)
    ap.add_argument("--out_csv", default="", type=str)
    ap.add_argument("--out_plot", default="", type=str)
    args = ap.parse_args()

    base_dir = args.base_dir
    out_csv = args.out_csv or os.path.join(base_dir, "delta_oracle_sweep_summary.csv")
    out_plot = args.out_plot or os.path.join(base_dir, "delta_oracle_sweep_summary.png")

    per_runs = []
    by_k_rows = []

    for root, _, files in os.walk(base_dir):
        if "delta_oracle_summary.csv" not in files:
            continue
        path = os.path.join(root, "delta_oracle_summary.csv")
        pr, bk = parse_summary(path)
        if pr:
            pr["__path"] = path
            per_runs.append(pr)
            for row in bk:
                row["iteration"] = pr.get("iteration", "")
                row["view_index"] = pr.get("view_index", "")
                by_k_rows.append(row)

    # Write combined CSV
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("# per_run\n")
        writer = csv.writer(f)
        writer.writerow([
            "iteration","view_index","alpha0","scale_factor","depth_multipliers",
            "depth_source","pcd_coverage","visible_ratio","g_min","g_mean",
            "tail_tau","tail_frac","num_candidates"
        ])
        for r in per_runs:
            writer.writerow([
                r.get("iteration",""), r.get("view_index",""), r.get("alpha0",""), r.get("scale_factor",""),
                r.get("depth_multipliers",""), r.get("depth_source",""), r.get("pcd_coverage",""),
                r.get("visible_ratio",""), r.get("g_min",""), r.get("g_mean",""),
                r.get("tail_tau",""), r.get("tail_frac",""), r.get("num_candidates","")
            ])

        f.write("\n# by_iteration\n")
        writer.writerow([
            "iteration","mean_visible","std_visible","mean_gmin","std_gmin",
            "mean_tail","std_tail"
        ])
        by_iter = defaultdict(list)
        for r in per_runs:
            it = r.get("iteration","")
            by_iter[it].append(r)

        for it in sorted(by_iter.keys(), key=lambda x: int(x) if str(x).lstrip("-").isdigit() else 1_000_000):
            if not str(it).lstrip("-").isdigit():
                continue
            items = by_iter[it]
            vis = [fnum(x.get("visible_ratio","nan")) for x in items]
            gmin = [fnum(x.get("g_min","nan")) for x in items]
            tail = [fnum(x.get("tail_frac","nan")) for x in items]
            def mstd(a):
                a = [v for v in a if not math.isnan(v)]
                if not a:
                    return float("nan"), float("nan")
                return float(np.mean(a)), float(np.std(a))
            mv, sv = mstd(vis)
            mg, sg = mstd(gmin)
            mt, st = mstd(tail)
            writer.writerow([it, mv, sv, mg, sg, mt, st])

        f.write("\n# by_iteration_K\n")
        writer.writerow(["iteration","K","mean_delta","std_delta","mean_min_g","std_min_g"])
        by_iter_k = defaultdict(list)
        for r in by_k_rows:
            it = r.get("iteration","")
            k = r.get("K","")
            by_iter_k[(it, k)].append(r)

        for (it, k) in sorted(by_iter_k.keys(), key=lambda x: (int(x[0]) if str(x[0]).lstrip("-").isdigit() else 1_000_000, int(x[1]) if str(x[1]).isdigit() else 1_000_000)):
            if not str(it).lstrip("-").isdigit():
                continue
            items = by_iter_k[(it, k)]
            d = [fnum(x.get("delta_mean","nan")) for x in items]
            mg = [fnum(x.get("min_g_mean","nan")) for x in items]
            def mstd(a):
                a = [v for v in a if not math.isnan(v)]
                if not a:
                    return float("nan"), float("nan")
                return float(np.mean(a)), float(np.std(a))
            md, sd = mstd(d)
            mmg, smg = mstd(mg)
            writer.writerow([it, k, md, sd, mmg, smg])

    if HAS_MPL:
        # Plot: delta(K) per iteration + visible/gmin/tail vs iteration
        # Collect by_iteration_K
        iter_keys = sorted({k[0] for k in by_iter_k.keys() if str(k[0]).lstrip("-").isdigit()}, key=lambda x: int(x))
        k_values = sorted({int(k[1]) for k in by_iter_k.keys() if str(k[1]).isdigit()})

        fig, axs = plt.subplots(2, 2, figsize=(11, 8))
        ax0 = axs[0, 0]
        for it in iter_keys:
            means = []
            stds = []
            for K in k_values:
                items = by_iter_k.get((it, str(K)), [])
                vals = [fnum(x.get("delta_mean","nan")) for x in items]
                vals = [v for v in vals if not math.isnan(v)]
                means.append(np.mean(vals) if vals else np.nan)
                stds.append(np.std(vals) if vals else np.nan)
            ax0.errorbar(k_values, means, yerr=stds, marker="o", label=f"iter {it}")
        ax0.set_title("δ(K) vs K")
        ax0.set_xlabel("K")
        ax0.set_ylabel("delta")
        ax0.legend()

        # Visible ratio vs iteration
        ax1 = axs[0, 1]
        iters = []
        vis_mean = []
        vis_std = []
        gmin_mean = []
        gmin_std = []
        tail_mean = []
        tail_std = []
        for it in iter_keys:
            items = by_iter[it]
            vis = [fnum(x.get("visible_ratio","nan")) for x in items if not math.isnan(fnum(x.get("visible_ratio","nan")))]
            gmin = [fnum(x.get("g_min","nan")) for x in items if not math.isnan(fnum(x.get("g_min","nan")))]
            tail = [fnum(x.get("tail_frac","nan")) for x in items if not math.isnan(fnum(x.get("tail_frac","nan")))]
            iters.append(int(it))
            vis_mean.append(np.mean(vis) if vis else np.nan)
            vis_std.append(np.std(vis) if vis else np.nan)
            gmin_mean.append(np.mean(gmin) if gmin else np.nan)
            gmin_std.append(np.std(gmin) if gmin else np.nan)
            tail_mean.append(np.mean(tail) if tail else np.nan)
            tail_std.append(np.std(tail) if tail else np.nan)

        ax1.errorbar(iters, vis_mean, yerr=vis_std, marker="o")
        ax1.set_title("Visible ratio vs iteration")
        ax1.set_xlabel("iteration")
        ax1.set_ylabel("visible_ratio")

        ax2 = axs[1, 0]
        ax2.errorbar(iters, gmin_mean, yerr=gmin_std, marker="o")
        ax2.set_title("g_min vs iteration")
        ax2.set_xlabel("iteration")
        ax2.set_ylabel("g_min")

        ax3 = axs[1, 1]
        ax3.errorbar(iters, tail_mean, yerr=tail_std, marker="o")
        ax3.set_title("tail_frac vs iteration")
        ax3.set_xlabel("iteration")
        ax3.set_ylabel("tail_frac")

        fig.tight_layout()
        fig.savefig(out_plot, dpi=150)

    print(f"Wrote {out_csv}")
    if HAS_MPL:
        print(f"Wrote {out_plot}")


if __name__ == "__main__":
    main()
