#!/usr/bin/env python3
#
# Aggregate multiple sweep_summary.csv files into one summary CSV + plot.
#

import argparse
import os
import re
from typing import Dict, List, Tuple

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False


def parse_summary_csv(path: str):
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines()]

    config_line = ""
    by_epsilon_rows = []
    section = None

    for line in lines:
        if not line:
            continue
        if line.startswith("# config"):
            config_line = line
            continue
        if line.startswith("#"):
            if "by_epsilon" in line:
                section = "by_epsilon"
            else:
                section = None
            continue
        if section == "by_epsilon":
            if line.startswith("epsilon,"):
                continue
            parts = line.split(",")
            if len(parts) < 9:
                continue
            by_epsilon_rows.append(
                {
                    "epsilon": float(parts[0]),
                    "mean_r2": float(parts[1]),
                    "std_r2": float(parts[2]),
                    "mean_pearson": float(parts[3]),
                    "std_pearson": float(parts[4]),
                    "mean_spearman": float(parts[5]),
                    "std_spearman": float(parts[6]),
                    "mean_topk": float(parts[7]),
                    "std_topk": float(parts[8]),
                }
            )

    return config_line, by_epsilon_rows


def extract_dataset_iter(path: str) -> Tuple[str, str]:
    # Expected: .../oracle_verify/<dataset>/iter_<iter>/sweep_summary.csv
    parts = path.replace("\\", "/").split("/")
    dataset = "unknown"
    iteration = "unknown"
    for i in range(len(parts) - 1):
        if parts[i] == "oracle_verify" and i + 2 < len(parts):
            dataset = parts[i + 1]
            iter_part = parts[i + 2]
            if iter_part.startswith("iter_"):
                iteration = iter_part.replace("iter_", "")
    return dataset, iteration


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_list", required=True, type=str)
    parser.add_argument("--out_dir", required=True, type=str)
    args = parser.parse_args()

    with open(args.input_list, "r", encoding="utf-8") as f:
        paths = [line.strip() for line in f.readlines() if line.strip()]

    rows = []
    configs = {}
    for path in paths:
        if not os.path.isfile(path):
            continue
        dataset, iteration = extract_dataset_iter(path)
        config, by_eps = parse_summary_csv(path)
        configs[f"{dataset}/iter_{iteration}"] = config
        for row in by_eps:
            out = {"dataset": dataset, "iteration": iteration}
            out.update(row)
            rows.append(out)

    os.makedirs(args.out_dir, exist_ok=True)
    out_csv = os.path.join(args.out_dir, "cross_dataset_summary.csv")
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("# cross-dataset summary (from sweep_summary.csv by_epsilon)\n")
        for k, v in configs.items():
            if v:
                f.write(f"# {k} {v}\n")
        f.write("dataset,iteration,epsilon,mean_r2,std_r2,mean_pearson,std_pearson,mean_spearman,std_spearman,mean_topk,std_topk\n")
        for r in rows:
            f.write(
                f"{r['dataset']},{r['iteration']},{r['epsilon']},{r['mean_r2']},{r['std_r2']},"
                f"{r['mean_pearson']},{r['std_pearson']},{r['mean_spearman']},{r['std_spearman']},"
                f"{r['mean_topk']},{r['std_topk']}\n"
            )

    if not HAS_MPL:
        print(f"Wrote {out_csv} (matplotlib not available for plot).")
        return

    # Plot: one figure, 2x2 panels, lines per dataset/iter
    # Group by dataset+iteration
    grouped = {}
    for r in rows:
        key = f"{r['dataset']}/iter_{r['iteration']}"
        grouped.setdefault(key, []).append(r)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    ax_r2, ax_pr = axes[0]
    ax_sp, ax_tk = axes[1]

    for key, items in grouped.items():
        items = sorted(items, key=lambda x: x["epsilon"])
        eps = [x["epsilon"] for x in items]
        ax_r2.plot(eps, [x["mean_r2"] for x in items], marker="o", label=key)
        ax_pr.plot(eps, [x["mean_pearson"] for x in items], marker="o", label=key)
        ax_sp.plot(eps, [x["mean_spearman"] for x in items], marker="o", label=key)
        ax_tk.plot(eps, [x["mean_topk"] for x in items], marker="o", label=key)

    ax_r2.set_title("R2 vs epsilon")
    ax_pr.set_title("Pearson r vs epsilon")
    ax_sp.set_title("Spearman rho vs epsilon")
    ax_tk.set_title("Top-K overlap vs epsilon")
    for ax in [ax_r2, ax_pr, ax_sp, ax_tk]:
        ax.set_xlabel("epsilon")
        ax.grid(True, alpha=0.2)

    ax_r2.set_ylabel("R2")
    ax_pr.set_ylabel("Pearson r")
    ax_sp.set_ylabel("Spearman rho")
    ax_tk.set_ylabel("Top-K")

    # Put legend outside
    handles, labels = ax_r2.get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right", fontsize=8)
    plt.tight_layout(rect=[0, 0, 0.82, 1])

    out_png = os.path.join(args.out_dir, "cross_dataset_summary.png")
    plt.savefig(out_png, dpi=200)
    plt.close(fig)

    print(f"Wrote {out_csv} and {out_png}")


if __name__ == "__main__":
    main()
