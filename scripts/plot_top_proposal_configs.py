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


def parse_delta_summary(path):
    by_k = []
    if not os.path.isfile(path):
        return by_k
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("#") and "by_K" in line:
            i += 1
            if i < len(lines) and lines[i].startswith("K,"):
                i += 1
            while i < len(lines) and not lines[i].startswith("#"):
                row = next(csv.reader([lines[i]]))
                if len(row) >= 5:
                    by_k.append({
                        "K": int(float(row[0])),
                        "delta_mean": fnum(row[1]),
                        "delta_std": fnum(row[2]),
                    })
                i += 1
            continue
        i += 1
    return by_k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="proposal_quality_sweep.csv")
    ap.add_argument("--top_n", default=6, type=int)
    ap.add_argument("--out_dir", default="", type=str)
    args = ap.parse_args()

    in_path = args.input
    out_dir = args.out_dir or os.path.dirname(in_path)
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    with open(in_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            r["_score"] = fnum(r.get("score", "nan"))
            rows.append(r)

    groups = defaultdict(list)
    for r in rows:
        key = (
            r.get("iteration", ""),
            r.get("rc", ""),
            r.get("rf", ""),
            r.get("kc", ""),
            r.get("kf", ""),
            r.get("w_mode", ""),
        )
        groups[key].append(r)

    scored = []
    for key, items in groups.items():
        scores = [fnum(it.get("score", "nan")) for it in items]
        scores = [s for s in scores if not math.isnan(s)]
        if not scores:
            continue
        scored.append((key, float(np.mean(scores)), float(np.std(scores)), len(items)))

    scored.sort(key=lambda x: x[1], reverse=True)
    top = scored[: max(1, args.top_n)]

    out_csv = os.path.join(out_dir, "top_proposal_configs.csv")
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "rank","iteration","rc","rf","kc","kf","w_mode",
            "score_mean","score_std","num_views",
            "k_values","delta_mean","delta_std"
        ])

        plot_rows = []
        for rank, (key, sc_mean, sc_std, nviews) in enumerate(top, start=1):
            it, rc, rf, kc, kf, w_mode = key
            # Aggregate delta(K) across views for this config.
            delta_by_k = defaultdict(list)
            for r in groups[key]:
                out_dir_r = r.get("out_dir", "")
                if not out_dir_r:
                    continue
                path = os.path.join(out_dir_r, "delta_oracle_summary.csv")
                for row in parse_delta_summary(path):
                    if math.isnan(row["delta_mean"]):
                        continue
                    delta_by_k[row["K"]].append(row["delta_mean"])
            if not delta_by_k:
                continue
            k_values = sorted(delta_by_k.keys())
            d_mean = [float(np.mean(delta_by_k[k])) for k in k_values]
            d_std = [float(np.std(delta_by_k[k])) for k in k_values]
            writer.writerow([
                rank, it, rc, rf, kc, kf, w_mode,
                sc_mean, sc_std, nviews,
                ";".join(str(k) for k in k_values),
                ";".join(f"{v:.6g}" for v in d_mean),
                ";".join(f"{v:.6g}" for v in d_std),
            ])
            plot_rows.append((key, sc_mean, k_values, d_mean, d_std))

    if HAS_MPL and plot_rows:
        n = len(plot_rows)
        ncols = 2 if n > 1 else 1
        nrows = int(math.ceil(n / ncols))
        fig, axs = plt.subplots(nrows, ncols, figsize=(6 * ncols, 3.5 * nrows))
        if not isinstance(axs, np.ndarray):
            axs = np.array([axs])
        axs = axs.flatten()

        for i, (key, sc_mean, k_values, d_mean, d_std) in enumerate(plot_rows):
            it, rc, rf, kc, kf, w_mode = key
            ax = axs[i]
            ax.errorbar(k_values, d_mean, yerr=d_std, marker="o")
            ax.set_title(f"iter {it} | w={w_mode} | rc={rc} rf={rf} kc={kc} kf={kf}\nscore={sc_mean:.4f}")
            ax.set_xlabel("K")
            ax.set_ylabel("delta")
        for j in range(len(plot_rows), len(axs)):
            axs[j].axis("off")
        fig.tight_layout()
        out_plot = os.path.join(out_dir, "top_proposal_configs_delta.png")
        fig.savefig(out_plot, dpi=150)

    print(f"Wrote {out_csv}")
    if HAS_MPL and plot_rows:
        print(f"Wrote {os.path.join(out_dir, 'top_proposal_configs_delta.png')}")


if __name__ == "__main__":
    main()
