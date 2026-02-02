#!/usr/bin/env python3
#
# Compare baseline vs oracle A/B logs and plot metrics.
#

import argparse
import os
import csv

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False


def load_csv(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("iteration,"):
                continue
            parts = line.split(",")
            if len(parts) < 5:
                continue
            rows.append(
                {
                    "iteration": int(parts[0]),
                    "num_points": int(parts[1]),
                    "psnr": float(parts[2]),
                    "l1": float(parts[3]),
                    "lpips": float(parts[4]) if parts[4] not in ("", "nan") else float("nan"),
                }
            )
    rows = sorted(rows, key=lambda r: r["iteration"])
    return rows


def gain_per_atom(rows):
    gains = []
    for i in range(1, len(rows)):
        dn = rows[i]["num_points"] - rows[i - 1]["num_points"]
        if dn <= 0:
            gains.append(float("nan"))
        else:
            gains.append((rows[i]["psnr"] - rows[i - 1]["psnr"]) / dn)
    return gains


def auc_psnr_vs_n(rows):
    if len(rows) < 2:
        return float("nan")
    n = np.array([r["num_points"] for r in rows], dtype=np.float64)
    ps = np.array([r["psnr"] for r in rows], dtype=np.float64)
    # Ensure strictly increasing in n for integration
    order = np.argsort(n)
    n = n[order]
    ps = ps[order]
    return float(np.trapz(ps, n))


def summarize(rows):
    if not rows:
        return {}
    gains = np.array([g for g in gain_per_atom(rows) if not np.isnan(g)], dtype=np.float64)
    dn = np.array([rows[i]["num_points"] - rows[i - 1]["num_points"] for i in range(1, len(rows))], dtype=np.float64)
    dn_pos = dn[dn > 0]
    last = rows[-1]
    return {
        "final_num_points": last["num_points"],
        "final_psnr": last["psnr"],
        "final_l1": last["l1"],
        "final_lpips": last["lpips"],
        "best_psnr": float(np.nanmax([r["psnr"] for r in rows])),
        "best_l1": float(np.nanmin([r["l1"] for r in rows])),
        "best_lpips": float(np.nanmin([r["lpips"] for r in rows])) if not all(np.isnan([r["lpips"] for r in rows])) else float("nan"),
        "auc_psnr_vs_n": auc_psnr_vs_n(rows),
        "mean_gain_per_atom": float(np.nanmean(gains)) if gains.size else float("nan"),
        "median_gain_per_atom": float(np.nanmedian(gains)) if gains.size else float("nan"),
        "mean_dn": float(np.mean(dn_pos)) if dn_pos.size else float("nan"),
        "num_steps": len(rows),
    }


def main():
    parser = argparse.ArgumentParser(description="Plot A/B comparison from ab_metrics.csv logs.")
    parser.add_argument("--baseline", required=True, type=str)
    parser.add_argument("--oracle", required=True, type=str)
    parser.add_argument("--out_dir", required=True, type=str)
    args = parser.parse_args()

    base_rows = load_csv(args.baseline)
    oracle_rows = load_csv(args.oracle)

    if not base_rows or not oracle_rows:
        raise RuntimeError("Missing or empty CSV logs. Check paths.")

    os.makedirs(args.out_dir, exist_ok=True)

    # Save combined CSV for convenience
    out_csv = os.path.join(args.out_dir, "ab_compare.csv")
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("method,iteration,num_points,psnr,l1,lpips\n")
        for r in base_rows:
            f.write(f"baseline,{r['iteration']},{r['num_points']},{r['psnr']},{r['l1']},{r['lpips']}\n")
        for r in oracle_rows:
            f.write(f"oracle,{r['iteration']},{r['num_points']},{r['psnr']},{r['l1']},{r['lpips']}\n")

    if not HAS_MPL:
        print(f"Wrote {out_csv} (matplotlib not available for plots).")
        return

    # Plot metrics vs num_points
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    ax_psnr, ax_lpips = axes[0]
    ax_gain, ax_l1 = axes[1]

    def plot_curve(rows, label, color):
        n = [r["num_points"] for r in rows]
        ps = [r["psnr"] for r in rows]
        l1 = [r["l1"] for r in rows]
        lp = [r["lpips"] for r in rows]
        ax_psnr.plot(n, ps, marker="o", label=label, color=color)
        ax_l1.plot(n, l1, marker="o", label=label, color=color)
        if not all(np.isnan(lp)):
            ax_lpips.plot(n, lp, marker="o", label=label, color=color)

    plot_curve(base_rows, "baseline", "tab:blue")
    plot_curve(oracle_rows, "oracle", "tab:orange")

    ax_psnr.set_title("PSNR vs #Gaussians")
    ax_psnr.set_xlabel("#Gaussians")
    ax_psnr.set_ylabel("PSNR")
    ax_psnr.grid(True, alpha=0.2)

    ax_lpips.set_title("LPIPS vs #Gaussians")
    ax_lpips.set_xlabel("#Gaussians")
    ax_lpips.set_ylabel("LPIPS")
    ax_lpips.grid(True, alpha=0.2)

    ax_l1.set_title("L1 vs #Gaussians")
    ax_l1.set_xlabel("#Gaussians")
    ax_l1.set_ylabel("L1")
    ax_l1.grid(True, alpha=0.2)

    base_gain = gain_per_atom(base_rows)
    oracle_gain = gain_per_atom(oracle_rows)
    ax_gain.plot(range(1, len(base_rows)), base_gain, marker="o", label="baseline", color="tab:blue")
    ax_gain.plot(range(1, len(oracle_rows)), oracle_gain, marker="o", label="oracle", color="tab:orange")
    ax_gain.set_title("Gain-per-atom (ΔPSNR/ΔN)")
    ax_gain.set_xlabel("Densify step")
    ax_gain.set_ylabel("ΔPSNR/ΔN")
    ax_gain.grid(True, alpha=0.2)

    ax_psnr.legend()
    ax_lpips.legend()
    ax_l1.legend()
    ax_gain.legend()

    plt.tight_layout()
    out_png = os.path.join(args.out_dir, "ab_compare.png")
    plt.savefig(out_png, dpi=200)
    plt.close(fig)

    # Summary CSV
    base_sum = summarize(base_rows)
    oracle_sum = summarize(oracle_rows)
    summary_csv = os.path.join(args.out_dir, "ab_compare_summary.csv")
    with open(summary_csv, "w", encoding="utf-8") as f:
        f.write("method,final_num_points,final_psnr,final_l1,final_lpips,best_psnr,best_l1,best_lpips,auc_psnr_vs_n,mean_gain_per_atom,median_gain_per_atom,mean_dn,num_steps\n")
        f.write(
            "baseline,{final_num_points},{final_psnr},{final_l1},{final_lpips},{best_psnr},{best_l1},{best_lpips},{auc_psnr_vs_n},{mean_gain_per_atom},{median_gain_per_atom},{mean_dn},{num_steps}\n"
            .format(**base_sum)
        )
        f.write(
            "oracle,{final_num_points},{final_psnr},{final_l1},{final_lpips},{best_psnr},{best_l1},{best_lpips},{auc_psnr_vs_n},{mean_gain_per_atom},{median_gain_per_atom},{mean_dn},{num_steps}\n"
            .format(**oracle_sum)
        )
        # Delta row (oracle - baseline)
        def safe_diff(a, b):
            try:
                return float(a) - float(b)
            except Exception:
                return float("nan")
        f.write(
            "delta,"
            f"{safe_diff(oracle_sum.get('final_num_points'), base_sum.get('final_num_points'))},"
            f"{safe_diff(oracle_sum.get('final_psnr'), base_sum.get('final_psnr'))},"
            f"{safe_diff(oracle_sum.get('final_l1'), base_sum.get('final_l1'))},"
            f"{safe_diff(oracle_sum.get('final_lpips'), base_sum.get('final_lpips'))},"
            f"{safe_diff(oracle_sum.get('best_psnr'), base_sum.get('best_psnr'))},"
            f"{safe_diff(oracle_sum.get('best_l1'), base_sum.get('best_l1'))},"
            f"{safe_diff(oracle_sum.get('best_lpips'), base_sum.get('best_lpips'))},"
            f"{safe_diff(oracle_sum.get('auc_psnr_vs_n'), base_sum.get('auc_psnr_vs_n'))},"
            f"{safe_diff(oracle_sum.get('mean_gain_per_atom'), base_sum.get('mean_gain_per_atom'))},"
            f"{safe_diff(oracle_sum.get('median_gain_per_atom'), base_sum.get('median_gain_per_atom'))},"
            f"{safe_diff(oracle_sum.get('mean_dn'), base_sum.get('mean_dn'))},"
            f"{safe_diff(oracle_sum.get('num_steps'), base_sum.get('num_steps'))}\n"
        )

    print(f"Wrote {out_csv}, {summary_csv}, and {out_png}")


if __name__ == "__main__":
    main()
