#!/usr/bin/env python3
#
# Compute mean ΔN from ab_metrics.csv
#

import argparse
import math


def load_num_points(csv_path):
    pts = []
    with open(csv_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("iteration,"):
                continue
            parts = line.split(",")
            if len(parts) < 2:
                continue
            try:
                pts.append(int(parts[1]))
            except ValueError:
                pass
    return pts


def main():
    parser = argparse.ArgumentParser(description="Compute ΔN statistics from ab_metrics.csv")
    parser.add_argument("--csv", required=True, type=str)
    args = parser.parse_args()

    pts = load_num_points(args.csv)
    if len(pts) < 2:
        print("nan 0 nan")
        return

    dn = [pts[i] - pts[i - 1] for i in range(1, len(pts))]
    dn_pos = [x for x in dn if x > 0]

    mean_pos = sum(dn_pos) / len(dn_pos) if dn_pos else float("nan")
    mean_all = sum(dn) / len(dn) if dn else float("nan")
    count_pos = len(dn_pos)

    print(f"{mean_pos} {count_pos} {mean_all}")


if __name__ == "__main__":
    main()
