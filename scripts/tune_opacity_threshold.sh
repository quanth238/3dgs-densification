#!/usr/bin/env bash
set -euo pipefail

# Tune densify_opacity_threshold to match ΔN/interval of baseline (xyz)

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"

# ---- Edit these ----
DATASET_PATH="/home/tri-dev/dev/namn_workspace/dataset/mipnerf360/bonsai"
IMAGES_DIR="images_4"     # use "images" for TNT
BASE_MODEL="./experiment/tune_baseline_bonsai"
ORACLE_PREFIX="./experiment/tune_oracle_bonsai"
ITERATIONS=2000
SAVE_ITERS=(1000 2000)
THRESHOLDS=(2e-5 5e-5 1e-4 2e-4)

# Densify schedule
DENSIFY_FROM=500
DENSIFY_UNTIL=2000
DENSIFY_INTERVAL=100
OPACITY_RESET=3000

COMMON_ARGS=(
  --source_path "${DATASET_PATH}"
  --images "${IMAGES_DIR}"
  --iterations "${ITERATIONS}"
  --save_iterations "${SAVE_ITERS[@]}"
  --densify_from_iter "${DENSIFY_FROM}"
  --densify_until_iter "${DENSIFY_UNTIL}"
  --densification_interval "${DENSIFY_INTERVAL}"
  --opacity_reset_interval "${OPACITY_RESET}"
  --lambda_dssim 0
  --ab_log
  --ab_log_interval "${DENSIFY_INTERVAL}"
  --ab_eval_views 5
  --ab_eval_split train
  --ab_lpips
  --ab_lpips_net vgg
  --ab_seed 0
  --disable_viewer
  --quiet
)

SUMMARY_CSV="./experiment/tune_threshold_summary.csv"
mkdir -p "$(dirname "${SUMMARY_CSV}")"
echo "threshold,mean_pos_dn,mean_pos_count,mean_all_dn,abs_diff_to_baseline" > "${SUMMARY_CSV}"

echo "=== Baseline (xyz) ==="
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python train.py \
  --model_path "${BASE_MODEL}" \
  --densify_mode xyz \
  "${COMMON_ARGS[@]}"

BASE_CSV="${BASE_MODEL}/ab_logs/xyz/ab_metrics.csv"
read base_mean base_count base_all <<< "$(python scripts/compute_delta_n.py --csv "${BASE_CSV}")"
echo "Baseline mean_pos_dn=${base_mean}, count=${base_count}, mean_all_dn=${base_all}"

best_thr=""
best_diff=""

for thr in "${THRESHOLDS[@]}"; do
  echo "=== Oracle threshold ${thr} ==="
  ORACLE_MODEL="${ORACLE_PREFIX}_${thr}"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python train.py \
    --model_path "${ORACLE_MODEL}" \
    --densify_mode opacity \
    --densify_opacity_threshold "${thr}" \
    "${COMMON_ARGS[@]}"

  ORACLE_CSV="${ORACLE_MODEL}/ab_logs/opacity/ab_metrics.csv"
  read mean_pos count_pos mean_all <<< "$(python scripts/compute_delta_n.py --csv "${ORACLE_CSV}")"
  diff="$(python - <<PY
import math
bm = float("${base_mean}")
om = float("${mean_pos}")
print(abs(bm - om))
PY
)"

  echo "${thr},${mean_pos},${count_pos},${mean_all},${diff}" >> "${SUMMARY_CSV}"

  if [[ -z "${best_diff}" ]]; then
    best_diff="${diff}"
    best_thr="${thr}"
  else
    is_better="$(python - <<PY
import math
best = float("${best_diff}")
cur = float("${diff}")
print(1 if cur < best else 0)
PY
)"
    if [[ "${is_better}" == "1" ]]; then
      best_diff="${diff}"
      best_thr="${thr}"
    fi
  fi
done

echo "Summary: ${SUMMARY_CSV}"
echo "Best threshold: ${best_thr} (abs diff ${best_diff})"
