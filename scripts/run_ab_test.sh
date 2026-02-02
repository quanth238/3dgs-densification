#!/usr/bin/env bash
set -euo pipefail

# A/B test: baseline (xyz densify) vs oracle (opacity densify)
# Multi-dataset runner (mirrors run_oracle_sweep.sh patterns).

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"

# Dataset roots
DATASET_ROOT_MIPNERF360="/home/tri-dev/dev/namn_workspace/dataset/mipnerf360"
DATASET_ROOT_TNT="/home/tri-dev/dev/namn_workspace/dataset/tanks_and_temples"

# Scenes to run (edit to match what you have)
MIPNERF360_SCENES=(bonsai room)
TNT_SCENES=(train truck)

# Images dirs
DEFAULT_IMAGES_MIP="images_4"
DEFAULT_IMAGES_TNT="images"

# Training setup
ITERATIONS=7000
SAVE_ITERS=(1000 3000 7000)
DENSIFY_FROM=500
DENSIFY_UNTIL=7000
DENSIFY_INTERVAL=100
OPACITY_RESET=3000
LAMBDA_DSSIM=0
AB_EVAL_SPLIT="train"
AB_EVAL_VIEWS=5
AB_SEED=0
AB_LPIPS=1
AB_LPIPS_NET="vgg"

# Oracle threshold (set to tuned value)
DENSIFY_OPACITY_THRESHOLD="${DENSIFY_OPACITY_THRESHOLD:-2e-5}"

run_ab() {
  local name="$1"
  local dataset_path="$2"
  local images_dir="$3"

  local base_model="./experiment/ab_baseline_${name}"
  local oracle_model="./experiment/ab_oracle_${name}"
  local out_dir="./experiment/ab_compare/${name}"

  if [[ ! -d "${dataset_path}" ]]; then
    echo "Skip ${name}: dataset path not found: ${dataset_path}"
    return
  fi

  COMMON_ARGS=(
    --source_path "${dataset_path}"
    --images "${images_dir}"
    --iterations "${ITERATIONS}"
    --save_iterations "${SAVE_ITERS[@]}"
    --densify_from_iter "${DENSIFY_FROM}"
    --densify_until_iter "${DENSIFY_UNTIL}"
    --densification_interval "${DENSIFY_INTERVAL}"
    --opacity_reset_interval "${OPACITY_RESET}"
    --lambda_dssim "${LAMBDA_DSSIM}"
    --ab_log
    --ab_log_interval "${DENSIFY_INTERVAL}"
    --ab_eval_views "${AB_EVAL_VIEWS}"
    --ab_eval_split "${AB_EVAL_SPLIT}"
    --ab_lpips
    --ab_lpips_net "${AB_LPIPS_NET}"
    --ab_seed "${AB_SEED}"
    --disable_viewer
    --quiet
  )

  echo "=== Baseline (xyz) | ${name} ==="
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python train.py \
    --model_path "${base_model}" \
    --densify_mode xyz \
    "${COMMON_ARGS[@]}"

  echo "=== Oracle (opacity) | ${name} ==="
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python train.py \
    --model_path "${oracle_model}" \
    --densify_mode opacity \
    --densify_opacity_threshold "${DENSIFY_OPACITY_THRESHOLD}" \
    "${COMMON_ARGS[@]}"

  echo "=== Plot A/B | ${name} ==="
  python scripts/plot_ab_compare.py \
    --baseline "${base_model}/ab_logs/xyz/ab_metrics.csv" \
    --oracle "${oracle_model}/ab_logs/opacity/ab_metrics.csv" \
    --out_dir "${out_dir}"
}

# Run MipNeRF360 scenes
for scene in "${MIPNERF360_SCENES[@]}"; do
  run_ab "${scene}" "${DATASET_ROOT_MIPNERF360}/${scene}" "${DEFAULT_IMAGES_MIP}"
done

# Run TNT scenes
for scene in "${TNT_SCENES[@]}"; do
  run_ab "tnt_${scene}" "${DATASET_ROOT_TNT}/${scene}" "${DEFAULT_IMAGES_TNT}"
done
