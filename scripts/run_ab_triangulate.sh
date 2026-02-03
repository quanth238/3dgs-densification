#!/usr/bin/env bash
set -euo pipefail

# A/B test: baseline (xyz densify) vs triangulate-proposal densify.
# Logs PSNR/LPIPS vs #Gaussians via ab_metrics.csv and plots with plot_ab_compare.py.

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"

# Dataset roots (override via env if needed)
DATASET_ROOT_MIPNERF360="${DATASET_ROOT_MIPNERF360:-/home/tri-dev/dev/namn_workspace/dataset/mipnerf360}"
DATASET_ROOT_TNT="${DATASET_ROOT_TNT:-/home/tri-dev/dev/namn_workspace/dataset/tanks_and_temples}"

# Scenes to run (edit to match what you have)
MIPNERF360_SCENES=(bonsai room)
TNT_SCENES=(train truck)

# Images dirs
DEFAULT_IMAGES_MIP="${DEFAULT_IMAGES_MIP:-images_4}"
DEFAULT_IMAGES_TNT="${DEFAULT_IMAGES_TNT:-images}"

# Training setup
ITERATIONS="${ITERATIONS:-7000}"
SAVE_ITERS=(1000 3000 7000)
DENSIFY_FROM="${DENSIFY_FROM:-500}"
DENSIFY_UNTIL="${DENSIFY_UNTIL:-7000}"
DENSIFY_INTERVAL="${DENSIFY_INTERVAL:-100}"
OPACITY_RESET="${OPACITY_RESET:-3000}"
LAMBDA_DSSIM="${LAMBDA_DSSIM:-0}"

# A/B eval setup
AB_EVAL_SPLIT="${AB_EVAL_SPLIT:-train}"
AB_EVAL_VIEWS="${AB_EVAL_VIEWS:-5}"
AB_SEED="${AB_SEED:-0}"
AB_LPIPS_NET="${AB_LPIPS_NET:-vgg}"

# Triangulation proposal params (override via env if needed)
TRIANG_VIEW_INDICES="${TRIANG_VIEW_INDICES:-1,2,3,4,5}"
TRIANG_NUM_VIEWS="${TRIANG_NUM_VIEWS:-5}"
TRIANG_VIEW_STRIDE="${TRIANG_VIEW_STRIDE:-10}"
TRIANG_NUM_PIXELS="${TRIANG_NUM_PIXELS:-150}"
TRIANG_DOWNSCALE="${TRIANG_DOWNSCALE:-1}"
TRIANG_PATCH="${TRIANG_PATCH:-2}"
TRIANG_MAX_SAMPLES="${TRIANG_MAX_SAMPLES:-128}"
TRIANG_MIN_NCC="${TRIANG_MIN_NCC:-0.3}"
TRIANG_DEPTH_MIN="${TRIANG_DEPTH_MIN:-0.3}"
TRIANG_DEPTH_MAX="${TRIANG_DEPTH_MAX:-80}"
TRIANG_SCALE_FACTOR="${TRIANG_SCALE_FACTOR:-3.0}"
TRIANG_SCALE_MIN_MULT="${TRIANG_SCALE_MIN_MULT:-0.5}"
TRIANG_SCALE_MAX_MULT="${TRIANG_SCALE_MAX_MULT:-1.5}"
TRIANG_ALPHA0="${TRIANG_ALPHA0:-1e-2}"
TRIANG_MAX_CANDIDATES="${TRIANG_MAX_CANDIDATES:-2000}"

# Relocation + insert (set either *_NUM or *_FRAC).
# Defaults ensure N grows (so A/B is fair vs baseline).
TRIANG_RELOCATE_NUM="${TRIANG_RELOCATE_NUM:-0}"
TRIANG_RELOCATE_FRAC="${TRIANG_RELOCATE_FRAC:-0.005}"
TRIANG_RELOCATE_BY="${TRIANG_RELOCATE_BY:-resp_opacity}"
TRIANG_INSERT_NUM="${TRIANG_INSERT_NUM:-0}"
TRIANG_INSERT_FRAC="${TRIANG_INSERT_FRAC:-0.01}"
TRIANG_WEIGHT_NUM_VIEWS="${TRIANG_WEIGHT_NUM_VIEWS:-5}"
TRIANG_WEIGHT_VIEW_STRIDE="${TRIANG_WEIGHT_VIEW_STRIDE:-10}"

run_ab() {
  local name="$1"
  local dataset_path="$2"
  local images_dir="$3"

  local base_model="./experiment/ab_base_${name}"
  local tri_model="./experiment/ab_triang_${name}"
  local out_dir="./experiment/ab_compare_triang/${name}"

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

  echo "=== Triangulate | ${name} ==="
  if [[ "${TRIANG_INSERT_NUM}" == "0" && "${TRIANG_INSERT_FRAC}" == "0" ]]; then
    echo "Warning: triangulate insert is disabled (TRIANG_INSERT_NUM=0 and TRIANG_INSERT_FRAC=0)."
    echo "         This will freeze #Gaussians and makes A/B comparison unfair."
  fi
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python train.py \
    --model_path "${tri_model}" \
    --densify_mode triangulate \
    --triang_view_indices "${TRIANG_VIEW_INDICES}" \
    --triang_num_views "${TRIANG_NUM_VIEWS}" \
    --triang_view_stride "${TRIANG_VIEW_STRIDE}" \
    --triang_num_pixels "${TRIANG_NUM_PIXELS}" \
    --triang_downscale "${TRIANG_DOWNSCALE}" \
    --triang_patch "${TRIANG_PATCH}" \
    --triang_max_samples "${TRIANG_MAX_SAMPLES}" \
    --triang_min_ncc "${TRIANG_MIN_NCC}" \
    --triang_depth_min "${TRIANG_DEPTH_MIN}" \
    --triang_depth_max "${TRIANG_DEPTH_MAX}" \
    --triang_scale_factor "${TRIANG_SCALE_FACTOR}" \
    --triang_scale_min_mult "${TRIANG_SCALE_MIN_MULT}" \
    --triang_scale_max_mult "${TRIANG_SCALE_MAX_MULT}" \
    --triang_alpha0 "${TRIANG_ALPHA0}" \
    --triang_max_candidates "${TRIANG_MAX_CANDIDATES}" \
    --triang_relocate_num "${TRIANG_RELOCATE_NUM}" \
    --triang_relocate_frac "${TRIANG_RELOCATE_FRAC}" \
    --triang_relocate_by "${TRIANG_RELOCATE_BY}" \
    --triang_insert_num "${TRIANG_INSERT_NUM}" \
    --triang_insert_frac "${TRIANG_INSERT_FRAC}" \
    --triang_weight_num_views "${TRIANG_WEIGHT_NUM_VIEWS}" \
    --triang_weight_view_stride "${TRIANG_WEIGHT_VIEW_STRIDE}" \
    "${COMMON_ARGS[@]}"

  echo "=== Plot A/B | ${name} ==="
  python scripts/plot_ab_compare.py \
    --baseline "${base_model}/ab_logs/xyz/ab_metrics.csv" \
    --oracle "${tri_model}/ab_logs/triangulate/ab_metrics.csv" \
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
