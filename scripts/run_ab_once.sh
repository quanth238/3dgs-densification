#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Run baseline vs triangulate once and print summary.

Usage:
  bash scripts/run_ab_once.sh \
    --source_path /path/to/dataset/mipnerf360/bonsai \
    --images images_4 \
    --base_model ./experiment/ab_base_bonsai \
    --tri_model ./experiment/ab_triang_bonsai \
    [--iterations 7000] \
    [--triang_view_indices "1,2,3,4,5"] \
    [--triang_num_pixels 150] \
    [--triang_min_ncc 0.3] \
    [--triang_relocate_frac 0.005] \
    [--triang_insert_frac 0.01] \
    [--triang_relocate_num 0] \
    [--triang_insert_num 0] \
    [--triang_max_candidates 2000] \
    [--ab_log_interval 100] \
    [--ab_eval_split train] \
    [--ab_eval_views 5] \
    [--ab_seed 0] \
    [--ab_lpips] \
    [--ab_lpips_net vgg] \
    [--ab_log_dir_base /path/to/log_dir] \
    [--ab_log_dir_tri /path/to/log_dir] \
    [--densify_from_iter 500] \
    [--densify_until_iter 15000] \
    [--densification_interval 100] \
    [--percent_dense 0.01] \
    [--densify_grad_threshold 0.0002] \
    [--densify_opacity_threshold 0.005] \
    [--densify_opacity_use_abs] \
    [--base_extra "...] \
    [--tri_extra "...] \
    [--skip_triangulate] \
    [--skip_train]
EOF
}

SOURCE_PATH=""
IMAGES="images_4"
BASE_MODEL="./experiment/ab_base_bonsai"
TRI_MODEL="./experiment/ab_triang_bonsai"
ITERATIONS="7000"
TRIANG_VIEW_INDICES="1,2,3,4,5"
TRIANG_NUM_PIXELS="150"
TRIANG_MIN_NCC="0.3"
TRIANG_RELOCATE_FRAC="0.005"
TRIANG_INSERT_FRAC="0.01"
TRIANG_RELOCATE_NUM="0"
TRIANG_INSERT_NUM="0"
TRIANG_MAX_CANDIDATES="2000"
AB_LOG_INTERVAL="100"
AB_EVAL_SPLIT="train"
AB_EVAL_VIEWS="5"
AB_SEED="0"
AB_LPIPS=1
AB_LPIPS_NET="vgg"
AB_LOG_DIR_BASE=""
AB_LOG_DIR_TRI=""
DENSIFY_FROM_ITER=""
DENSIFY_UNTIL_ITER=""
DENSIFICATION_INTERVAL=""
PERCENT_DENSE=""
DENSIFY_GRAD_THRESHOLD=""
DENSIFY_OPACITY_THRESHOLD=""
DENSIFY_OPACITY_USE_ABS=0
SKIP_TRAIN=0
SKIP_TRIANGULATE=0
BASE_EXTRA_ARGS=()
TRI_EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source_path) SOURCE_PATH="$2"; shift 2 ;;
    --images) IMAGES="$2"; shift 2 ;;
    --base_model) BASE_MODEL="$2"; shift 2 ;;
    --tri_model) TRI_MODEL="$2"; shift 2 ;;
    --iterations) ITERATIONS="$2"; shift 2 ;;
    --triang_view_indices) TRIANG_VIEW_INDICES="$2"; shift 2 ;;
    --triang_num_pixels) TRIANG_NUM_PIXELS="$2"; shift 2 ;;
    --triang_min_ncc) TRIANG_MIN_NCC="$2"; shift 2 ;;
    --triang_relocate_frac) TRIANG_RELOCATE_FRAC="$2"; shift 2 ;;
    --triang_insert_frac) TRIANG_INSERT_FRAC="$2"; shift 2 ;;
    --triang_relocate_num) TRIANG_RELOCATE_NUM="$2"; shift 2 ;;
    --triang_insert_num) TRIANG_INSERT_NUM="$2"; shift 2 ;;
    --triang_max_candidates) TRIANG_MAX_CANDIDATES="$2"; shift 2 ;;
    --ab_log_interval) AB_LOG_INTERVAL="$2"; shift 2 ;;
    --ab_eval_split) AB_EVAL_SPLIT="$2"; shift 2 ;;
    --ab_eval_views) AB_EVAL_VIEWS="$2"; shift 2 ;;
    --ab_seed) AB_SEED="$2"; shift 2 ;;
    --ab_lpips) AB_LPIPS=1; shift 1 ;;
    --ab_lpips_net) AB_LPIPS_NET="$2"; shift 2 ;;
    --ab_log_dir_base) AB_LOG_DIR_BASE="$2"; shift 2 ;;
    --ab_log_dir_tri) AB_LOG_DIR_TRI="$2"; shift 2 ;;
    --densify_from_iter) DENSIFY_FROM_ITER="$2"; shift 2 ;;
    --densify_until_iter) DENSIFY_UNTIL_ITER="$2"; shift 2 ;;
    --densification_interval) DENSIFICATION_INTERVAL="$2"; shift 2 ;;
    --percent_dense) PERCENT_DENSE="$2"; shift 2 ;;
    --densify_grad_threshold) DENSIFY_GRAD_THRESHOLD="$2"; shift 2 ;;
    --densify_opacity_threshold) DENSIFY_OPACITY_THRESHOLD="$2"; shift 2 ;;
    --densify_opacity_use_abs) DENSIFY_OPACITY_USE_ABS=1; shift 1 ;;
    --base_extra) BASE_EXTRA_ARGS+=($2); shift 2 ;;
    --tri_extra) TRI_EXTRA_ARGS+=($2); shift 2 ;;
    --skip_triangulate) SKIP_TRIANGULATE=1; shift 1 ;;
    --skip_train) SKIP_TRAIN=1; shift 1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1"; usage; exit 1 ;;
  esac
done

if [[ -z "$SOURCE_PATH" ]]; then
  echo "Error: --source_path is required."
  usage
  exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES=3
fi

if [[ -z "$AB_LOG_DIR_BASE" ]]; then
  AB_LOG_DIR_BASE="${BASE_MODEL}/ab_logs_clean"
fi
if [[ -z "$AB_LOG_DIR_TRI" ]]; then
  AB_LOG_DIR_TRI="${TRI_MODEL}/ab_logs_clean"
fi

COMMON_DENSIFY_ARGS=()
if [[ -n "$DENSIFY_FROM_ITER" ]]; then
  COMMON_DENSIFY_ARGS+=(--densify_from_iter "$DENSIFY_FROM_ITER")
fi
if [[ -n "$DENSIFY_UNTIL_ITER" ]]; then
  COMMON_DENSIFY_ARGS+=(--densify_until_iter "$DENSIFY_UNTIL_ITER")
fi
if [[ -n "$DENSIFICATION_INTERVAL" ]]; then
  COMMON_DENSIFY_ARGS+=(--densification_interval "$DENSIFICATION_INTERVAL")
fi
if [[ -n "$PERCENT_DENSE" ]]; then
  COMMON_DENSIFY_ARGS+=(--percent_dense "$PERCENT_DENSE")
fi
if [[ -n "$DENSIFY_GRAD_THRESHOLD" ]]; then
  COMMON_DENSIFY_ARGS+=(--densify_grad_threshold "$DENSIFY_GRAD_THRESHOLD")
fi
if [[ -n "$DENSIFY_OPACITY_THRESHOLD" ]]; then
  COMMON_DENSIFY_ARGS+=(--densify_opacity_threshold "$DENSIFY_OPACITY_THRESHOLD")
fi
if [[ "$DENSIFY_OPACITY_USE_ABS" -eq 1 ]]; then
  COMMON_DENSIFY_ARGS+=(--densify_opacity_use_abs)
fi

if [[ "$SKIP_TRAIN" -eq 0 ]]; then
  echo "=== Baseline (xyz) ==="
  "$PYTHON_BIN" train.py \
    --source_path "$SOURCE_PATH" \
    --model_path "$BASE_MODEL" \
    --images "$IMAGES" \
    --iterations "$ITERATIONS" \
    --densify_mode xyz \
    --ab_log --ab_log_interval "$AB_LOG_INTERVAL" \
    --ab_eval_views "$AB_EVAL_VIEWS" --ab_eval_split "$AB_EVAL_SPLIT" --ab_seed "$AB_SEED" \
    ${AB_LPIPS:+--ab_lpips} --ab_lpips_net "$AB_LPIPS_NET" \
    --ab_log_dir "$AB_LOG_DIR_BASE" \
    "${COMMON_DENSIFY_ARGS[@]}" \
    "${BASE_EXTRA_ARGS[@]}"

  echo "=== Triangulate ==="
  if [[ "$SKIP_TRIANGULATE" -eq 0 ]]; then
    "$PYTHON_BIN" train.py \
      --source_path "$SOURCE_PATH" \
      --model_path "$TRI_MODEL" \
      --images "$IMAGES" \
      --iterations "$ITERATIONS" \
      --densify_mode triangulate \
      --triang_view_indices "$TRIANG_VIEW_INDICES" \
      --triang_num_pixels "$TRIANG_NUM_PIXELS" \
      --triang_min_ncc "$TRIANG_MIN_NCC" \
      --triang_relocate_frac "$TRIANG_RELOCATE_FRAC" \
      --triang_insert_frac "$TRIANG_INSERT_FRAC" \
      --triang_relocate_num "$TRIANG_RELOCATE_NUM" \
      --triang_insert_num "$TRIANG_INSERT_NUM" \
      --triang_max_candidates "$TRIANG_MAX_CANDIDATES" \
      --triang_weight_num_views 5 --triang_weight_view_stride 10 \
      --ab_log --ab_log_interval "$AB_LOG_INTERVAL" \
      --ab_eval_views "$AB_EVAL_VIEWS" --ab_eval_split "$AB_EVAL_SPLIT" --ab_seed "$AB_SEED" \
      ${AB_LPIPS:+--ab_lpips} --ab_lpips_net "$AB_LPIPS_NET" \
      --ab_log_dir "$AB_LOG_DIR_TRI" \
      "${COMMON_DENSIFY_ARGS[@]}" \
      "${TRI_EXTRA_ARGS[@]}"
  else
    echo "=== Triangulate skipped ==="
  fi
fi

BASE_CSV="${AB_LOG_DIR_BASE}/xyz/ab_metrics.csv"
TRI_CSV="${AB_LOG_DIR_TRI}/triangulate/ab_metrics.csv"
OUT_DIR="${TRI_MODEL}/ab_compare"

if [[ "$SKIP_TRIANGULATE" -eq 1 ]]; then
  if [[ -f "$BASE_CSV" ]]; then
    echo "=== Baseline metrics ==="
    echo "$BASE_CSV"
  else
    echo "Error: missing baseline ab_metrics.csv: $BASE_CSV"
    exit 1
  fi
  exit 0
fi

if [[ ! -f "$BASE_CSV" || ! -f "$TRI_CSV" ]]; then
  echo "Error: missing ab_metrics.csv. Baseline: $BASE_CSV, Triangulate: $TRI_CSV"
  exit 1
fi

"$PYTHON_BIN" scripts/plot_ab_compare.py \
  --baseline "$BASE_CSV" \
  --oracle "$TRI_CSV" \
  --out_dir "$OUT_DIR"

echo "=== Summary (${OUT_DIR}/ab_compare_summary.csv) ==="
cat "${OUT_DIR}/ab_compare_summary.csv"
