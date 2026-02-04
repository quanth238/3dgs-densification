#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Run A/B/C in one go:
  A = baseline 3DGS (densify_mode=xyz)
  B = method (configurable densify_mode) OR skip with --skip_method
  C = ConeGS (reference/conegs)

Usage:
  bash scripts/run_abc_once.sh \
    --source_path /path/to/dataset/mipnerf360/bonsai \
    --images images_4 \
    --base_model ./experiment/ab_base_bonsai_clean \
    --method_model ./experiment/ab_method_bonsai_clean \
    --conegs_model ./reference/conegs/experiment/conegs_bonsai \
    --scene_name bonsai \
    [--iterations 7000] \
    [--ab_log_interval 50] \
    [--ab_eval_split train] \
    [--ab_eval_views 5] \
    [--ab_seed 0] \
    [--ab_lpips] \
    [--ab_lpips_net vgg] \
    [--skip_base] \
    [--skip_method] \
    [--skip_conegs] \
    [--base_extra "...] \
    [--method_mode triangulate] \
    [--method_extra "...] \
    [--conegs_extra "optimization.iterations=30000 ..."]

Notes:
- Logs are written to <model>/ab_logs_clean by default.
- ConeGS runs from reference/conegs via Hydra defaults.yaml.
EOF
}

SOURCE_PATH=""
IMAGES="images_4"
BASE_MODEL="./experiment/ab_base_bonsai_clean"
METHOD_MODEL="./experiment/ab_method_bonsai_clean"
CONEGS_MODEL="./reference/conegs/experiment/conegs_bonsai"
SCENE_NAME="bonsai"
ITERATIONS="7000"
AB_LOG_INTERVAL="50"
AB_EVAL_SPLIT="train"
AB_EVAL_VIEWS="5"
AB_SEED="0"
AB_LPIPS=1
AB_LPIPS_NET="vgg"
METHOD_MODE="triangulate"
SKIP_BASE=0
SKIP_METHOD=0
SKIP_CONEGS=0
BASE_EXTRA_ARGS=()
METHOD_EXTRA_ARGS=()
CONEGS_EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source_path) SOURCE_PATH="$2"; shift 2 ;;
    --images) IMAGES="$2"; shift 2 ;;
    --base_model) BASE_MODEL="$2"; shift 2 ;;
    --method_model) METHOD_MODEL="$2"; shift 2 ;;
    --conegs_model) CONEGS_MODEL="$2"; shift 2 ;;
    --scene_name) SCENE_NAME="$2"; shift 2 ;;
    --iterations) ITERATIONS="$2"; shift 2 ;;
    --ab_log_interval) AB_LOG_INTERVAL="$2"; shift 2 ;;
    --ab_eval_split) AB_EVAL_SPLIT="$2"; shift 2 ;;
    --ab_eval_views) AB_EVAL_VIEWS="$2"; shift 2 ;;
    --ab_seed) AB_SEED="$2"; shift 2 ;;
    --ab_lpips) AB_LPIPS=1; shift 1 ;;
    --ab_lpips_net) AB_LPIPS_NET="$2"; shift 2 ;;
    --method_mode) METHOD_MODE="$2"; shift 2 ;;
    --skip_base) SKIP_BASE=1; shift 1 ;;
    --skip_method) SKIP_METHOD=1; shift 1 ;;
    --skip_conegs) SKIP_CONEGS=1; shift 1 ;;
    --base_extra) BASE_EXTRA_ARGS+=($2); shift 2 ;;
    --method_extra) METHOD_EXTRA_ARGS+=($2); shift 2 ;;
    --conegs_extra) CONEGS_EXTRA_ARGS+=($2); shift 2 ;;
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

BASE_LOG_DIR="${BASE_MODEL}/ab_logs_clean"
METHOD_LOG_DIR="${METHOD_MODEL}/ab_logs_clean"

if [[ "$SKIP_BASE" -eq 0 ]]; then
  echo "=== A: Baseline (xyz) ==="
  "$PYTHON_BIN" train.py \
    --source_path "$SOURCE_PATH" \
    --model_path "$BASE_MODEL" \
    --images "$IMAGES" \
    --iterations "$ITERATIONS" \
    --densify_mode xyz \
    --ab_log --ab_log_interval "$AB_LOG_INTERVAL" \
    --ab_eval_views "$AB_EVAL_VIEWS" --ab_eval_split "$AB_EVAL_SPLIT" --ab_seed "$AB_SEED" \
    ${AB_LPIPS:+--ab_lpips} --ab_lpips_net "$AB_LPIPS_NET" \
    --ab_log_dir "$BASE_LOG_DIR" \
    "${BASE_EXTRA_ARGS[@]}"
fi

if [[ "$SKIP_METHOD" -eq 0 ]]; then
  echo "=== B: Method (${METHOD_MODE}) ==="
  "$PYTHON_BIN" train.py \
    --source_path "$SOURCE_PATH" \
    --model_path "$METHOD_MODEL" \
    --images "$IMAGES" \
    --iterations "$ITERATIONS" \
    --densify_mode "$METHOD_MODE" \
    --ab_log --ab_log_interval "$AB_LOG_INTERVAL" \
    --ab_eval_views "$AB_EVAL_VIEWS" --ab_eval_split "$AB_EVAL_SPLIT" --ab_seed "$AB_SEED" \
    ${AB_LPIPS:+--ab_lpips} --ab_lpips_net "$AB_LPIPS_NET" \
    --ab_log_dir "$METHOD_LOG_DIR" \
    "${METHOD_EXTRA_ARGS[@]}"
fi

if [[ "$SKIP_CONEGS" -eq 0 ]]; then
  echo "=== C: ConeGS ==="
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  CONEGS_DIR="${SCRIPT_DIR}/../reference/conegs"
  if [[ ! -d "$CONEGS_DIR" ]]; then
    echo "Error: ConeGS repo not found at ${CONEGS_DIR}"
    exit 1
  fi
  pushd "$CONEGS_DIR" >/dev/null
  "$PYTHON_BIN" train.py --config-name defaults.yaml \
    gaussian_model.source_path="$SOURCE_PATH" \
    gaussian_model.images="$IMAGES" \
    scene_name="$SCENE_NAME" \
    run_name="$(basename "$CONEGS_MODEL")" \
    gaussian_model.model_path="$CONEGS_MODEL" \
    "${CONEGS_EXTRA_ARGS[@]}"
  popd >/dev/null
fi

echo "=== Done ==="
echo "Baseline log: ${BASE_LOG_DIR}/xyz/ab_metrics.csv"
echo "Method log:   ${METHOD_LOG_DIR}/${METHOD_MODE}/ab_metrics.csv"
echo "ConeGS model: ${CONEGS_MODEL}"
