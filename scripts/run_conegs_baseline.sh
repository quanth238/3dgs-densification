#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_conegs_baseline.sh \
    --source_path /path/to/dataset/mipnerf360/bonsai \
    --model_path ./experiment/conegs_bonsai \
    --images images_4 \
    --scene_name bonsai \
    [--run_name conegs_bonsai] \
    [--max_points 0] \
    [--gaussian_percentage_increase 0.02] \
    [--extra "optimization.iterations=30000 ..."]

Notes:
- Runs ConeGS from reference/conegs with Hydra config defaults.yaml.
- Set max_points>0 for budgeted training, 0 for unbudgeted (matches ConeGS README).
EOF
}

SOURCE_PATH=""
MODEL_PATH=""
IMAGES="images_4"
SCENE_NAME="bonsai"
RUN_NAME=""
MAX_POINTS="0"
GAUSS_PCT="0.02"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source_path) SOURCE_PATH="$2"; shift 2 ;;
    --model_path) MODEL_PATH="$2"; shift 2 ;;
    --images) IMAGES="$2"; shift 2 ;;
    --scene_name) SCENE_NAME="$2"; shift 2 ;;
    --run_name) RUN_NAME="$2"; shift 2 ;;
    --max_points) MAX_POINTS="$2"; shift 2 ;;
    --gaussian_percentage_increase) GAUSS_PCT="$2"; shift 2 ;;
    --extra) EXTRA_ARGS+=($2); shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1"; usage; exit 1 ;;
  esac
done

if [[ -z "$SOURCE_PATH" || -z "$MODEL_PATH" ]]; then
  echo "Error: --source_path and --model_path are required."
  usage
  exit 1
fi

if [[ -z "$RUN_NAME" ]]; then
  RUN_NAME="conegs_${SCENE_NAME}"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONEGS_DIR="${SCRIPT_DIR}/../reference/conegs"

if [[ ! -d "$CONEGS_DIR" ]]; then
  echo "Error: ConeGS repo not found at ${CONEGS_DIR}"
  exit 1
fi

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES=3
fi

pushd "$CONEGS_DIR" >/dev/null
python train.py --config-name defaults.yaml \
  gaussian_model.source_path="$SOURCE_PATH" \
  gaussian_model.images="$IMAGES" \
  scene_name="$SCENE_NAME" \
  run_name="$RUN_NAME" \
  gaussian_model.model_path="$MODEL_PATH" \
  optimization.max_points="$MAX_POINTS" \
  optimization.gaussian_percentage_increase="$GAUSS_PCT" \
  "${EXTRA_ARGS[@]}"
popd >/dev/null
