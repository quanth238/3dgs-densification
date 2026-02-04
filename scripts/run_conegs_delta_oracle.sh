#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_conegs_delta_oracle.sh \
    --source_path /path/to/dataset/mipnerf360/bonsai \
    --model_path ./experiment/conegs_bonsai \
  --iteration 30000 \
  --images images_4 \
  --split train --view_index 0 \
  [--use_cfg] \
  [--ply_path /path/to/point_cloud.ply] \
  [--out_dir ./experiment/conegs_bonsai/oracle_verify/delta_oracle_conegs] \
  [--extra "--proposal_mode triangulate --triang_view_indices 1,2,3,4,5 ..."]

Notes:
- Uses scripts/estimate_delta_oracle.py in THIS repo, but loads the ConeGS PLY
  from <model_path>/point_cloud/iteration_<iter>/point_cloud.ply.
- Pass additional options via --extra "...".
EOF
}

SOURCE_PATH=""
MODEL_PATH=""
ITERATION="30000"
IMAGES="images_4"
SPLIT="train"
VIEW_INDEX="0"
OUT_DIR=""
EXTRA_ARGS=()
USE_CFG=0
PLY_PATH=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source_path) SOURCE_PATH="$2"; shift 2 ;;
    --model_path) MODEL_PATH="$2"; shift 2 ;;
    --iteration) ITERATION="$2"; shift 2 ;;
    --images) IMAGES="$2"; shift 2 ;;
    --split) SPLIT="$2"; shift 2 ;;
    --view_index) VIEW_INDEX="$2"; shift 2 ;;
    --out_dir) OUT_DIR="$2"; shift 2 ;;
    --use_cfg) USE_CFG=1; shift 1 ;;
    --ply_path) PLY_PATH="$2"; shift 2 ;;
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

if [[ -z "$OUT_DIR" ]]; then
  OUT_DIR="${MODEL_PATH}/oracle_verify/delta_oracle_conegs"
fi

if [[ -z "$PLY_PATH" ]]; then
  EXPECTED_PLY="${MODEL_PATH}/point_cloud/iteration_${ITERATION}/point_cloud.ply"
  if [[ -f "$EXPECTED_PLY" ]]; then
    PLY_PATH="$EXPECTED_PLY"
  else
    # Try to locate a point_cloud.ply under model_path.
    FOUND_PLY=$(find "$MODEL_PATH" -type f -name "point_cloud.ply" 2>/dev/null | head -n 1 || true)
    if [[ -n "$FOUND_PLY" ]]; then
      PLY_PATH="$FOUND_PLY"
      echo "Info: using PLY at $PLY_PATH"
    fi
  fi
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${SCRIPT_DIR}/.."

PYTHON_BIN="${PYTHON_BIN:-python}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  PYTHON_BIN="python3"
fi

CUDA_VISIBLE_DEVICES=3 "$PYTHON_BIN" "${ROOT_DIR}/scripts/estimate_delta_oracle.py" \
  --source_path "$SOURCE_PATH" \
  --model_path "$MODEL_PATH" \
  --images "$IMAGES" \
  --iteration "$ITERATION" --split "$SPLIT" --view_index "$VIEW_INDEX" \
  $( [[ "$USE_CFG" -eq 1 ]] && echo "" || echo "--no_cfg" ) \
  --out_dir "$OUT_DIR" \
  $( [[ -n "$PLY_PATH" ]] && echo "--ply_path $PLY_PATH" ) \
  "${EXTRA_ARGS[@]}"
