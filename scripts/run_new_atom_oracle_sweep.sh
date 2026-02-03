#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_new_atom_oracle_sweep.sh \
    --source_path /path/to/dataset \
    --model_path ./experiment/scene \
    --images images_4 \
    --iters "1000,3000,7000" \
    --view_index 0 --num_views 5 --view_stride 10 \
    --alpha0 1e-2 --epsilon 2e-3 --scale_factor 3.0 \
    --depth_multipliers "0.5,0.7,0.9,1.0" \
    --use_pcd_depth --pcd_fallback_render

Outputs:
  <model_path>/oracle_verify/new_atom/new_atom_sweep_summary.csv
  <model_path>/oracle_verify/new_atom/new_atom_sweep_summary.png
EOF
}

SOURCE_PATH=""
MODEL_PATH=""
IMAGES=""
ITERS="3000"
SPLIT="train"
VIEW_INDEX="0"
NUM_VIEWS="1"
VIEW_STRIDE="1"
NUM_PIXELS="200"
NUM_SAMPLES="100"
ALPHA0="1e-2"
EPSILON="2e-3"
SCALE_FACTOR="3.0"
DEPTH_MULTS="0.5,0.7,0.9,1.0"
USE_DATASET_DEPTH="0"
USE_PCD_DEPTH="0"
PCD_FALLBACK="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source_path) SOURCE_PATH="$2"; shift 2;;
    --model_path) MODEL_PATH="$2"; shift 2;;
    --images) IMAGES="$2"; shift 2;;
    --iters) ITERS="$2"; shift 2;;
    --split) SPLIT="$2"; shift 2;;
    --view_index) VIEW_INDEX="$2"; shift 2;;
    --num_views) NUM_VIEWS="$2"; shift 2;;
    --view_stride) VIEW_STRIDE="$2"; shift 2;;
    --num_pixels) NUM_PIXELS="$2"; shift 2;;
    --num_samples) NUM_SAMPLES="$2"; shift 2;;
    --alpha0) ALPHA0="$2"; shift 2;;
    --epsilon) EPSILON="$2"; shift 2;;
    --scale_factor) SCALE_FACTOR="$2"; shift 2;;
    --depth_multipliers) DEPTH_MULTS="$2"; shift 2;;
    --use_dataset_depth) USE_DATASET_DEPTH="1"; shift 1;;
    --use_pcd_depth) USE_PCD_DEPTH="1"; shift 1;;
    --pcd_fallback_render) PCD_FALLBACK="1"; shift 1;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown arg: $1"; usage; exit 1;;
  esac
done

if [[ -z "${SOURCE_PATH}" || -z "${MODEL_PATH}" ]]; then
  echo "Error: --source_path and --model_path are required."
  usage
  exit 1
fi

OUT_BASE="${MODEL_PATH}/oracle_verify/new_atom"
mkdir -p "${OUT_BASE}"

COMMON_ARGS=(--source_path "${SOURCE_PATH}" --model_path "${MODEL_PATH}" --split "${SPLIT}" \
  --num_pixels "${NUM_PIXELS}" --num_samples "${NUM_SAMPLES}" \
  --alpha0 "${ALPHA0}" --epsilon "${EPSILON}" \
  --scale_factor "${SCALE_FACTOR}" --depth_multipliers "${DEPTH_MULTS}" \
  --central_diff)

if [[ -n "${IMAGES}" ]]; then
  COMMON_ARGS+=(--images "${IMAGES}")
fi
if [[ "${USE_DATASET_DEPTH}" == "1" ]]; then
  COMMON_ARGS+=(--use_dataset_depth)
fi
if [[ "${USE_PCD_DEPTH}" == "1" ]]; then
  COMMON_ARGS+=(--use_pcd_depth)
fi
if [[ "${PCD_FALLBACK}" == "1" ]]; then
  COMMON_ARGS+=(--pcd_fallback_render)
fi

IFS=',' read -ra ITER_LIST <<< "${ITERS}"

for it in "${ITER_LIST[@]}"; do
  it="$(echo "$it" | xargs)"
  for ((i=0; i<NUM_VIEWS; i++)); do
    view=$((VIEW_INDEX + i * VIEW_STRIDE))
    out_dir="${OUT_BASE}/iter_${it}/view_${view}"
    echo "=== iter=${it}, view=${view} ==="
    if ! python scripts/verify_new_atom_oracle.py \
      "${COMMON_ARGS[@]}" \
      --iteration "${it}" \
      --view_index "${view}" \
      --out_dir "${out_dir}" \
      --no_plot; then
      echo "Warn: failed iter=${it}, view=${view}"
    fi
  done
done

python scripts/aggregate_new_atom_oracle.py --base_dir "${OUT_BASE}"
echo "Summary: ${OUT_BASE}/new_atom_sweep_summary.csv"
