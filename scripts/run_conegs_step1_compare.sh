#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_conegs_step1_compare.sh \
    --source_path /path/to/dataset/mipnerf360 \
    --scene_name bonsai \
    [--images images_4] \
    [--iterations 30000] \
    [--quantiles 0.3,0.5,0.7] \
    [--model_base ./reference/conegs/experiment/conegs_bonsai_base] \
    [--model_md ./reference/conegs/experiment/conegs_bonsai_md] \
    [--view_index 0] \
    [--triang_views 1,2,3,4,5] \
    [--num_pixels 150] \
    [--triang_min_ncc 0.3] \
    [--k_list 50,100,200,400] \
    [--num_trials 20] \
    [--out_file ./experiment/conegs_step1_compare.csv] \
    [--skip_train] \
    [--skip_delta]

Notes:
- Runs ConeGS baseline (median depth) and multi-depth (quantiles) back-to-back.
- Then runs delta-oracle on both models and writes a single CSV summary.
- Set CUDA_VISIBLE_DEVICES=3 by default if not already set.
EOF
}

SOURCE_PATH=""
SCENE_NAME=""
IMAGES="images_4"
ITERATIONS="30000"
QUANTILES="0.3,0.5,0.7"
MODEL_BASE="./reference/conegs/experiment/conegs_bonsai_base"
MODEL_MD="./reference/conegs/experiment/conegs_bonsai_md"
VIEW_INDEX="0"
TRIANG_VIEWS="1,2,3,4,5"
NUM_PIXELS="150"
TRIANG_MIN_NCC="0.3"
K_LIST="50,100,200,400"
NUM_TRIALS="20"
OUT_FILE="./experiment/conegs_step1_compare.csv"
SKIP_TRAIN=0
SKIP_DELTA=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source_path) SOURCE_PATH="$2"; shift 2 ;;
    --scene_name) SCENE_NAME="$2"; shift 2 ;;
    --images) IMAGES="$2"; shift 2 ;;
    --iterations) ITERATIONS="$2"; shift 2 ;;
    --quantiles) QUANTILES="$2"; shift 2 ;;
    --model_base) MODEL_BASE="$2"; shift 2 ;;
    --model_md) MODEL_MD="$2"; shift 2 ;;
    --view_index) VIEW_INDEX="$2"; shift 2 ;;
    --triang_views) TRIANG_VIEWS="$2"; shift 2 ;;
    --num_pixels) NUM_PIXELS="$2"; shift 2 ;;
    --triang_min_ncc) TRIANG_MIN_NCC="$2"; shift 2 ;;
    --k_list) K_LIST="$2"; shift 2 ;;
    --num_trials) NUM_TRIALS="$2"; shift 2 ;;
    --out_file) OUT_FILE="$2"; shift 2 ;;
    --skip_train) SKIP_TRAIN=1; shift 1 ;;
    --skip_delta) SKIP_DELTA=1; shift 1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1"; usage; exit 1 ;;
  esac
done

if [[ -z "$SOURCE_PATH" || -z "$SCENE_NAME" ]]; then
  echo "Error: --source_path and --scene_name are required."
  usage
  exit 1
fi

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES=3
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ "$SKIP_TRAIN" -eq 0 ]]; then
  echo "=== ConeGS baseline (median depth) ==="
  bash "${SCRIPT_DIR}/run_conegs_baseline.sh" \
    --source_path "$SOURCE_PATH" \
    --model_path "$MODEL_BASE" \
    --images "$IMAGES" \
    --scene_name "$SCENE_NAME" \
    --extra "optimization.iterations=${ITERATIONS}"

  echo "=== ConeGS multi-depth (quantiles=${QUANTILES}) ==="
  bash "${SCRIPT_DIR}/run_conegs_baseline.sh" \
    --source_path "$SOURCE_PATH" \
    --model_path "$MODEL_MD" \
    --images "$IMAGES" \
    --scene_name "$SCENE_NAME" \
    --extra "optimization.iterations=${ITERATIONS} nerf_model.cumsum_quantiles=[${QUANTILES}]"
fi

if [[ "$SKIP_DELTA" -eq 0 ]]; then
  echo "=== Delta-oracle (baseline) ==="
  bash "${SCRIPT_DIR}/run_conegs_delta_oracle.sh" \
    --source_path "${SOURCE_PATH%/}/${SCENE_NAME}" \
    --model_path "$MODEL_BASE" \
    --iteration "$ITERATIONS" \
    --images "$IMAGES" \
    --split train --view_index "$VIEW_INDEX" \
    --out_dir "${MODEL_BASE}/oracle_verify/delta_oracle" \
    --extra "--proposal_mode triangulate --triang_view_indices ${TRIANG_VIEWS} --num_pixels ${NUM_PIXELS} --triang_min_ncc ${TRIANG_MIN_NCC} --k_list ${K_LIST} --num_trials ${NUM_TRIALS}"

  echo "=== Delta-oracle (multi-depth) ==="
  bash "${SCRIPT_DIR}/run_conegs_delta_oracle.sh" \
    --source_path "${SOURCE_PATH%/}/${SCENE_NAME}" \
    --model_path "$MODEL_MD" \
    --iteration "$ITERATIONS" \
    --images "$IMAGES" \
    --split train --view_index "$VIEW_INDEX" \
    --out_dir "${MODEL_MD}/oracle_verify/delta_oracle" \
    --extra "--proposal_mode triangulate --triang_view_indices ${TRIANG_VIEWS} --num_pixels ${NUM_PIXELS} --triang_min_ncc ${TRIANG_MIN_NCC} --k_list ${K_LIST} --num_trials ${NUM_TRIALS}"
fi

mkdir -p "$(dirname "$OUT_FILE")"
echo "method,summary_path,g_min,delta_kmax,delta_norm,visible_ratio,num_candidates" > "$OUT_FILE"

extract_summary() {
  local method="$1"
  local summary_path="$2"
  "$PYTHON_BIN" - <<'PY' "$method" "$summary_path"
import csv, sys, math, os
method = sys.argv[1]
path = sys.argv[2]
g_min = float("nan")
visible = float("nan")
num_candidates = float("nan")
delta_kmax = float("nan")
if not path or not os.path.isfile(path):
    print(f"{method},{path},nan,nan,nan,nan,nan")
    sys.exit(0)

with open(path, "r", encoding="utf-8") as f:
    lines = [ln.strip() for ln in f if ln.strip()]

in_per = False
in_byk = False
last_byk = None
for ln in lines:
    if ln.startswith("#"):
        in_per = "per_run" in ln
        in_byk = "by_K" in ln
        continue
    if in_per:
        row = next(csv.reader([ln]))
        if len(row) >= 12:
            try:
                g_min = float(row[8])
                visible = float(row[7])
                num_candidates = float(row[12])
            except Exception:
                pass
        in_per = False
    elif in_byk:
        row = next(csv.reader([ln]))
        if len(row) >= 2 and row[0] != "K":
            last_byk = row
        elif row[0] == "K":
            continue

if last_byk:
    try:
        delta_kmax = float(last_byk[1])
    except Exception:
        pass

delta_norm = float("nan")
if g_min and not math.isnan(g_min) and g_min != 0 and not math.isnan(delta_kmax):
    delta_norm = abs(delta_kmax) / abs(g_min)

print(f"{method},{path},{g_min},{delta_kmax},{delta_norm},{visible},{num_candidates}")
PY
}

BASE_SUM="${MODEL_BASE}/oracle_verify/delta_oracle/delta_oracle_summary.csv"
MD_SUM="${MODEL_MD}/oracle_verify/delta_oracle/delta_oracle_summary.csv"

extract_summary "baseline" "$BASE_SUM" >> "$OUT_FILE"
extract_summary "multidepth" "$MD_SUM" >> "$OUT_FILE"

echo "Wrote ${OUT_FILE}"
