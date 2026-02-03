#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_delta_oracle_stability.sh \
    --source_path /path/to/dataset \
    --model_path ./experiment/scene \
    --images images_4 \
    --iteration 3000 --split train \
    --view_indices "10,20,40" \
    --proposal_mode gaussian_perturb --w_mode resp \
    --weight_num_views 5 --weight_view_stride 10 \
    --rc 5.0 --rf 0.25 --kc 200 --kf 400 \
    --k_list "50,100,200,400" --num_trials 20 \
    --out_dir ./experiment/scene/oracle_verify/delta_oracle_stability
EOF
}

SOURCE_PATH=""
MODEL_PATH=""
IMAGES=""
ITERATION="3000"
SPLIT="train"
VIEW_INDICES="10,20,40"
NUM_PIXELS="200"
ALPHA0="1e-2"
SCALE_FACTOR="3.0"
DEPTH_MULTS="0.5,0.7,0.9,1.0"
K_LIST="50,100,200,400"
NUM_TRIALS="20"
PROPOSAL_MODE="gaussian_perturb"
W_MODE="resp"
RC="5.0"
RF="0.25"
KC="200"
KF="400"
WEIGHT_NUM_VIEWS="5"
WEIGHT_VIEW_STRIDE="10"
WEIGHT_VIEW_INDICES=""
USE_DATASET_DEPTH="0"
USE_PCD_DEPTH="0"
PCD_FALLBACK="0"
OUT_DIR=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source_path) SOURCE_PATH="$2"; shift 2;;
    --model_path) MODEL_PATH="$2"; shift 2;;
    --images) IMAGES="$2"; shift 2;;
    --iteration) ITERATION="$2"; shift 2;;
    --split) SPLIT="$2"; shift 2;;
    --view_indices) VIEW_INDICES="$2"; shift 2;;
    --num_pixels) NUM_PIXELS="$2"; shift 2;;
    --alpha0) ALPHA0="$2"; shift 2;;
    --scale_factor) SCALE_FACTOR="$2"; shift 2;;
    --depth_multipliers) DEPTH_MULTS="$2"; shift 2;;
    --k_list) K_LIST="$2"; shift 2;;
    --num_trials) NUM_TRIALS="$2"; shift 2;;
    --proposal_mode) PROPOSAL_MODE="$2"; shift 2;;
    --w_mode) W_MODE="$2"; shift 2;;
    --rc) RC="$2"; shift 2;;
    --rf) RF="$2"; shift 2;;
    --kc) KC="$2"; shift 2;;
    --kf) KF="$2"; shift 2;;
    --weight_num_views) WEIGHT_NUM_VIEWS="$2"; shift 2;;
    --weight_view_stride) WEIGHT_VIEW_STRIDE="$2"; shift 2;;
    --weight_view_indices) WEIGHT_VIEW_INDICES="$2"; shift 2;;
    --use_dataset_depth) USE_DATASET_DEPTH="1"; shift 1;;
    --use_pcd_depth) USE_PCD_DEPTH="1"; shift 1;;
    --pcd_fallback_render) PCD_FALLBACK="1"; shift 1;;
    --out_dir) OUT_DIR="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown arg: $1"; usage; exit 1;;
  esac
done

if [[ -z "${SOURCE_PATH}" || -z "${MODEL_PATH}" ]]; then
  echo "Error: --source_path and --model_path are required."
  usage
  exit 1
fi

OUT_DIR="${OUT_DIR:-${MODEL_PATH}/oracle_verify/delta_oracle_stability}"
mkdir -p "${OUT_DIR}"

COMMON_ARGS=(--source_path "${SOURCE_PATH}" --model_path "${MODEL_PATH}" --split "${SPLIT}" \
  --num_pixels "${NUM_PIXELS}" --alpha0 "${ALPHA0}" --scale_factor "${SCALE_FACTOR}" \
  --depth_multipliers "${DEPTH_MULTS}" --k_list "${K_LIST}" --num_trials "${NUM_TRIALS}" \
  --proposal_mode "${PROPOSAL_MODE}" --w_mode "${W_MODE}" --rc "${RC}" --rf "${RF}" \
  --kc "${KC}" --kf "${KF}" --weight_num_views "${WEIGHT_NUM_VIEWS}" \
  --weight_view_stride "${WEIGHT_VIEW_STRIDE}")

if [[ -n "${IMAGES}" ]]; then
  COMMON_ARGS+=(--images "${IMAGES}")
fi
if [[ -n "${WEIGHT_VIEW_INDICES}" ]]; then
  COMMON_ARGS+=(--weight_view_indices "${WEIGHT_VIEW_INDICES}")
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

IFS=',' read -ra VIEW_LIST <<< "${VIEW_INDICES}"

SUMMARY_CSV="${OUT_DIR}/stability_summary.csv"
echo "iteration,view_index,g_min,delta_kmax,delta_norm,visible_ratio,summary_path" > "${SUMMARY_CSV}"

for v in "${VIEW_LIST[@]}"; do
  v="$(echo "$v" | xargs)"
  out_view="${OUT_DIR}/view_${v}"
  CUDA_VISIBLE_DEVICES=3 python scripts/estimate_delta_oracle.py \
    "${COMMON_ARGS[@]}" \
    --iteration "${ITERATION}" \
    --view_index "${v}" \
    --out_dir "${out_view}"

  # Extract g_min, visible_ratio, and delta(Kmax) from the summary CSV.
  sum_csv="${out_view}/delta_oracle_summary.csv"
  parsed=$(python - "${sum_csv}" <<'PY'
import csv
import sys

path = sys.argv[1]
rows = []
with open(path, "r", encoding="utf-8") as f:
    for line in f:
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        rows.append(next(csv.reader([s])))

g_min = "nan"
visible = "nan"
delta_kmax = "nan"

if len(rows) >= 2:
    header = rows[0]
    data = rows[1]
    if "g_min" in header:
        g_min = data[header.index("g_min")]
    if "visible_ratio" in header:
        visible = data[header.index("visible_ratio")]

idx = None
for i, r in enumerate(rows):
    if len(r) >= 2 and r[0] == "K" and "delta_mean" in r:
        idx = i
        break
if idx is not None:
    for r in rows[idx + 1 :]:
        if len(r) >= 2:
            try:
                float(r[0])
            except Exception:
                continue
            delta_kmax = r[1]

try:
    g = float(g_min)
    d = float(delta_kmax)
    delta_norm = d / abs(g) if abs(g) > 0 else float("nan")
except Exception:
    delta_norm = float("nan")

print(f"{g_min},{delta_kmax},{delta_norm},{visible}")
PY
)
  IFS=',' read -r g_min delta_kmax delta_norm visible <<< "${parsed}"
  echo "${ITERATION},${v},${g_min},${delta_kmax},${delta_norm},${visible},${sum_csv}" >> "${SUMMARY_CSV}"
done

echo "Wrote ${SUMMARY_CSV}"
