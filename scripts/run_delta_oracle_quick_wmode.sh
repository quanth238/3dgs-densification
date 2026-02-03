#!/usr/bin/env bash
set -euo pipefail

# Quick test for w_mode=error_opacity (and error) on view 20/40.

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_delta_oracle_quick_wmode.sh \
    --source_path /path/to/dataset \
    --model_path ./experiment/scene \
    [--images images_4] \
    [--iteration 3000] \
    [--views "20,40"] \
    [--alpha0 1e-2 --scale_factor 1.0] \
    [--k_list "50,100,200,400"] [--num_trials 50] \
    [--weight_num_views 1 --weight_view_stride 1 --weight_view_indices ""] \
    [--rc 6.0 --rf 0.25 --kc 200 --kf 400]
EOF
}

SOURCE_PATH=""
MODEL_PATH=""
IMAGES=""
ITERATION="3000"
VIEWS="20,40"
ALPHA0="1e-2"
SCALE_FACTOR="1.0"
K_LIST="50,100,200,400"
NUM_TRIALS="50"
WEIGHT_NUM_VIEWS=""
WEIGHT_VIEW_STRIDE=""
WEIGHT_VIEW_INDICES=""
RC="6.0"
RF="0.25"
KC="200"
KF="400"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source_path) SOURCE_PATH="$2"; shift 2;;
    --model_path) MODEL_PATH="$2"; shift 2;;
    --images) IMAGES="$2"; shift 2;;
    --iteration) ITERATION="$2"; shift 2;;
    --views) VIEWS="$2"; shift 2;;
    --alpha0) ALPHA0="$2"; shift 2;;
    --scale_factor) SCALE_FACTOR="$2"; shift 2;;
    --k_list) K_LIST="$2"; shift 2;;
    --num_trials) NUM_TRIALS="$2"; shift 2;;
    --weight_num_views) WEIGHT_NUM_VIEWS="$2"; shift 2;;
    --weight_view_stride) WEIGHT_VIEW_STRIDE="$2"; shift 2;;
    --weight_view_indices) WEIGHT_VIEW_INDICES="$2"; shift 2;;
    --rc) RC="$2"; shift 2;;
    --rf) RF="$2"; shift 2;;
    --kc) KC="$2"; shift 2;;
    --kf) KF="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown arg: $1"; usage; exit 1;;
  esac
done

if [[ -z "${SOURCE_PATH}" || -z "${MODEL_PATH}" ]]; then
  echo "Error: --source_path and --model_path are required."
  usage
  exit 1
fi

OUT_BASE="${MODEL_PATH}/oracle_verify/delta_oracle_quick_wmode"
mkdir -p "${OUT_BASE}"
SUMMARY_CSV="${OUT_BASE}/quick_wmode_summary.csv"

echo "iteration,view_index,w_mode,rc,rf,kc,kf,g_min,delta_kmax,delta_norm,visible_ratio,summary_path" > "${SUMMARY_CSV}"

COMMON_ARGS=(--source_path "${SOURCE_PATH}" --model_path "${MODEL_PATH}" \
  --split train --iteration "${ITERATION}" --alpha0 "${ALPHA0}" --scale_factor "${SCALE_FACTOR}" \
  --k_list "${K_LIST}" --num_trials "${NUM_TRIALS}" \
  --proposal_mode gaussian_perturb \
  --rc "${RC}" --rf "${RF}" --kc "${KC}" --kf "${KF}")

if [[ -n "${IMAGES}" ]]; then
  COMMON_ARGS+=(--images "${IMAGES}")
fi
if [[ -n "${WEIGHT_NUM_VIEWS}" ]]; then
  COMMON_ARGS+=(--weight_num_views "${WEIGHT_NUM_VIEWS}")
fi
if [[ -n "${WEIGHT_VIEW_STRIDE}" ]]; then
  COMMON_ARGS+=(--weight_view_stride "${WEIGHT_VIEW_STRIDE}")
fi
if [[ -n "${WEIGHT_VIEW_INDICES}" ]]; then
  COMMON_ARGS+=(--weight_view_indices "${WEIGHT_VIEW_INDICES}")
fi

run_one() {
  local view="$1"
  local wmode="$2"
  local tag="w_${wmode}_rc_${RC}_rf_${RF}_kc_${KC}_kf_${KF}"
  local out_dir="${OUT_BASE}/${tag}/view_${view}"
  python scripts/estimate_delta_oracle.py \
    "${COMMON_ARGS[@]}" \
    --w_mode "${wmode}" \
    --view_index "${view}" \
    --out_dir "${out_dir}"

  python - "${out_dir}/delta_oracle_summary.csv" "${ITERATION}" "${view}" "${wmode}" "${RC}" "${RF}" "${KC}" "${KF}" >> "${SUMMARY_CSV}" <<'PY'
import csv, math, sys
path, it, view, wmode, rc, rf, kc, kf = sys.argv[1:]
per = {}
byk = []
with open(path, "r", encoding="utf-8") as f:
    lines = [ln.strip() for ln in f if ln.strip()]
i = 0
while i < len(lines):
    if lines[i].startswith("#") and "per_run" in lines[i]:
        if i + 2 < len(lines):
            header = next(csv.reader([lines[i+1]]))
            row = next(csv.reader([lines[i+2]]))
            per = dict(zip(header, row))
            i += 3
            continue
    if lines[i].startswith("#") and "by_K" in lines[i]:
        i += 1
        if i < len(lines) and lines[i].startswith("K,"):
            i += 1
        while i < len(lines) and not lines[i].startswith("#"):
            row = next(csv.reader([lines[i]]))
            if len(row) >= 2:
                byk.append((int(float(row[0])), float(row[1])))
            i += 1
        continue
    i += 1

def fnum(x):
    try:
        return float(x)
    except Exception:
        return float("nan")

gmin = fnum(per.get("g_min","nan"))
visible = fnum(per.get("visible_ratio","nan"))
kmax = None
delta_kmax = float("nan")
if byk:
    byk.sort(key=lambda x: x[0])
    kmax, delta_kmax = byk[-1]
delta_norm = delta_kmax / (abs(gmin) + 1e-12) if not math.isnan(delta_kmax) and abs(gmin) > 0 else float("nan")

print(",".join([
    str(it), str(view), wmode, str(rc), str(rf), str(kc), str(kf),
    str(gmin), str(delta_kmax), str(delta_norm), str(visible), path
]))
PY
}

IFS=',' read -ra VIEW_LIST <<< "${VIEWS}"
for v in "${VIEW_LIST[@]}"; do
  v="$(echo "$v" | xargs)"
  run_one "${v}" "error_opacity"
  run_one "${v}" "error"
done

echo "Wrote ${SUMMARY_CSV}"
