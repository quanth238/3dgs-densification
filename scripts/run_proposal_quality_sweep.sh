#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_proposal_quality_sweep.sh \
    --source_path /path/to/dataset \
    --model_path ./experiment/scene \
    [--images images_4] \
    [--iteration 3000] [--split train] \
    [--view_index 0 --num_views 1 --view_stride 1] \
    [--alpha0 1e-2 --scale_factor 1.0] \
    [--k_list "50,100,200,400"] [--num_trials 20] \
    [--rc_list "1.0,2.0,3.0"] [--rf_list "0.25,0.5,1.0"] \
    [--kc_list "200,400"] [--kf_list "200,400"] \
    [--w_modes "error,opacity,error_opacity,uniform"]

This runs a proposal-quality sweep for the Gaussian-perturb proposal mode and
records visible_ratio, tail_frac, delta(K) in one summary CSV.
EOF
}

SOURCE_PATH=""
MODEL_PATH=""
IMAGES=""
ITERATION="3000"
SPLIT="train"
VIEW_INDEX="0"
NUM_VIEWS="1"
VIEW_STRIDE="1"
ALPHA0="1e-2"
SCALE_FACTOR="1.0"
K_LIST="50,100,200,400"
NUM_TRIALS="20"
RC_LIST="1.0,2.0,3.0"
RF_LIST="0.25,0.5,1.0"
KC_LIST="200,400"
KF_LIST="200,400"
W_MODES="error,opacity,error_opacity,uniform"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source_path) SOURCE_PATH="$2"; shift 2;;
    --model_path) MODEL_PATH="$2"; shift 2;;
    --images) IMAGES="$2"; shift 2;;
    --iteration) ITERATION="$2"; shift 2;;
    --split) SPLIT="$2"; shift 2;;
    --view_index) VIEW_INDEX="$2"; shift 2;;
    --num_views) NUM_VIEWS="$2"; shift 2;;
    --view_stride) VIEW_STRIDE="$2"; shift 2;;
    --alpha0) ALPHA0="$2"; shift 2;;
    --scale_factor) SCALE_FACTOR="$2"; shift 2;;
    --k_list) K_LIST="$2"; shift 2;;
    --num_trials) NUM_TRIALS="$2"; shift 2;;
    --rc_list) RC_LIST="$2"; shift 2;;
    --rf_list) RF_LIST="$2"; shift 2;;
    --kc_list) KC_LIST="$2"; shift 2;;
    --kf_list) KF_LIST="$2"; shift 2;;
    --w_modes) W_MODES="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown arg: $1"; usage; exit 1;;
  esac
done

if [[ -z "${SOURCE_PATH}" || -z "${MODEL_PATH}" ]]; then
  echo "Error: --source_path and --model_path are required."
  usage
  exit 1
fi

OUT_BASE="${MODEL_PATH}/oracle_verify/delta_oracle_proposal"
mkdir -p "${OUT_BASE}"
SUMMARY_CSV="${OUT_BASE}/proposal_quality_sweep.csv"

echo "iteration,view_index,rc,rf,kc,kf,w_mode,visible_ratio,tail_frac,g_min,delta_kmax,delta_norm,score,k_max,out_dir" > "${SUMMARY_CSV}"

COMMON_ARGS=(--source_path "${SOURCE_PATH}" --model_path "${MODEL_PATH}" --split "${SPLIT}" \
  --alpha0 "${ALPHA0}" --scale_factor "${SCALE_FACTOR}" \
  --k_list "${K_LIST}" --num_trials "${NUM_TRIALS}" \
  --proposal_mode gaussian_perturb)

if [[ -n "${IMAGES}" ]]; then
  COMMON_ARGS+=(--images "${IMAGES}")
fi

IFS=',' read -ra RC_ARR <<< "${RC_LIST}"
IFS=',' read -ra RF_ARR <<< "${RF_LIST}"
IFS=',' read -ra KC_ARR <<< "${KC_LIST}"
IFS=',' read -ra KF_ARR <<< "${KF_LIST}"
IFS=',' read -ra WM_ARR <<< "${W_MODES}"

for rc in "${RC_ARR[@]}"; do
  rc="$(echo "$rc" | xargs)"
  for rf in "${RF_ARR[@]}"; do
    rf="$(echo "$rf" | xargs)"
    for kc in "${KC_ARR[@]}"; do
      kc="$(echo "$kc" | xargs)"
      for kf in "${KF_ARR[@]}"; do
        kf="$(echo "$kf" | xargs)"
        for wm in "${WM_ARR[@]}"; do
          wm="$(echo "$wm" | xargs)"
          for ((i=0; i<NUM_VIEWS; i++)); do
            view=$((VIEW_INDEX + i * VIEW_STRIDE))
            out_dir="${OUT_BASE}/rc_${rc}_rf_${rf}_kc_${kc}_kf_${kf}_w_${wm}/iter_${ITERATION}/view_${view}"
            echo "=== iter=${ITERATION}, view=${view}, rc=${rc}, rf=${rf}, kc=${kc}, kf=${kf}, w=${wm} ==="
            python scripts/estimate_delta_oracle.py \
              "${COMMON_ARGS[@]}" \
              --iteration "${ITERATION}" \
              --view_index "${view}" \
              --rc "${rc}" \
              --rf "${rf}" \
              --kc "${kc}" \
              --kf "${kf}" \
              --w_mode "${wm}" \
              --out_dir "${out_dir}"

            python - "${out_dir}/delta_oracle_summary.csv" "${ITERATION}" "${view}" "${rc}" "${rf}" "${kc}" "${kf}" "${wm}" "${out_dir}" >> "${SUMMARY_CSV}" <<'PY'
import csv, math, sys
path, it, view, rc, rf, kc, kf, wm, out_dir = sys.argv[1:]
per_run = {}
by_k = []
with open(path, "r", encoding="utf-8") as f:
    lines = [ln.strip() for ln in f if ln.strip()]
i = 0
while i < len(lines):
    line = lines[i]
    if line.startswith("#") and "per_run" in line:
        if i + 2 < len(lines):
            header = next(csv.reader([lines[i + 1]]))
            row = next(csv.reader([lines[i + 2]]))
            per_run = dict(zip(header, row))
            i += 3
            continue
    if line.startswith("#") and "by_K" in line:
        i += 1
        if i < len(lines) and lines[i].startswith("K,"):
            i += 1
        while i < len(lines) and not lines[i].startswith("#"):
            row = next(csv.reader([lines[i]]))
            if len(row) >= 5:
                by_k.append({
                    "K": int(float(row[0])),
                    "delta_mean": float(row[1]),
                })
            i += 1
        continue
    i += 1

def fnum(x):
    try:
        return float(x)
    except Exception:
        return float("nan")

visible = fnum(per_run.get("visible_ratio", "nan"))
tail = fnum(per_run.get("tail_frac", "nan"))
g_min = fnum(per_run.get("g_min", "nan"))

delta_kmax = float("nan")
k_max = 0
if by_k:
    by_k.sort(key=lambda r: r["K"])
    k_max = by_k[-1]["K"]
    delta_kmax = by_k[-1]["delta_mean"]

den = abs(g_min) + 1e-12
delta_norm = delta_kmax / den if not math.isnan(delta_kmax) and den > 0 else float("nan")

score = float("nan")
if not math.isnan(visible) and not math.isnan(tail) and not math.isnan(delta_norm):
    score = visible + tail - delta_norm

print(",".join([
    str(it), str(view), str(rc), str(rf), str(kc), str(kf), wm,
    str(visible), str(tail), str(g_min), str(delta_kmax), str(delta_norm), str(score), str(k_max), out_dir
]))
PY
          done
        done
      done
    done
  done
done

python - "${SUMMARY_CSV}" <<'PY'
import csv, math, sys
path = sys.argv[1]
rows = []
with open(path, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for r in reader:
        try:
            r["_score"] = float(r.get("score", "nan"))
        except Exception:
            r["_score"] = float("nan")
        rows.append(r)
rows = [r for r in rows if not math.isnan(r["_score"])]
rows.sort(key=lambda r: r["_score"], reverse=True)
print("Proposal sweep done:", path)
if rows:
    print("Best by score:")
    print(rows[0])
PY
