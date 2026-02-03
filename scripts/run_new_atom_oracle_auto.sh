#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_new_atom_oracle_auto.sh \
    --source_path /path/to/dataset \
    --model_path ./experiment/scene \
    [--images images_4] [--iteration 3000] [--split train] [--view_index 0] \
    [--num_pixels 200] [--num_samples 100] [--use_dataset_depth]

This script sweeps alpha/epsilon/scale/depth-multipliers and records summary CSV.
EOF
}

SOURCE_PATH=""
MODEL_PATH=""
IMAGES=""
ITERATION="-1"
SPLIT="train"
VIEW_INDEX="0"
NUM_PIXELS="200"
NUM_SAMPLES="100"
USE_DATASET_DEPTH="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source_path) SOURCE_PATH="$2"; shift 2;;
    --model_path) MODEL_PATH="$2"; shift 2;;
    --images) IMAGES="$2"; shift 2;;
    --iteration) ITERATION="$2"; shift 2;;
    --split) SPLIT="$2"; shift 2;;
    --view_index) VIEW_INDEX="$2"; shift 2;;
    --num_pixels) NUM_PIXELS="$2"; shift 2;;
    --num_samples) NUM_SAMPLES="$2"; shift 2;;
    --use_dataset_depth) USE_DATASET_DEPTH="1"; shift 1;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown arg: $1"; usage; exit 1;;
  esac
done

if [[ -z "${SOURCE_PATH}" || -z "${MODEL_PATH}" ]]; then
  echo "Error: --source_path and --model_path are required."
  usage
  exit 1
fi

BASE_OUT="${MODEL_PATH}/oracle_verify/new_atom/auto"
mkdir -p "${BASE_OUT}"

CSV_PATH="${BASE_OUT}/auto_sweep.csv"
echo "tag,use_dataset_depth,alpha0,epsilon,scale_factor,depth_multipliers,visible_ratio,r2,pearson,spearman,depth_source" > "${CSV_PATH}"

ALPHAS=("1e-3" "2e-3" "5e-3" "1e-2")
EPSILONS=("1e-3" "2e-3" "5e-3")
SCALES=("1.0" "1.5" "2.0" "3.0")
DEPTH_MULTS=("0.5,0.7,0.9,1.0" "0.7,1.0,1.3" "0.8,1.0,1.2,1.4")

COMMON_ARGS=(--source_path "${SOURCE_PATH}" --model_path "${MODEL_PATH}" --iteration "${ITERATION}" \
  --split "${SPLIT}" --view_index "${VIEW_INDEX}" --num_pixels "${NUM_PIXELS}" --num_samples "${NUM_SAMPLES}" --central_diff)

if [[ -n "${IMAGES}" ]]; then
  COMMON_ARGS+=(--images "${IMAGES}")
fi

if [[ "${USE_DATASET_DEPTH}" == "1" ]]; then
  COMMON_ARGS+=(--use_dataset_depth)
fi

run_one() {
  local alpha="$1"
  local eps="$2"
  local scale="$3"
  local dm="$4"
  local tag="a${alpha}_e${eps}_s${scale}_d$(echo "${dm}" | tr ',' '-')"
  local out_dir="${BASE_OUT}/${tag}"

  python scripts/verify_new_atom_oracle.py \
    "${COMMON_ARGS[@]}" \
    --alpha0 "${alpha}" \
    --epsilon "${eps}" \
    --scale_factor "${scale}" \
    --depth_multipliers "${dm}" \
    --out_dir "${out_dir}" \
    --no_plot

  if [[ ! -f "${out_dir}/summary.txt" ]]; then
    echo "Missing summary for ${tag}"
    return
  fi

  python - "${out_dir}/summary.txt" "${tag}" "${alpha}" "${eps}" "${scale}" "${dm}" "${USE_DATASET_DEPTH}" >> "${CSV_PATH}" <<'PY'
import sys
path, tag, alpha, eps, scale, dm, use_depth = sys.argv[1:]
vals = {}
with open(path, "r", encoding="utf-8") as f:
    for line in f:
        if ":" in line:
            k, v = line.strip().split(":", 1)
            vals[k.strip()] = v.strip()
def g(key):
    return vals.get(key, "")
print(",".join([
    tag, use_depth, alpha, eps, scale, dm,
    g("visible_ratio"), g("r2"), g("pearson"), g("spearman"), g("depth_source")
]))
PY
}

for alpha in "${ALPHAS[@]}"; do
  for eps in "${EPSILONS[@]}"; do
    for scale in "${SCALES[@]}"; do
      for dm in "${DEPTH_MULTS[@]}"; do
        run_one "${alpha}" "${eps}" "${scale}" "${dm}"
      done
    done
  done
done

python - "${CSV_PATH}" <<'PY'
import sys, csv, math
path = sys.argv[1]
rows = []
with open(path, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for r in reader:
        def fnum(x):
            try:
                return float(x)
            except:
                return float("nan")
        r["_visible"] = fnum(r.get("visible_ratio", "nan"))
        r["_pearson"] = fnum(r.get("pearson", "nan"))
        r["_r2"] = fnum(r.get("r2", "nan"))
        rows.append(r)
rows = [r for r in rows if not math.isnan(r["_visible"])]
rows.sort(key=lambda r: (r["_visible"], r["_pearson"], r["_r2"]), reverse=True)
best = rows[0] if rows else None
print("Auto sweep done:", path)
if best:
    print("Best (by visible_ratio, pearson, r2):")
    print(best)
PY
