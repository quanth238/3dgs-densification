#!/usr/bin/env bash
set -euo pipefail

# Cross-dataset + multi-iteration sweep for verify_oracle.py
# Edit DATASETS and ITERATIONS to match your setup.

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"

# Format: "name:dataset_path:model_path"
DATASETS=(
  "bonsai:/home/tri-dev/dev/namn_workspace/dataset/mipnerf360/bonsai:./experiment/bonsai"
  # "bicycle:/home/tri-dev/dev/namn_workspace/dataset/mipnerf360/bicycle:./experiment/bicycle"
  # "room:/home/tri-dev/dev/namn_workspace/dataset/mipnerf360/room:./experiment/room"
)

# Iterations to test. Leave empty to use latest (-1).
ITERATIONS=(1000 3000 7000)

AUTO_TRAIN=1
# Extra args for training (e.g., -r 4)
DEFAULT_IMAGES="images_4"
TRAIN_EXTRA_ARGS=("--images" "${DEFAULT_IMAGES}" "-r" "4" "--disable_viewer" "--quiet")

EPS_LIST="1e-3,2e-3,5e-3"
NUM_SAMPLES=300
NUM_VIEWS=5
VIEW_STRIDE=10
VIEW_INDEX=0
SPLIT="train"
LAMBDA_DSSIM=0
CENTRAL_DIFF=1
ALPHA_MIN=0.05
ALPHA_MAX=0.95
TOPK=20

list_available_iters() {
  local pc_dir="$1"
  if [[ ! -d "${pc_dir}" ]]; then
    return
  fi
  ls -1 "${pc_dir}" | sed -n 's/^iteration_//p' | sort -n
}

train_if_missing() {
  local name="$1"
  local dataset_path="$2"
  local model_path="$3"
  local pc_dir="${model_path}/point_cloud"
  shift 3
  local missing_iters=("$@")

  if [[ ${#missing_iters[@]} -eq 0 ]]; then
    return
  fi
  if [[ "${AUTO_TRAIN}" != "1" ]]; then
    echo "Warn ${name}: missing iterations (${missing_iters[*]}). AUTO_TRAIN=0, skipping training."
    return
  fi

  local max_iter
  max_iter=$(printf '%s\n' "${missing_iters[@]}" | sort -n | tail -1)

  echo "Auto-train ${name}: missing iterations (${missing_iters[*]}), training to ${max_iter}."

  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python train.py \
    --source_path "${dataset_path}" \
    --model_path "${model_path}" \
    --iterations "${max_iter}" \
    --save_iterations "${missing_iters[@]}" \
    "${TRAIN_EXTRA_ARGS[@]}"
}

run_verify() {
  local name="$1"
  local dataset_path="$2"
  local model_path="$3"
  local iter="$4"
  local out_dir="${model_path}/oracle_verify/${name}/iter_${iter}"

  echo "=== ${name} | iter=${iter} ==="
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python verify_oracle.py \
    --source_path "${dataset_path}" \
    --model_path "${model_path}" \
    --images "${DEFAULT_IMAGES}" \
    --iteration "${iter}" \
    --num_samples "${NUM_SAMPLES}" \
    --eps_list "${EPS_LIST}" \
    --num_views "${NUM_VIEWS}" --view_stride "${VIEW_STRIDE}" --view_index "${VIEW_INDEX}" \
    --split "${SPLIT}" \
    --lambda_dssim "${LAMBDA_DSSIM}" \
    --central_diff \
    --alpha_min "${ALPHA_MIN}" --alpha_max "${ALPHA_MAX}" \
    --topk "${TOPK}" \
    --out_dir "${out_dir}"
}

for entry in "${DATASETS[@]}"; do
  IFS=":" read -r name dataset_path model_path <<< "${entry}"

  if [[ ! -d "${dataset_path}" ]]; then
    echo "Skip ${name}: dataset path not found: ${dataset_path}"
    continue
  fi
  if [[ ! -d "${model_path}" ]]; then
    echo "Skip ${name}: model path not found: ${model_path}"
    continue
  fi

  pc_dir="${model_path}/point_cloud"
  if [[ ! -d "${pc_dir}" ]]; then
    echo "Warn ${name}: no point_cloud at ${pc_dir}"
    train_if_missing "${name}" "${dataset_path}" "${model_path}" "${ITERATIONS[@]}"
  fi

  # Determine available iterations
  mapfile -t available_iters < <(list_available_iters "${pc_dir}")
  if [[ ${#available_iters[@]} -eq 0 ]]; then
    echo "Warn ${name}: no iteration_* folders in ${pc_dir}"
    train_if_missing "${name}" "${dataset_path}" "${model_path}" "${ITERATIONS[@]}"
    mapfile -t available_iters < <(list_available_iters "${pc_dir}")
    if [[ ${#available_iters[@]} -eq 0 ]]; then
      echo "Skip ${name}: still no iteration_* folders after training."
      continue
    fi
  fi

  if [[ ${#ITERATIONS[@]} -eq 0 ]]; then
    latest="${available_iters[-1]}"
    run_verify "${name}" "${dataset_path}" "${model_path}" "${latest}"
    continue
  fi

  missing=()
  for iter in "${ITERATIONS[@]}"; do
    if ! printf '%s\n' "${available_iters[@]}" | grep -q "^${iter}$"; then
      missing+=("${iter}")
    fi
  done

  if [[ ${#missing[@]} -gt 0 ]]; then
    echo "Warn ${name}: missing iterations (${missing[*]}) in ${pc_dir}."
    train_if_missing "${name}" "${dataset_path}" "${model_path}" "${missing[@]}"
    mapfile -t available_iters < <(list_available_iters "${pc_dir}")
  fi

  for iter in "${ITERATIONS[@]}"; do
    if printf '%s\n' "${available_iters[@]}" | grep -q "^${iter}$"; then
      run_verify "${name}" "${dataset_path}" "${model_path}" "${iter}"
    else
      echo "Warn ${name}: iteration_${iter} not found in ${pc_dir}, skipping."
    fi
  done
done
