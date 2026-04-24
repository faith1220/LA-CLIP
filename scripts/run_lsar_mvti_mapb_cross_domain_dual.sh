#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_DIR="${DATA_DIR:-${ROOT_DIR}/data_local}"
CLIP_DOWNLOAD_DIR="${CLIP_DOWNLOAD_DIR:-${ROOT_DIR}/download/clip}"
RUN_TAG="${RUN_TAG:-clean_lsar_mvti_mapb_$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-${ROOT_DIR}/runs/${RUN_TAG}}"
LAUNCH_MODE="${LAUNCH_MODE:-auto}"

GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
SESSION0="${SESSION0:-clean_m2v_mapb}"
SESSION1="${SESSION1:-clean_v2m_mapb}"

USE_LSAR="${USE_LSAR:-1}"
USE_MVTI="${USE_MVTI:-1}"
USE_MAPB="${USE_MAPB:-1}"
LSAR_BOTTLENECK_RATIO="${LSAR_BOTTLENECK_RATIO:-4}"
LAMBDA_PROTO="${LAMBDA_PROTO:-0.1}"
PROTOTYPE_K="${PROTOTYPE_K:-4}"
PROTOTYPE_MOMENTUM="${PROTOTYPE_MOMENTUM:-0.95}"
PROTOTYPE_TEMPERATURE="${PROTOTYPE_TEMPERATURE:-0.07}"
PROTOTYPE_MAX_SAMPLES="${PROTOTYPE_MAX_SAMPLES:-4096}"
PROTOTYPE_FUSION_ALPHA="${PROTOTYPE_FUSION_ALPHA:-0.25}"

MODEL="${MODEL:-ViT-L/14@336px}"
IMG_SIZE="${IMG_SIZE:-518}"
BATCH_SIZE="${BATCH_SIZE:-8}"
EPOCHS="${EPOCHS:-2}"
SEED="${SEED:-122}"
FEATURE_LAYERS="${FEATURE_LAYERS:-6 12 18 24}"
MEMORY_LAYERS="${MEMORY_LAYERS:-6 12 18 24}"
WEIGHT_DIR="${WEIGHT_DIR:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

mkdir -p "${RUN_ROOT}/mvtec_to_visa" "${RUN_ROOT}/visa_to_mvtec"

read -r -a FEATURE_LAYERS_ARR <<< "${FEATURE_LAYERS}"
read -r -a MEMORY_LAYERS_ARR <<< "${MEMORY_LAYERS}"

if [[ "${USE_MAPB}" == "1" ]]; then
  SCORE_MODE="prototype"
else
  SCORE_MODE="clip"
fi

COMMON_ARGS=(
  main.py
  --data_dir "${DATA_DIR}"
  --clip_download_dir "${CLIP_DOWNLOAD_DIR}"
  --model "${MODEL}"
  --img_size "${IMG_SIZE}"
  --batch_size "${BATCH_SIZE}"
  --epochs "${EPOCHS}"
  --seed "${SEED}"
  --feature_layers "${FEATURE_LAYERS_ARR[@]}"
  --memory_layers "${MEMORY_LAYERS_ARR[@]}"
  --use_lsar "${USE_LSAR}"
  --lsar_bottleneck_ratio "${LSAR_BOTTLENECK_RATIO}"
  --use_mvti "${USE_MVTI}"
  --use_mapb "${USE_MAPB}"
  --score_mode "${SCORE_MODE}"
  --lambda_proto "${LAMBDA_PROTO}"
  --prototype_k "${PROTOTYPE_K}"
  --prototype_momentum "${PROTOTYPE_MOMENTUM}"
  --prototype_temperature "${PROTOTYPE_TEMPERATURE}"
  --prototype_max_samples "${PROTOTYPE_MAX_SAMPLES}"
  --prototype_fusion_alpha "${PROTOTYPE_FUSION_ALPHA}"
)

if [[ -n "${WEIGHT_DIR}" ]]; then
  COMMON_ARGS+=(--weight "${WEIGHT_DIR}")
fi

if [[ -n "${EXTRA_ARGS// }" ]]; then
  read -r -a EXTRA_ARGS_ARR <<< "${EXTRA_ARGS}"
  COMMON_ARGS+=("${EXTRA_ARGS_ARR[@]}")
fi

stop_screen_session() {
  local session_name="$1"
  if ! command -v screen >/dev/null 2>&1; then
    return 0
  fi
  if screen -ls | grep -q "[.]${session_name}[[:space:]]"; then
    screen -S "${session_name}" -X quit
    sleep 1
  fi
}

stop_pid_file() {
  local pid_file="$1"
  if [[ -f "${pid_file}" ]]; then
    local pid
    pid="$(cat "${pid_file}")"
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" 2>/dev/null || true
      sleep 1
    fi
  fi
}

resolve_launch_mode() {
  if [[ "${LAUNCH_MODE}" == "auto" ]]; then
    if command -v screen >/dev/null 2>&1; then
      echo "screen"
    else
      echo "nohup"
    fi
    return 0
  fi
  echo "${LAUNCH_MODE}"
}

build_launch_cmd() {
  local gpu_id="$1"
  local dataset="$2"
  local test_dataset="$3"
  local log_dir="$4"
  local log_file="$5"
  local -a cmd=(
    "${PYTHON_BIN}"
    "${COMMON_ARGS[@]}"
    --log_dir "${log_dir}"
    --dataset "${dataset}"
    --test_dataset "${test_dataset}"
  )
  local command_text
  command_text="$(printf '%q ' "${cmd[@]}")"
  printf 'cd %q && CUDA_VISIBLE_DEVICES=%q PYTHONUNBUFFERED=1 %s 2>&1 | tee %q' \
    "${ROOT_DIR}" "${gpu_id}" "${command_text}" "${log_file}"
}

launch_one() {
  local session_name="$1"
  local gpu_id="$2"
  local dataset="$3"
  local test_dataset="$4"
  local log_dir="$5"
  local log_file="$6"
  local command_file="${log_dir}/launch_cmd.txt"
  local pid_file="${log_dir}/pid.txt"
  local launch_cmd
  local mode

  mkdir -p "${log_dir}"
  launch_cmd="$(build_launch_cmd "${gpu_id}" "${dataset}" "${test_dataset}" "${log_dir}" "${log_file}")"
  printf '%s\n' "${launch_cmd}" > "${command_file}"

  mode="$(resolve_launch_mode)"
  stop_screen_session "${session_name}"
  stop_pid_file "${pid_file}"

  if [[ "${mode}" == "screen" ]]; then
    screen -dmS "${session_name}" bash -lc "${launch_cmd}"
  elif [[ "${mode}" == "nohup" ]]; then
    nohup bash -lc "${launch_cmd}" >/dev/null 2>&1 &
    echo "$!" > "${pid_file}"
  else
    echo "Unsupported LAUNCH_MODE=${mode}" >&2
    exit 1
  fi
}

LOG0="${RUN_ROOT}/mvtec_to_visa/stdout.log"
LOG1="${RUN_ROOT}/visa_to_mvtec/stdout.log"

cat > "${RUN_ROOT}/launch.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
$(build_launch_cmd "${GPU0}" "mvtec" "visa" "${RUN_ROOT}/mvtec_to_visa" "${LOG0}")
$(printf '\n')
$(build_launch_cmd "${GPU1}" "visa" "mvtec" "${RUN_ROOT}/visa_to_mvtec" "${LOG1}")
EOF

launch_one "${SESSION0}" "${GPU0}" "mvtec" "visa" "${RUN_ROOT}/mvtec_to_visa" "${LOG0}"
launch_one "${SESSION1}" "${GPU1}" "visa" "mvtec" "${RUN_ROOT}/visa_to_mvtec" "${LOG1}"

echo "RUN_ROOT=${RUN_ROOT}"
echo "SESSION0=${SESSION0}"
echo "SESSION1=${SESSION1}"
echo "LAUNCH_MODE=$(resolve_launch_mode)"
