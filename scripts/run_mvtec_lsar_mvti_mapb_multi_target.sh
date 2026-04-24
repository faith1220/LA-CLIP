#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_DIR="${DATA_DIR:-${ROOT_DIR}/data_local}"
CLIP_DOWNLOAD_DIR="${CLIP_DOWNLOAD_DIR:-${ROOT_DIR}/download/clip}"
RUN_TAG="${RUN_TAG:-clean_mvtec_lsar_mvti_mapb_multi_target_$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-${ROOT_DIR}/runs/${RUN_TAG}}"
LOG_DIR="${LOG_DIR:-${RUN_ROOT}/mvtec_to_multi_target}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/stdout.log}"
LAUNCH_MODE="${LAUNCH_MODE:-auto}"

GPU="${GPU:-0}"
SESSION="${SESSION:-clean_mvtec_mapb_multi_target}"

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
TEST_DATASETS="${TEST_DATASETS:-visa dagm dtd isic clinic colon brainmri br35h kvasir}"
WEIGHT_DIR="${WEIGHT_DIR:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

mkdir -p "${LOG_DIR}"

read -r -a FEATURE_LAYERS_ARR <<< "${FEATURE_LAYERS}"
read -r -a MEMORY_LAYERS_ARR <<< "${MEMORY_LAYERS}"
read -r -a TEST_DATASETS_ARR <<< "${TEST_DATASETS}"

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
  --log_dir "${LOG_DIR}"
  --dataset "mvtec"
  --test_dataset "${TEST_DATASETS_ARR[@]}"
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
  local command_text
  command_text="$(printf '%q ' "${PYTHON_BIN}" "${COMMON_ARGS[@]}")"
  printf 'cd %q && CUDA_VISIBLE_DEVICES=%q PYTHONUNBUFFERED=1 %s 2>&1 | tee %q' \
    "${ROOT_DIR}" "${GPU}" "${command_text}" "${LOG_FILE}"
}

COMMAND_FILE="${LOG_DIR}/launch_cmd.txt"
PID_FILE="${LOG_DIR}/pid.txt"
LAUNCH_CMD="$(build_launch_cmd)"
printf '%s\n' "${LAUNCH_CMD}" > "${COMMAND_FILE}"

cat > "${RUN_ROOT}/launch.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
${LAUNCH_CMD}
EOF

stop_screen_session "${SESSION}"
stop_pid_file "${PID_FILE}"

if [[ "$(resolve_launch_mode)" == "screen" ]]; then
  screen -dmS "${SESSION}" bash -lc "${LAUNCH_CMD}"
elif [[ "$(resolve_launch_mode)" == "nohup" ]]; then
  nohup bash -lc "${LAUNCH_CMD}" >/dev/null 2>&1 &
  echo "$!" > "${PID_FILE}"
else
  echo "Unsupported LAUNCH_MODE=$(resolve_launch_mode)" >&2
  exit 1
fi

echo "RUN_ROOT=${RUN_ROOT}"
echo "LOG_DIR=${LOG_DIR}"
echo "SESSION=${SESSION}"
echo "LAUNCH_MODE=$(resolve_launch_mode)"
