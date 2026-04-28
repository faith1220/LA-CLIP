#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_DIR="${DATA_DIR:-${ROOT_DIR}/data_local}"
CLIP_DOWNLOAD_DIR="${CLIP_DOWNLOAD_DIR:-${ROOT_DIR}/download/clip}"

RUN_TAG="${RUN_TAG:-baseline_vs_latest_mapb_$(date +%Y%m%d_%H%M%S)}"
REPORT_ROOT="${REPORT_ROOT:-${ROOT_DIR}/reports/latest_fg_mapb_component/${RUN_TAG}}"
LOG_ROOT="${LOG_ROOT:-${ROOT_DIR}/logs/latest_fg_mapb_component/${RUN_TAG}}"
COMMAND_ROOT="${REPORT_ROOT}/commands"

LAUNCH_MODE="${LAUNCH_MODE:-auto}"
RUN_FILTER="${RUN_FILTER:-all}"

GPU_BASE_M2V="${GPU_BASE_M2V:-0}"
GPU_BASE_V2M="${GPU_BASE_V2M:-1}"
GPU_MAPB_M2V="${GPU_MAPB_M2V:-2}"
GPU_MAPB_V2M="${GPU_MAPB_V2M:-3}"

SESSION_BASE_M2V="${SESSION_BASE_M2V:-baseline_m2v}"
SESSION_BASE_V2M="${SESSION_BASE_V2M:-baseline_v2m}"
SESSION_MAPB_M2V="${SESSION_MAPB_M2V:-latest_mapb_m2v}"
SESSION_MAPB_V2M="${SESSION_MAPB_V2M:-latest_mapb_v2m}"

MODEL="${MODEL:-ViT-L/14@336px}"
IMG_SIZE="${IMG_SIZE:-518}"
BATCH_SIZE="${BATCH_SIZE:-8}"
EPOCHS="${EPOCHS:-2}"
SEED="${SEED:-122}"
FEATURE_LAYERS="${FEATURE_LAYERS:-6 12 18 24}"
MEMORY_LAYERS="${MEMORY_LAYERS:-6 12 18 24}"
WEIGHT_DIR="${WEIGHT_DIR:-}"

NUM_AB_PROMPTS="${NUM_AB_PROMPTS:-4}"
AB_AGG="${AB_AGG:-sum_prob}"

BASE_EXTRA_ARGS="${BASE_EXTRA_ARGS:-}"
MAPB_EXTRA_ARGS="${MAPB_EXTRA_ARGS:-}"

mkdir -p "${REPORT_ROOT}" "${LOG_ROOT}" "${COMMAND_ROOT}"

read -r -a FEATURE_LAYERS_ARR <<< "${FEATURE_LAYERS}"
read -r -a MEMORY_LAYERS_ARR <<< "${MEMORY_LAYERS}"

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
)

if [[ -n "${WEIGHT_DIR}" ]]; then
  COMMON_ARGS+=(--weight "${WEIGHT_DIR}")
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

append_extra_args() {
  local var_name="$1"
  local -n target_array="$2"
  local extra_value="${!var_name:-}"
  if [[ -n "${extra_value// }" ]]; then
    local -a extra_arr
    read -r -a extra_arr <<< "${extra_value}"
    target_array+=("${extra_arr[@]}")
  fi
}

write_command_file() {
  local command_file="$1"
  local launch_cmd="$2"
  cat > "${command_file}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
${launch_cmd}
EOF
}

build_launch_cmd() {
  local exp_name="$1"
  local exp_mode="$2"
  local gpu_id="$3"
  local dataset="$4"
  local test_dataset="$5"

  local report_dir="${REPORT_ROOT}/${exp_name}"
  local stdout_log="${LOG_ROOT}/${exp_name}.log"
  local prompt_diag_json="${report_dir}/prompt_diag.json"

  mkdir -p "${report_dir}"

  local -a cmd=(
    "${PYTHON_BIN}"
    "${COMMON_ARGS[@]}"
    --log_dir "${report_dir}"
    --dataset "${dataset}"
    --test_dataset "${test_dataset}"
    --use_lsar 0
    --use_mvti 0
    --score_mode clip
  )

  if [[ "${exp_mode}" == "baseline" ]]; then
    cmd+=(--fg_prompt off)
    append_extra_args BASE_EXTRA_ARGS cmd
  elif [[ "${exp_mode}" == "latest_mapb" ]]; then
    cmd+=(
      --fg_prompt on
      --num_ab_prompts "${NUM_AB_PROMPTS}"
      --ab_agg "${AB_AGG}"
      --dump_prompt_diag_json "${prompt_diag_json}"
    )
    append_extra_args MAPB_EXTRA_ARGS cmd
  else
    echo "Unsupported exp_mode=${exp_mode}" >&2
    exit 1
  fi

  local command_text
  command_text="$(printf '%q ' "${cmd[@]}")"
  printf 'cd %q && CUDA_VISIBLE_DEVICES=%q PYTHONUNBUFFERED=1 %s 2>&1 | tee %q' \
    "${ROOT_DIR}" "${gpu_id}" "${command_text}" "${stdout_log}"
}

launch_one() {
  local exp_name="$1"
  local session_name="$2"
  local gpu_id="$3"
  local exp_mode="$4"
  local dataset="$5"
  local test_dataset="$6"

  local report_dir="${REPORT_ROOT}/${exp_name}"
  local command_file="${COMMAND_ROOT}/${exp_name}.sh"
  local pid_file="${report_dir}/pid.txt"
  local launch_cmd
  local mode

  launch_cmd="$(build_launch_cmd "${exp_name}" "${exp_mode}" "${gpu_id}" "${dataset}" "${test_dataset}")"
  write_command_file "${command_file}" "${launch_cmd}"

  mode="$(resolve_launch_mode)"
  stop_screen_session "${session_name}"
  stop_pid_file "${pid_file}"

  if [[ "${mode}" == "print" ]]; then
    printf '[PRINT] %s\n' "${launch_cmd}"
    return 0
  fi

  if [[ "${mode}" == "inline" ]]; then
    bash -lc "${launch_cmd}"
    return 0
  fi

  if [[ "${mode}" == "screen" ]]; then
    screen -dmS "${session_name}" bash -lc "${launch_cmd}"
    return 0
  fi

  if [[ "${mode}" == "nohup" ]]; then
    nohup bash -lc "${launch_cmd}" >/dev/null 2>&1 &
    echo "$!" > "${pid_file}"
    return 0
  fi

  echo "Unsupported LAUNCH_MODE=${mode}" >&2
  exit 1
}

run_baseline_group() {
  launch_one "baseline_m2v" "${SESSION_BASE_M2V}" "${GPU_BASE_M2V}" "baseline" "mvtec" "visa"
  launch_one "baseline_v2m" "${SESSION_BASE_V2M}" "${GPU_BASE_V2M}" "baseline" "visa" "mvtec"
}

run_latest_mapb_group() {
  launch_one "latest_mapb_m2v" "${SESSION_MAPB_M2V}" "${GPU_MAPB_M2V}" "latest_mapb" "mvtec" "visa"
  launch_one "latest_mapb_v2m" "${SESSION_MAPB_V2M}" "${GPU_MAPB_V2M}" "latest_mapb" "visa" "mvtec"
}

case "${RUN_FILTER}" in
  baseline)
    run_baseline_group
    ;;
  mapb|latest_mapb)
    run_latest_mapb_group
    ;;
  all)
    run_baseline_group
    run_latest_mapb_group
    ;;
  *)
    echo "Unsupported RUN_FILTER=${RUN_FILTER}. Expected one of: baseline, mapb, latest_mapb, all" >&2
    exit 1
    ;;
esac

cat <<EOF
RUN_TAG=${RUN_TAG}
REPORT_ROOT=${REPORT_ROOT}
LOG_ROOT=${LOG_ROOT}
COMMAND_ROOT=${COMMAND_ROOT}
LAUNCH_MODE=$(resolve_launch_mode)
RUN_FILTER=${RUN_FILTER}
NUM_AB_PROMPTS=${NUM_AB_PROMPTS}
AB_AGG=${AB_AGG}

Experiments:
  baseline_m2v     gpu=${GPU_BASE_M2V}  session=${SESSION_BASE_M2V}
  baseline_v2m     gpu=${GPU_BASE_V2M}  session=${SESSION_BASE_V2M}
  latest_mapb_m2v  gpu=${GPU_MAPB_M2V}  session=${SESSION_MAPB_M2V}
  latest_mapb_v2m  gpu=${GPU_MAPB_V2M}  session=${SESSION_MAPB_V2M}
EOF
