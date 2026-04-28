#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
EXP_ROOT="${ROOT_DIR}/experiments/cap_baseline_crossdomain"
COMMAND_DIR="${EXP_ROOT}/commands"
LOG_ROOT="${EXP_ROOT}/logs"
CKPT_ROOT="${EXP_ROOT}/ckpts"
REPORT_ROOT="${EXP_ROOT}/reports"
FAILED_FILE="${EXP_ROOT}/failed_runs.txt"
STATUS_TSV="${REPORT_ROOT}/run_status.tsv"

mkdir -p "${COMMAND_DIR}" "${LOG_ROOT}" "${CKPT_ROOT}" "${REPORT_ROOT}"
: > "${FAILED_FILE}"
printf 'experiment\tstart_time\tend_time\texit_code\ttee_log\tckpt_dir\n' > "${STATUS_TSV}"

DATA_DIR="${DATA_DIR:-${ROOT_DIR}/data_local}"
CLIP_DOWNLOAD_DIR="${CLIP_DOWNLOAD_DIR:-$HOME/.cache/clip}"
PYTHON_BIN="${PYTHON_BIN:-}"

if [[ -z "${PYTHON_BIN}" ]]; then
  if [[ -x "${HOME}/.conda/envs/multiads/bin/python" ]]; then
    PYTHON_BIN="${HOME}/.conda/envs/multiads/bin/python"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  else
    echo "No usable python interpreter found." >&2
    exit 127
  fi
fi

select_gpu() {
  local selected
  selected="$(
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null \
      | awk -F, 'NF >= 2 {gsub(/ /, "", $1); gsub(/ /, "", $2); print $1 "," $2}' \
      | sort -t, -k2,2n -k1,1n \
      | head -n 1 \
      | cut -d, -f1
  )"
  if [[ -n "${selected}" ]]; then
    printf '%s\n' "${selected}"
  else
    printf '0\n'
  fi
}

GPU_ID="${GPU_ID:-}"
if [[ -z "${GPU_ID}" ]]; then
  GPU_ID="$(select_gpu)"
fi

EXPERIMENTS=(
  "mvtec_to_visa_A0_baseline"
  "mvtec_to_visa_A1_cap"
  "visa_to_mvtec_A0_baseline"
  "visa_to_mvtec_A1_cap"
)

run_one() {
  local experiment_name="$1"
  local cmd_script="${COMMAND_DIR}/${experiment_name}.sh"
  local tee_log="${LOG_ROOT}/${experiment_name}.log"
  local ckpt_dir="${CKPT_ROOT}/${experiment_name}"
  local start_time
  local end_time
  local exit_code
  local cmd_preview

  start_time="$(date '+%Y-%m-%d %H:%M:%S')"
  cmd_preview="$(DATA_DIR="${DATA_DIR}" CLIP_DOWNLOAD_DIR="${CLIP_DOWNLOAD_DIR}" PYTHON_BIN="${PYTHON_BIN}" bash "${cmd_script}" --print-cmd)"

  {
    echo "============================================================"
    echo "Experiment: ${experiment_name}"
    echo "Start time: ${start_time}"
    echo "GPU_ID: ${GPU_ID}"
    echo "Command: ${cmd_preview}"
    echo "============================================================"
  } | tee "${tee_log}"

  set +e
  CUDA_VISIBLE_DEVICES="${GPU_ID}" \
  DATA_DIR="${DATA_DIR}" \
  CLIP_DOWNLOAD_DIR="${CLIP_DOWNLOAD_DIR}" \
  PYTHON_BIN="${PYTHON_BIN}" \
  bash "${cmd_script}" 2>&1 | tee -a "${tee_log}"
  exit_code="${PIPESTATUS[0]}"
  set -e

  end_time="$(date '+%Y-%m-%d %H:%M:%S')"
  {
    echo "End time: ${end_time}"
    echo "Exit code: ${exit_code}"
  } | tee -a "${tee_log}"

  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${experiment_name}" \
    "${start_time}" \
    "${end_time}" \
    "${exit_code}" \
    "${tee_log}" \
    "${ckpt_dir}" >> "${STATUS_TSV}"

  if [[ "${exit_code}" -ne 0 ]]; then
    printf '%s\t%s\n' "${experiment_name}" "${tee_log}" >> "${FAILED_FILE}"
  fi
}

echo "Using GPU_ID=${GPU_ID}"
echo "DATA_DIR=${DATA_DIR}"
echo "CLIP_DOWNLOAD_DIR=${CLIP_DOWNLOAD_DIR}"
echo "PYTHON_BIN=${PYTHON_BIN}"

for experiment_name in "${EXPERIMENTS[@]}"; do
  run_one "${experiment_name}"
done

set +e
"${PYTHON_BIN}" "${EXP_ROOT}/parse_results.py" --exp-root "${EXP_ROOT}" 2>&1 | tee -a "${LOG_ROOT}/parse_results.log"
PARSE_EXIT="${PIPESTATUS[0]}"
set -e

echo "Result parser exit code: ${PARSE_EXIT}" | tee -a "${LOG_ROOT}/parse_results.log"
exit 0
