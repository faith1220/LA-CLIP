#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
EXP_ROOT="${ROOT_DIR}/experiments/cap_baseline_crossdomain"
EXP_NAME="visa_to_mvtec_A0_baseline"

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

mkdir -p "${EXP_ROOT}/ckpts/${EXP_NAME}"

CMD=(
  "${PYTHON_BIN}" main.py
  --dataset visa
  --test_dataset mvtec
  --data_dir "${DATA_DIR}"
  --clip_download_dir "${CLIP_DOWNLOAD_DIR}"
  --log_dir "${EXP_ROOT}/ckpts/${EXP_NAME}"
  --model "ViT-L/14@336px"
  --epochs 2
  --batch_size 8
  --img_size 518
  --feature_layers 6 12 18 24
  --memory_layers 6 12 18 24
  --fewshot 0
  --seed 122
  --fg_prompt off
  --use_lsar 0
  --use_mapb 0
  --use_mvti 0
  --score_mode clip
)

if [[ "${1:-}" == "--print-cmd" ]]; then
  printf '%q ' "${CMD[@]}"
  printf '\n'
  exit 0
fi

cd "${ROOT_DIR}"
printf '[%s] Command: ' "${EXP_NAME}"
printf '%q ' "${CMD[@]}"
printf '\n'
exec "${CMD[@]}"
