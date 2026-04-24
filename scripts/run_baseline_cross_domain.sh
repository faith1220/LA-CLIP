#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

DATASET="${1:-mvtec}"
TEST_DATASET="${2:-visa}"
DATA_DIR="${DATA_DIR:-${ROOT_DIR}/data_local}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/runs/baseline_${DATASET}_to_${TEST_DATASET}}"
WEIGHT_ARG=()

if [[ -n "${WEIGHT_DIR:-}" ]]; then
  WEIGHT_ARG=(--weight "${WEIGHT_DIR}")
fi

python main.py \
  --log_dir "${LOG_DIR}" \
  --dataset "${DATASET}" \
  --test_dataset "${TEST_DATASET}" \
  --data_dir "${DATA_DIR}" \
  "${WEIGHT_ARG[@]}"
