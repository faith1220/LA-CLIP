#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCAL_DATA_ROOT="${LOCAL_DATA_ROOT:-/home/hyz/MyDisk/data}"

mkdir -p "${ROOT_DIR}"

if [[ -L "${ROOT_DIR}/data_local" || -d "${ROOT_DIR}/data_local" ]]; then
  rm -rf "${ROOT_DIR}/data_local"
fi

ln -s "${LOCAL_DATA_ROOT}" "${ROOT_DIR}/data_local"
echo "Linked ${ROOT_DIR}/data_local -> ${LOCAL_DATA_ROOT}"
