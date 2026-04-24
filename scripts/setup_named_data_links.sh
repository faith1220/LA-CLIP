#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_LINK_DIR="${DATA_LINK_DIR:-${ROOT_DIR}/data_local}"
DATA_SOURCE_ROOT="${DATA_SOURCE_ROOT:-/mnt/sdc/user/data}"

mkdir -p "${DATA_LINK_DIR}"

link_one() {
  local link_name="$1"
  shift
  local link_path="${DATA_LINK_DIR}/${link_name}"
  local target=""
  local candidate

  for candidate in "$@"; do
    if [[ -e "${candidate}" ]]; then
      target="${candidate}"
      break
    fi
  done

  if [[ -z "${target}" ]]; then
    echo "Skip ${link_name}: source not found" >&2
    return 0
  fi

  if [[ -L "${link_path}" ]]; then
    local current_target
    current_target="$(readlink -f "${link_path}")"
    if [[ "${current_target}" == "${target}" ]]; then
      echo "Keep ${link_path} -> ${target}"
      return 0
    fi
    rm -f "${link_path}"
  elif [[ -e "${link_path}" ]]; then
    echo "Skip ${link_path}: existing non-symlink path" >&2
    return 0
  fi

  ln -s "${target}" "${link_path}"
  echo "Linked ${link_path} -> ${target}"
}

link_one "mvtec" "${DATA_SOURCE_ROOT}/mvtec_anomaly_detection" "${DATA_SOURCE_ROOT}/mvtec"
link_one "visa" "${DATA_SOURCE_ROOT}/VisA_pytorch" "${DATA_SOURCE_ROOT}/visa"
link_one "DAGM_KaggleUpload" "${DATA_SOURCE_ROOT}/DAGM_KaggleUpload" "${DATA_SOURCE_ROOT}/DAGM_anomaly_detection"
link_one "DTD-Synthetic" "${DATA_SOURCE_ROOT}/DTD-Synthetic"
link_one "ISIC2016" "${DATA_SOURCE_ROOT}/ISIC2016"
link_one "CVC-ClinicDB" "${DATA_SOURCE_ROOT}/CVC-ClinicDB"
link_one "CVC-ColonDB" "${DATA_SOURCE_ROOT}/CVC-ColonDB"
link_one "BrainMRI" "${DATA_SOURCE_ROOT}/BrainMRI"
link_one "Br35H" "${DATA_SOURCE_ROOT}/Br35H"
link_one "Kvasir" "${DATA_SOURCE_ROOT}/Kvasir"
link_one "btad" "${DATA_SOURCE_ROOT}/btad" "${DATA_SOURCE_ROOT}/BTAD"
