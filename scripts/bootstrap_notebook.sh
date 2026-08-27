#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ -n "${CT_RESTORE_DATA_ROOT:-}" ]]; then
  DATA_ROOT="${CT_RESTORE_DATA_ROOT}"
elif [[ -n "${RUNPOD_POD_ID:-}" ]]; then
  DATA_ROOT="/workspace/ct_restore_data"
elif [[ -n "${COLAB_RELEASE_TAG:-}" || -d "/content" ]]; then
  DATA_ROOT="/content/ct_restore_data"
else
  DATA_ROOT="${REPO_DIR}/data"
fi

mkdir -p "${DATA_ROOT}" "${DATA_ROOT}/cache/torch" "${DATA_ROOT}/cache/pip"
export CT_RESTORE_DATA_ROOT="${DATA_ROOT}"
export TORCH_HOME="${DATA_ROOT}/cache/torch"
export PIP_CACHE_DIR="${DATA_ROOT}/cache/pip"
export PYTHONUNBUFFERED=1

python -m pip install --upgrade pip setuptools wheel
python -m pip install -e "${REPO_DIR}[train,notebook]"

echo "CT_RESTORE_DATA_ROOT=${CT_RESTORE_DATA_ROOT}"
df -h "${DATA_ROOT}"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
fi
ct-restore doctor

echo "Bootstrap complete. Persist data and checkpoints under: ${DATA_ROOT}"
