#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if ! command -v python >/dev/null 2>&1; then
  echo "python is not on PATH" >&2
  exit 1
fi

PYTHON_INFO="$(python - <<'PY'
import os
import site
import sys

site_packages = next((path for path in site.getsitepackages()
                      if path.endswith('site-packages')), '')
print(sys.prefix)
print(site_packages)
PY
)"

CONDA_PREFIX_DIR="$(printf '%s\n' "${PYTHON_INFO}" | sed -n '1p')"
SITE_PACKAGES_DIR="$(printf '%s\n' "${PYTHON_INFO}" | sed -n '2p')"

LIB_DIRS=("${CONDA_PREFIX_DIR}/lib")
if [[ -n "${SITE_PACKAGES_DIR}" && -d "${SITE_PACKAGES_DIR}/nvidia" ]]; then
  while IFS= read -r libdir; do
    LIB_DIRS+=("${libdir}")
  done < <(find "${SITE_PACKAGES_DIR}/nvidia" -maxdepth 3 -path '*/lib' -type d | sort)
fi

CUDA_DATA_DIR="${SITE_PACKAGES_DIR}/nvidia/cuda_nvcc"
LD_PATH="$(IFS=:; printf '%s' "${LIB_DIRS[*]}")"

export LD_LIBRARY_PATH="${LD_PATH}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

if [[ -d "${CUDA_DATA_DIR}" ]]; then
  export XLA_FLAGS="--xla_gpu_cuda_data_dir=${CUDA_DATA_DIR}${XLA_FLAGS:+ ${XLA_FLAGS}}"
fi

if [[ -n "${JAX_PLATFORMS:-}" ]]; then
  unset JAX_PLATFORMS
fi

exec python "${ROOT_DIR}/train_online.py" \
  --jax_platform=cuda \
  --require_gpu=True \
  "$@"
