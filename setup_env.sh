#!/usr/bin/env bash
# Setup script for IsaacSimCustom: Isaac Sim 5.1 + Isaac Lab + Newton
# Usage: bash setup_env.sh
set -e

ENV_NAME="env_isaacsim"
PYTHON_VERSION="3.11"
REPO_URL="https://github.com/mathismusic/IsaacSimCustom.git"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Creating conda environment '${ENV_NAME}' with Python ${PYTHON_VERSION} ==="
conda create -n "${ENV_NAME}" python="${PYTHON_VERSION}" -y

echo "=== Installing cmake (3.31.x) via conda-forge ==="
# cmake 4.x breaks egl_probe build, so pin to 3.x
conda install -n "${ENV_NAME}" -c conda-forge "cmake<4" -y

echo "=== Installing uv ==="
conda run -n "${ENV_NAME}" pip install uv

echo "=== Installing Isaac Sim 5.1 ==="
conda run -n "${ENV_NAME}" uv pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com

echo "=== Detecting CUDA version ==="
CUDA_VERSION=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+')
CUDA_MAJOR=$(echo "${CUDA_VERSION}" | cut -d. -f1)
CUDA_MINOR=$(echo "${CUDA_VERSION}" | cut -d. -f2)
CUDA_TAG="cu${CUDA_MAJOR}${CUDA_MINOR}"
echo "Detected CUDA ${CUDA_VERSION} -> using PyTorch index: ${CUDA_TAG}"

echo "=== Installing PyTorch 2.7 (${CUDA_TAG}) ==="
conda run -n "${ENV_NAME}" uv pip install -U torch==2.7.0 torchvision==0.22.0 --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"

echo "=== Cloning IsaacSimCustom (if not already present) ==="
if [ ! -d "${SCRIPT_DIR}/IsaacLab" ]; then
    cd "$(dirname "${SCRIPT_DIR}")"
    git clone --recurse-submodules "${REPO_URL}"
    cd IsaacSimCustom
else
    echo "IsaacSimCustom already present at ${SCRIPT_DIR}, skipping clone."
fi

echo "=== Installing Isaac Lab (editable) ==="
cd "${SCRIPT_DIR}/IsaacLab"
conda run -n "${ENV_NAME}" ./isaaclab.sh --install

echo "=== Installing Newton (editable) ==="
conda run -n "${ENV_NAME}" pip install -e "${SCRIPT_DIR}/newton"

echo ""
echo "=== Done! ==="
echo "Activate with: conda activate ${ENV_NAME}"
echo ""
echo "Editable packages installed:"
conda run -n "${ENV_NAME}" pip list --editable
