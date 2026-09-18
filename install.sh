#!/usr/bin/env bash
#
# PyTorch on JetPack 7.2 for Jetson Orin NX 16GB (sm_87)
# CUDA 13.2, Python 3.12, existing Conda env: aditya
#
# Tested assumptions:
# - JetPack / L4T: R39.2.x
# - CUDA: 13.2
# - Python: 3.12
# - Architecture: aarch64
# - Conda / Miniforge environment: aditya
#
# IMPORTANT:
# Run this script from a shell where Conda is initialized.
#

set -euo pipefail

CONDA_ENV="aditya"
CU_INDEX="https://download.pytorch.org/whl/cu132"

echo "============================================================"
echo " Jetson JP7.2 + CUDA 13.2 + PyTorch setup"
echo " Conda environment: ${CONDA_ENV}"
echo "============================================================"
echo


# ------------------------------------------------------------
# Step 0: Locate CUDA 13.2
# ------------------------------------------------------------

echo "==> Step 0: locate CUDA 13.2"

CUDA_DIR="$(ls -d /usr/local/cuda-13.2 2>/dev/null || true)"

if [ -z "${CUDA_DIR}" ]; then
    echo "ERROR: Could not find /usr/local/cuda-13.2"
    echo
    echo "Available CUDA installs:"
    ls -d /usr/local/cuda* 2>/dev/null || echo "  (none found)"
    exit 1
fi

export CUDA_HOME="${CUDA_DIR}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

echo "CUDA_HOME: ${CUDA_HOME}"
echo

echo "==> nvcc check"
which nvcc
nvcc --version
echo


# ------------------------------------------------------------
# Step 1: Persist CUDA paths
# ------------------------------------------------------------

echo "==> Step 1: persist CUDA 13.2 paths"

if ! grep -q "CUDA 13.2 for JetPack 7.2" "${HOME}/.bashrc" 2>/dev/null; then
    cat <<EOF >> "${HOME}/.bashrc"

# CUDA 13.2 for JetPack 7.2
export CUDA_HOME=/usr/local/cuda-13.2
export PATH=\$CUDA_HOME/bin:\$PATH
export LD_LIBRARY_PATH=\$CUDA_HOME/lib64:\${LD_LIBRARY_PATH:-}
EOF

    echo "Added CUDA exports to ~/.bashrc"
else
    echo "CUDA configuration already present in ~/.bashrc"
fi

echo


# ------------------------------------------------------------
# Step 2: System prerequisites
# ------------------------------------------------------------

echo "==> Step 2: install system prerequisites"

sudo apt-get update

sudo apt-get install -y \
    build-essential \
    cmake \
    git \
    pkg-config \
    wget \
    libopenblas-dev \
    libjpeg-dev \
    zlib1g-dev

echo


# ------------------------------------------------------------
# Step 3: NVPL + cuDSS
# ------------------------------------------------------------

echo "==> Step 3: check NVPL"

if ! ldconfig -p | grep -q libnvpl_lapack; then

    echo "NVPL not found. Installing..."

    if [ ! -f /usr/share/keyrings/cuda-archive-keyring.gpg ]; then

        TMP_DEB="$(mktemp --suffix=.deb)"

        wget -qO "${TMP_DEB}" \
            https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/sbsa/cuda-keyring_1.1-1_all.deb

        sudo dpkg -i "${TMP_DEB}"
        rm -f "${TMP_DEB}"

        sudo apt-get update
    fi

    sudo apt-get install -y nvpl

else
    echo "NVPL already installed"
fi

echo


echo "==> Check cuDSS"

if ! ldconfig -p | grep -q libcudss; then

    echo "cuDSS not found. Installing..."

    sudo apt-get install -y libcudss0-cuda-13

    echo "/usr/lib/aarch64-linux-gnu/libcudss/13" \
        | sudo tee /etc/ld.so.conf.d/cudss.conf >/dev/null

    sudo ldconfig

else
    echo "cuDSS already installed"
fi

echo


# ------------------------------------------------------------
# Step 4: Activate existing Conda environment
# ------------------------------------------------------------

echo "==> Step 4: activate Conda environment '${CONDA_ENV}'"

# Locate conda
if [ -f "${HOME}/miniforge3/etc/profile.d/conda.sh" ]; then

    # shellcheck disable=SC1091
    source "${HOME}/miniforge3/etc/profile.d/conda.sh"

elif [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then

    # shellcheck disable=SC1091
    source "${HOME}/miniconda3/etc/profile.d/conda.sh"

elif command -v conda >/dev/null 2>&1; then

    CONDA_BASE="$(conda info --base)"

    # shellcheck disable=SC1091
    source "${CONDA_BASE}/etc/profile.d/conda.sh"

else

    echo "ERROR: Conda not found."
    echo "Expected Miniforge under ~/miniforge3."
    exit 1
fi


conda activate "${CONDA_ENV}"


echo
echo "Conda environment:"
echo "  ${CONDA_PREFIX}"

echo
echo "Python:"
which python
python --version

echo


# ------------------------------------------------------------
# Step 5: Verify Python and architecture
# ------------------------------------------------------------

echo "==> Step 5: verify Python 3.12 + aarch64"

python - <<'PY'
import sys
import platform

print("Python:", sys.version)
print("Executable:", sys.executable)
print("Architecture:", platform.machine())

if sys.version_info[:2] != (3, 12):
    raise RuntimeError(
        f"Python 3.12 required, found {sys.version}"
    )

if platform.machine() != "aarch64":
    raise RuntimeError(
        f"aarch64 required, found {platform.machine()}"
    )

print("Python environment OK")
PY

echo


# ------------------------------------------------------------
# Step 6: Upgrade pip tooling inside Conda env
# ------------------------------------------------------------

echo "==> Step 6: upgrade pip/setuptools/wheel"

python -m pip install --upgrade \
    pip \
    setuptools \
    wheel \
    packaging

echo


# ------------------------------------------------------------
# Step 7: Remove conflicting PyTorch packages
# ------------------------------------------------------------

echo "==> Step 7: remove previous PyTorch packages"

python -m pip uninstall -y \
    torch \
    torchvision \
    torchaudio \
    torchdata \
    torchtext \
    torch-tensorrt \
    2>/dev/null || true

echo

echo "Purging pip cache..."
python -m pip cache purge || true

echo


# ------------------------------------------------------------
# Step 8: Install PyTorch CUDA 13.2 prerelease wheel
# ------------------------------------------------------------

echo "==> Step 8: install PyTorch cu132"

python -m pip install \
    --pre \
    torch \
    --extra-index-url "${CU_INDEX}"

echo


# ------------------------------------------------------------
# Step 9: Verify PyTorch CUDA
# ------------------------------------------------------------

echo "==> Step 9: verify PyTorch GPU support"

python - <<'PY'
import torch

print()
print("PyTorch version :", torch.__version__)
print("Torch CUDA      :", torch.version.cuda)
print("CUDA available  :", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA is not available in PyTorch"
    )

print("GPU             :", torch.cuda.get_device_name(0))
print("Capability      :", torch.cuda.get_device_capability(0))

print()
print("Running CUDA matrix multiplication test...")

x = torch.randn(
    2048,
    2048,
    device="cuda"
)

y = x @ x

torch.cuda.synchronize()

print("GPU computation result:", float(y.sum()))
print("CUDA KERNEL TEST: PASS")

PY

echo


# ------------------------------------------------------------
# Step 10: Install torchvision
# ------------------------------------------------------------

echo "==> Step 10: install torchvision cu132"

python -m pip install \
    --pre \
    torchvision \
    --extra-index-url "${CU_INDEX}"

echo


# ------------------------------------------------------------
# Step 11: Common Python packages
# ------------------------------------------------------------

echo "==> Step 11: install common Python packages"

python -m pip install \
    numpy \
    pillow \
    matplotlib \
    jupyterlab \
    ipykernel

echo


# ------------------------------------------------------------
# Step 12: Jupyter kernel
# ------------------------------------------------------------

echo "==> Step 12: register Jupyter kernel"

python -m ipykernel install \
    --user \
    --name "${CONDA_ENV}" \
    --display-name "Jetson JP7.2 - ${CONDA_ENV}"

echo


# ------------------------------------------------------------
# Step 13: Check TensorRT visibility
# ------------------------------------------------------------

echo "==> Step 13: check TensorRT inside Conda"

python - <<'PY'
try:
    import tensorrt

    print("TensorRT version :", tensorrt.__version__)
    print("TensorRT module  :", tensorrt.__file__)
    print("TensorRT import  : PASS")

except Exception as exc:

    print()
    print("TensorRT is NOT visible inside this Conda environment.")
    print()
    print("This does not affect PyTorch CUDA.")
    print()
    print("Error:")
    print(exc)
    print()
    print(
        "JetPack may have TensorRT installed only for the "
        "system Python."
    )
PY

echo


# ------------------------------------------------------------
# Step 14: Final stack summary
# ------------------------------------------------------------

echo "==> Step 14: final stack summary"

python - <<'PY'
import sys
import platform
import torch

print()
print("==========================================")
print(" Jetson Python / PyTorch Stack")
print("==========================================")

print("Python       :", sys.version.split()[0])
print("Python path  :", sys.executable)
print("Architecture :", platform.machine())

print("PyTorch      :", torch.__version__)
print("Torch CUDA   :", torch.version.cuda)
print("CUDA usable  :", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU           :", torch.cuda.get_device_name(0))
    print("Capability    :", torch.cuda.get_device_capability(0))

try:
    import torchvision
    print("Torchvision   :", torchvision.__version__)
except Exception as e:
    print("Torchvision   : ERROR:", e)

try:
    import tensorrt
    print("TensorRT      :", tensorrt.__version__)
except Exception:
    print("TensorRT      : not visible in Conda environment")

print("==========================================")
PY


echo
echo "============================================================"
echo " DONE"
echo "============================================================"
echo
echo "Conda environment:"
echo
echo "    conda activate ${CONDA_ENV}"
echo
echo "Then test with:"
echo
echo "    python -c 'import torch; print(torch.cuda.is_available())'"
echo
echo "or:"
echo
echo "    python -c 'import torch; print(torch.cuda.get_device_name(0))'"
echo
echo "NOTE:"
echo "  torchaudio is intentionally not installed."
echo "  TensorRT may require extra configuration because Conda does"
echo "  not automatically inherit JetPack system Python packages."
echo