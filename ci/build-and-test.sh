#!/usr/bin/env bash
#----------------------------------------------------------------------------------------
# SPDX-FileCopyrightText: 2023 - 2025 NeoN authors
#
# SPDX-License-Identifier: Unlicense
#----------------------------------------------------------------------------------------

set -euo pipefail

# Check arguments
if [ $# -lt 1 ]; then
    echo "Usage: $0 <gpu_type>"
    echo "  gpu_type: nvidia | amd"
    exit 1
fi

GPU_TYPE="$1"
echo "Selected GPU type: $GPU_TYPE"

echo "=== Tool versions ==="
cmake --version
g++ --version || clang++ --version

if [ "$GPU_TYPE" == "nvidia" ]; then
    echo "=== NVIDIA GPU and compiler driver info ==="
    nvidia-smi --query-gpu=gpu_name,memory.total,driver_version --format=csv
    nvcc --version

    echo "=== Configuring, building, and testing NeoN on NVIDIA ==="
    cmake --preset develop -DCMAKE_CUDA_ARCHITECTURES=90 -DNeoN_WITH_THREADS=OFF
    cmake --build --preset develop
    ctest --preset develop --output-on-failure

elif [ "$GPU_TYPE" == "amd" ]; then
    # Set up environment
    export PATH=/opt/rocm/bin:$PATH
    export HIPCC_CXX=/usr/bin/g++

    echo "=== AMD GPU and compiler driver info ==="
    rocminfo | grep "AMD"
    hipcc --version

    echo "=== Configuring, building, and testing NeoN on AMD ==="
    cmake --preset develop \
        -DCMAKE_CXX_COMPILER=hipcc \
        -DCMAKE_HIP_ARCHITECTURES=gfx90a \
        -DKokkos_ARCH_AMD_GFX90A=ON \
        -DNeoN_WITH_THREADS=OFF
    cmake --build --preset develop
    ctest --preset develop --output-on-failure

else
    echo "Unknown GPU type: $GPU_TYPE"
    exit 1
fi
