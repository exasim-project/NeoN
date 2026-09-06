#!/usr/bin/env bash
#----------------------------------------------------------------------------------------
# SPDX-FileCopyrightText: 2023 - 2025 NeoN authors
#
# SPDX-License-Identifier: Unlicense
#----------------------------------------------------------------------------------------

set -euo pipefail

# Check required environment variables
GPU_VENDOR=${GPU_VENDOR:?Error: Must set GPU vendor (nvidia|amd|intel)}
PRESET="develop"

echo "Selected GPU type: $GPU_VENDOR"

echo "=== Tool versions ==="
cmake --version
g++ --version || clang++ --version

if [ "$GPU_VENDOR" == "nvidia" ]; then
    echo "=== NVIDIA GPU and compiler driver info ==="
    nvidia-smi --query-gpu=gpu_name,memory.total,driver_version --format=csv
    nvcc --version

    echo "=== Configuring, building, and testing NeoN on NVIDIA ==="
    export CUDA_VISIBLE_DEVICES=0
    cmake --preset develop \
        -DCMAKE_CUDA_ARCHITECTURES=89 \
        -DNeoN_WITH_THREADS=OFF \
        -DNeoN_WITH_MPI=ON \
        -DNeoN_BUILD_BENCHMARKS=ON
    cmake --build --preset develop
    # The CI OpenMPI uses the shared-memory transport (mca_btl_vader) which is
    # not CUDA-aware.  Force staging through host buffers to avoid a SIGSEGV
    # when device pointers are passed to MPI_Isend.
    export NEON_FORCE_HOST_BUFFER=1
    ctest --preset develop -E bench --output-on-failure

elif [ "$GPU_VENDOR" == "amd" ]; then
    # Set up environment
    export CXX_COMPILER_PATH="$(which g++)"
    export CXX_SOURCE="${CXX_COMPILER_PATH%/*/*}"
    export CXX_LIBDIR="${CXX_SOURCE}/lib64"
    export LD_LIBRARY_PATH=${CXX_LIBDIR}:${LD_LIBRARY_PATH}

    echo "=== AMD GPU and compiler driver info ==="
    rocminfo | grep "AMD"
    hipcc --version
    which mpirun

    echo "=== Configuring, building, and testing NeoN on AMD ==="
    cmake --preset develop \
        -DCMAKE_PREFIX_PATH=/opt/rocm \
        -DCMAKE_C_COMPILER=/opt/rocm/llvm/bin/clang \
        -DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++ \
        -DCMAKE_CXX_FLAGS="--gcc-toolchain=${CXX_SOURCE}" \
        -DCMAKE_EXE_LINKER_FLAGS="-L${CXX_LIBDIR}" \
        -DCMAKE_HIP_ARCHITECTURES=gfx90a \
        -DKokkos_ARCH_AMD_GFX90A=ON \
        -DNeoN_WITH_THREADS=OFF \
        -DNeoN_WITH_MPI=ON \
        -DNeoN_BUILD_BENCHMARKS=ON
    cmake --build --preset develop
    # See NVIDIA comment above — same rationale for AMD ROCm MPI.
    export NEON_FORCE_HOST_BUFFER=1
    ctest --preset develop -E bench --output-on-failure

elif [ "$GPU_VENDOR" == "intel" ]; then
    SYCL_PI_TRACE=1
    sycl-ls 2>/dev/null | grep '^\[level_zero:gpu\]'

    # Compiler info (non-fatal)
    icpx --version 2>/dev/null | head -1 || echo "icpx not found"

    # Intel PVC has two tiles and implicit scaling routes work across them;
    # sycl::queue::wait() only drains the root-device queue and misses
    # in-flight work on tile 1, causing GPU page faults on freed USM memory.
    # COMPOSITE hierarchy exposes each tile as a separate L0 device so Kokkos
    # selects a single tile — all work and synchronisation stay on one tile.
    export ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE

    echo "=== Configuring, building, and testing NeoN on Intel ==="
    cmake --preset develop \
        -DCMAKE_CXX_COMPILER=icpx \
        -DCMAKE_CXX_FLAGS="-Wno-deprecated-declarations -Wno-sycl-2020-compat" \
        -DKokkos_ENABLE_SYCL=ON \
        -DKokkos_ARCH_INTEL_PVC=ON \
        -DNeoN_WITH_THREADS=OFF \
        -DNeoN_WITH_MPI=OFF \
        -DNeoN_BUILD_BENCHMARKS=ON \
        -DCMAKE_BUILD_TYPE="release"
    cmake --build --preset develop
    ctest --preset develop -E bench --output-on-failure

else
    echo "Unknown GPU type: $GPU_VENDOR"
    exit 1
fi
