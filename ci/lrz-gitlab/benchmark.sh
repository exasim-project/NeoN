#!/usr/bin/env bash
#----------------------------------------------------------------------------------------
# SPDX-FileCopyrightText: 2023 - 2025 NeoN authors
#
# SPDX-License-Identifier: Unlicense
#----------------------------------------------------------------------------------------

set -euo pipefail

PRESET="profiling"

# Check required environment variables
GPU_VENDOR=${GPU_VENDOR:?Error: Must set GPU vendor (nvidia|amd|intel)}
PR_NUMBER=${PR_NUMBER:?Error: Must set PR number}
RESULTS_DIR=${RESULTS_DIR:-results}
TARGET_REPO=${TARGET_REPO:?Must set TARGET_REPO}
REPO_NAME=$(basename "$TARGET_REPO" .git)
RUN_IDENTIFIER=${RUN_IDENTIFIER:?Must set RUN_IDENTIFIER}
CI_RUNNER_DESCRIPTION=${CI_RUNNER_DESCRIPTION:?Must set CI_RUNNER_DESCRIPTION}
API_TOKEN_GITHUB=${API_TOKEN_GITHUB:?Must set API_TOKEN_GITHUB}

echo "Selected GPU vendor: ${GPU_VENDOR}"

# Collect system info
collect_system_info() {
    mkdir -p "$RESULTS_DIR"
    {
        echo "===== CPU INFO ====="
        lscpu || echo "lscpu not available"
        echo ""

        echo "===== GPU INFO ====="
        if [[ "$1" == "nvidia" ]]; then
            nvidia-smi
        elif [[ "$1" == "amd" ]]; then
            rocm-smi --showproductname --showvbios
        elif [[ "$1" == "intel" ]]; then
            SYCL_PI_TRACE=1
            sycl-ls 2>/dev/null | grep '^\[level_zero:gpu\]'
        else
            echo "No GPU selected"
        fi
        echo ""

        echo "===== COMPILER INFO ====="
        echo "CMake:"
        cmake --version || echo "cmake not available"
        echo ""
        echo "C++ compiler:"
        g++ --version || clang++ --version || echo "No C++ compiler found"
        echo ""
        echo "CUDA/ROCm compiler:"
        nvcc --version 2>/dev/null || hipcc --version 2>/dev/null || icpx --version 2>/dev/null  || echo "No GPU compiler available"
    } > "${RESULTS_DIR}/system-info.log"
}

# Build & benchmark given branch
build_and_benchmark() {
    local branch=$1
    local output_dir=$2

    echo ">>> Checking out ${branch}"
    git fetch origin "${branch}"
    git checkout "${branch}"

    echo ">>> Configuring build"
    if [[ "$GPU_VENDOR" == "nvidia" ]]; then
        cmake --preset $PRESET -DCMAKE_CUDA_ARCHITECTURES=90 -DNeoN_WITH_THREADS=OFF
    elif [[ "$GPU_VENDOR" == "amd" ]]; then
        # Set up environment
        export CXX_COMPILER_PATH="$(which g++)"
        export CXX_SOURCE="${CXX_COMPILER_PATH%/*/*}"
        export CXX_LIBDIR="${CXX_SOURCE}/lib64"
        export LD_LIBRARY_PATH=${CXX_LIBDIR}:${LD_LIBRARY_PATH}
        cmake --preset $PRESET \
            -DCMAKE_PREFIX_PATH=/opt/rocm \
            -DCMAKE_C_COMPILER=/opt/rocm/llvm/bin/clang \
            -DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++ \
            -DCMAKE_CXX_FLAGS="--gcc-toolchain=${CXX_SOURCE}" \
            -DCMAKE_EXE_LINKER_FLAGS="-L${CXX_LIBDIR}" \
            -DCMAKE_HIP_ARCHITECTURES=gfx90a \
            -DKokkos_ARCH_AMD_GFX90A=ON \
            -DNeoN_WITH_THREADS=OFF
    elif [[ "$GPU_VENDOR" == "intel" ]]; then
        cmake --preset $PRESET \
            -DCMAKE_CXX_COMPILER=icpx \
            -DCMAKE_CXX_FLAGS="-Wno-deprecated-declarations -Wno-sycl-2020-compat" \
            -DKokkos_ENABLE_SYCL=ON \
            -DKokkos_ARCH_INTEL_PVC=ON \
            -DNeoN_WITH_THREADS=OFF \
            -DCMAKE_BUILD_TYPE="release"
    else
        cmake --preset $PRESET -DNeoN_WITH_THREADS=OFF
    fi

    echo ">>> Building"
    cmake --build --preset $PRESET

    echo ">>> Running benchmarks..."
    ctest --preset $PRESET

    pushd build/$PRESET/bin/benchmarks >/dev/null
    python3 ../../../../scripts/catch2json.py
    popd >/dev/null

    mkdir -p "${output_dir}"
    cp build/$PRESET/bin/benchmarks/*.json "${output_dir}/"

    rm -rf build
}

# Push benchmark results to GitHub
push_results() {
    git config --global user.email "gitlab-ci@users.noreply.github.com"
    git config --global user.name "GitLab CI"

    git clone "https://oauth2:${API_TOKEN_GITHUB}@${TARGET_REPO}"
    cd "${REPO_NAME}"

    git checkout "NeoN_PR_${PR_NUMBER}" || git checkout -b "NeoN_PR_${PR_NUMBER}"
    mkdir -p "NeoN/${PR_NUMBER}/${CI_RUNNER_DESCRIPTION}"
    cp -r ../${RESULTS_DIR}/* "NeoN/${PR_NUMBER}/${CI_RUNNER_DESCRIPTION}"

    git add .
    git commit -m "Benchmarks from GitLab pipeline ${RUN_IDENTIFIER}" || echo "No changes to commit"
    git pull --rebase || true
    git push origin "NeoN_PR_${PR_NUMBER}"
}

### Main execution ###
collect_system_info "${GPU_VENDOR}"

# Current branch
build_and_benchmark "$(git rev-parse --abbrev-ref HEAD)" "${RESULTS_DIR}"

# Develop branch
build_and_benchmark "develop" "${RESULTS_DIR}/develop"

# Push results
push_results
