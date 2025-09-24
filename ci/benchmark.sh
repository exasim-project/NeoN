#!/usr/bin/env bash
#----------------------------------------------------------------------------------------
# SPDX-FileCopyrightText: 2023 - 2025 NeoN authors
#
# SPDX-License-Identifier: Unlicense
#----------------------------------------------------------------------------------------

set -euo pipefail

RESULTS_DIR=${RESULTS_DIR:-results}
TARGET_REPO=${TARGET_REPO:?Must set TARGET_REPO}
REPO_NAME=$(basename "$TARGET_REPO" .git)
TARGET_BRANCH=${TARGET_BRANCH:?Must set TARGET_BRANCH}
RUN_IDENTIFIER=${RUN_IDENTIFIER:?Must set RUN_IDENTIFIER}
API_TOKEN_GITHUB=${API_TOKEN_GITHUB:?Must set API_TOKEN_GITHUB}

# Check GPU type argument
if [ $# -lt 1 ]; then
    echo "Usage: $0 <gpu_type>"
    echo "  gpu_type: nvidia | amd | none"
    exit 1
fi

GPU_VENDOR="$1"
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
        nvcc --version 2>/dev/null || hipcc --version 2>/dev/null || echo "No GPU compiler available"
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
        cmake --preset profiling -DCMAKE_CUDA_ARCHITECTURES=90 -DNeoN_WITH_THREADS=OFF
    elif [[ "$GPU_VENDOR" == "amd" ]]; then
        # Set up environment
        export PATH=/opt/rocm/bin:$PATH
        export HIPCC_CXX=/usr/bin/g++

        cmake --preset profiling \
            -DCMAKE_CXX_COMPILER=hipcc \
            -DCMAKE_HIP_ARCHITECTURES=gfx90a \
            -DKokkos_ARCH_AMD_GFX90A=ON \
            -DNeoN_WITH_THREADS=OFF
    else
        cmake --preset profiling -DNeoN_WITH_THREADS=OFF
    fi

    echo ">>> Building"
    cmake --build --preset profiling

    echo ">>> Running benchmarks..."
    ctest --preset profiling

    pushd build/profiling/bin/benchmarks >/dev/null
    python3 ../../../../scripts/catch2json.py
    popd >/dev/null

    mkdir -p "${output_dir}"
    cp build/profiling/bin/benchmarks/*.json "${output_dir}/"

    rm -rf build
}

# Push benchmark results to GitHub
push_results() {
    git config --global user.email "gitlab-ci@users.noreply.github.com"
    git config --global user.name "GitLab CI"

    git clone "https://oauth2:${API_TOKEN_GITHUB}@${TARGET_REPO}"
    cd "${REPO_NAME}"

    git checkout "${TARGET_BRANCH}" || git checkout -b "${TARGET_BRANCH}"
    mkdir -p "${RUN_IDENTIFIER}/${CI_RUNNER_DESCRIPTION}"
    cp -r ../${RESULTS_DIR}/* "${RUN_IDENTIFIER}/${CI_RUNNER_DESCRIPTION}"

    git add .
    git commit -m "Benchmarks from GitLab pipeline ${RUN_IDENTIFIER}" || echo "No changes to commit"
    git pull --rebase || true
    git push origin "${TARGET_BRANCH}"
}

### Main execution ###
collect_system_info "${GPU_VENDOR}"

# Current branch
build_and_benchmark "$(git rev-parse --abbrev-ref HEAD)" "${RESULTS_DIR}"

# Main branch
build_and_benchmark "main" "${RESULTS_DIR}/main"

# Push results
push_results
