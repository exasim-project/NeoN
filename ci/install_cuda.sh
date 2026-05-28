#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2026 NeoN authors
#
# SPDX-License-Identifier: MIT
# SPDX-License-Identifier: Unlicense

set -euo pipefail

version="${1:?Usage: install_cuda.sh <major.minor>}"
package_version="${version/./-}"
cuda_root="/usr/local/cuda-${version}"

if command -v nvcc >/dev/null 2>&1 && nvcc --version | grep -q "release ${version}"; then
    ln -sfn "${cuda_root}" /usr/local/cuda
    exit 0
fi

if [[ -r /etc/os-release ]]; then
    . /etc/os-release
else
    echo "Cannot determine Linux distribution." >&2
    exit 1
fi

case "${VERSION_ID%%.*}" in
    8)
        repo_platform="rhel8"
        ;;
    9)
        repo_platform="rhel9"
        ;;
    *)
        echo "Unsupported CUDA CI base image: ${ID:-unknown} ${VERSION_ID:-unknown}" >&2
        exit 1
        ;;
esac

if command -v dnf >/dev/null 2>&1; then
    package_manager="dnf"
elif command -v yum >/dev/null 2>&1; then
    package_manager="yum"
else
    echo "Expected dnf or yum in the manylinux container." >&2
    exit 1
fi

"${package_manager}" install -y ca-certificates curl
curl -fsSL "https://developer.download.nvidia.com/compute/cuda/repos/${repo_platform}/x86_64/cuda-${repo_platform}.repo" \
    -o /etc/yum.repos.d/cuda.repo
"${package_manager}" clean all
"${package_manager}" install -y "cuda-toolkit-${package_version}"

ln -sfn "${cuda_root}" /usr/local/cuda
"/usr/local/cuda/bin/nvcc" --version
