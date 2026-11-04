#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2026 NeoN authors
#
# SPDX-License-Identifier: Unlicense

# Build the NeoN conda package for one python version and one GPU flavour.
#
# Usage:
#   ci/build_conda_packages.sh [--python 3.12] [--gpu cpu|cuda|rocm] [--output-dir output]
#                              [--target-platform linux-64] [--cuda-version 12.8]
#                              [--rocm-version 6.3] [--] [extra rattler-build arguments]
#
# The recipe reads the package version from NEON_VERSION rather than from
# pyproject.toml, because reading a file from a recipe needs rattler-build's
# --experimental flag. This script derives it from pyproject.toml so that both
# local builds and CI stay in sync with neon.__version__.

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

python_version="3.12"
gpu_variant="cpu"
output_dir="${repo_root}/output"
target_platform=""
cuda_version="12.8"
rocm_version="6.3"
# One Kokkos architecture per package, matching the published CUDA wheel (NVIDIA Ampere, sm_80).
cuda_architectures="80"
kokkos_arch="AMPERE80"
hip_architectures="gfx90a"
# glibc floor for the linux packages. 2.17 is the widest baseline conda-forge still ships a
# sysroot for; macOS uses the deployment target instead.
glibc_version="2.17"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --python) python_version="$2"; shift 2 ;;
        --gpu) gpu_variant="$2"; shift 2 ;;
        --output-dir) output_dir="$2"; shift 2 ;;
        --target-platform) target_platform="$2"; shift 2 ;;
        --cuda-version) cuda_version="$2"; shift 2 ;;
        --rocm-version) rocm_version="$2"; shift 2 ;;
        --cuda-architectures) cuda_architectures="$2"; shift 2 ;;
        --kokkos-arch) kokkos_arch="$2"; shift 2 ;;
        --hip-architectures) hip_architectures="$2"; shift 2 ;;
        --glibc-version) glibc_version="$2"; shift 2 ;;
        --) shift; break ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

case "${gpu_variant}" in
    cpu | cuda | rocm) ;;
    *) echo "--gpu must be one of cpu, cuda, rocm (got ${gpu_variant})" >&2; exit 1 ;;
esac

if [[ ! "${python_version}" =~ ^([0-9]+)\.([0-9]+) ]]; then
    echo "--python expects a <major>.<minor> version, got '${python_version}'" >&2
    exit 1
fi

# conda-forge only builds nanobind (needed for the nanobind.stubgen POST_BUILD step) for
# python >=3.10, so that is the floor for the conda packages. Compare numerically rather than
# listing versions: anything below the floor has to be rejected here, or it fails much later
# inside rattler-build with an opaque dependency resolution error. Python 3.9 and older are
# covered by the PyPI wheels instead.
if (( BASH_REMATCH[1] < 3 || (BASH_REMATCH[1] == 3 && BASH_REMATCH[2] < 10) )); then
    echo "python ${python_version} is not supported by the conda packages: conda-forge has no" >&2
    echo "nanobind below python 3.10. Use the PyPI wheel for python 3.9 and older." >&2
    exit 1
fi

if [[ -z "${target_platform}" ]]; then
    case "$(uname -s)/$(uname -m)" in
        Linux/x86_64) target_platform="linux-64" ;;
        Linux/aarch64 | Linux/arm64) target_platform="linux-aarch64" ;;
        Darwin/x86_64) target_platform="osx-64" ;;
        Darwin/arm64) target_platform="osx-arm64" ;;
        *) echo "Cannot infer the target platform on $(uname -s)/$(uname -m)" >&2; exit 1 ;;
    esac
fi

NEON_VERSION="${NEON_VERSION:-$(python3 "${repo_root}/scripts/set_package_version.py" --print)}"
export NEON_VERSION
echo "Building neon-pde ${NEON_VERSION} for ${target_platform} (python ${python_version}, ${gpu_variant})" >&2

variant_file="$(mktemp -t neon-variant.XXXXXX)"
trap 'rm -f "${variant_file}"' EXIT

{
    echo "python:"
    echo "  - \"${python_version}\""
    echo "gpu_variant:"
    echo "  - ${gpu_variant}"

    # ${{ stdlib('c') }} in the recipe has no built-in default; the C runtime floor has to be
    # named per target platform.
    case "${target_platform}" in
        osx-*)
            # Kokkos uses C++17/20 aligned new/delete, which needs macOS 10.13+. The wheels pin
            # 11.0 for both architectures; keep the conda packages on the same floor.
            echo "c_stdlib:"
            echo "  - macosx_deployment_target"
            echo "c_stdlib_version:"
            echo "  - \"11.0\""
            echo "MACOSX_DEPLOYMENT_TARGET:"
            echo "  - \"11.0\""
            ;;
        *)
            echo "c_stdlib:"
            echo "  - sysroot"
            echo "c_stdlib_version:"
            echo "  - \"${glibc_version}\""
            ;;
    esac

    if [[ "${gpu_variant}" == "cuda" ]]; then
        echo "cuda_version:"
        echo "  - \"${cuda_version}\""
        echo "cuda_compiler:"
        echo "  - cuda-nvcc"
        echo "cuda_compiler_version:"
        echo "  - \"${cuda_version}\""
        echo "cuda_architectures:"
        echo "  - \"${cuda_architectures}\""
        echo "kokkos_arch:"
        echo "  - ${kokkos_arch}"
    fi

    if [[ "${gpu_variant}" == "rocm" ]]; then
        echo "rocm_version:"
        echo "  - \"${rocm_version}\""
        echo "hip_architectures:"
        echo "  - ${hip_architectures}"
    fi
} > "${variant_file}"

{
    echo "--- variant configuration ---"
    cat "${variant_file}"
    echo "-----------------------------"
} >&2

rattler_build="${RATTLER_BUILD:-rattler-build}"

"${rattler_build}" build \
    --recipe "${repo_root}/recipe/recipe.yaml" \
    --variant-config "${variant_file}" \
    --output-dir "${output_dir}" \
    --target-platform "${target_platform}" \
    --channel conda-forge \
    "$@"
