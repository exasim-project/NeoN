#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2026 NeoN authors
#
# SPDX-License-Identifier: Unlicense

set -euxo pipefail

gpu_variant="${NEON_GPU_VARIANT:?NEON_GPU_VARIANT must be set by the recipe}"

build_dir="${SRC_DIR}/build-conda"

cmake_args=(
    -G Ninja
    -D CMAKE_BUILD_TYPE=Release
    -D CMAKE_INSTALL_PREFIX="${PREFIX}"
    # A conda prefix is always lib/, but GNUInstallDirs would pick lib64 or a Debian multiarch
    # directory depending on the build image. The :STRING type is required: GNUInstallDirs declares
    # this as a cache PATH, and CMake rewrites a relative PATH cache entry into an absolute one
    # against the working directory, which would install everything under the build tree.
    -D CMAKE_INSTALL_LIBDIR:STRING=lib
    -D CMAKE_PREFIX_PATH="${PREFIX}"
    # Matches the wheel build, and is what makes the bundled Kokkos install as shared libraries.
    -D BUILD_SHARED_LIBS=ON
    -D Python_EXECUTABLE="${PYTHON}"
    -D Python_ROOT_DIR="${PREFIX}"
    -D Python_FIND_STRATEGY=LOCATION
    -D NeoN_BUILD_PYTHON_BINDINGS=ON
    -D NeoN_BUILD_TESTS=OFF
    -D NeoN_BUILD_BENCHMARKS=OFF
    -D NeoN_BUILD_DOC=OFF
    -D NeoN_DEVEL_TOOLS=OFF
    # Same reduced dependency set as the published wheels: no MPI or optional HPC backends, so the
    # package resolves without an external HPC stack.
    -D NeoN_WITH_MPI=OFF
    -D NeoN_WITH_PETSC=OFF
    -D NeoN_WITH_ADIOS2=OFF
    -D NeoN_WITH_SUNDIALS=OFF
    # This is what installs libNeoN and the bundled Kokkos runtime into $PREFIX/lib. The CUDA wheel
    # turns it off because auditwheel vendors those libraries instead; there is no such step here,
    # so turning it off would ship a package without its own shared library.
    -D NeoN_INSTALL_CMAKE_PACKAGE=ON
    -D Kokkos_ENABLE_SERIAL=ON
)

case "${gpu_variant}" in
    cpu)
        cmake_args+=(
            -D Kokkos_ENABLE_CUDA=OFF
            -D Kokkos_ENABLE_HIP=OFF
        )
        ;;
    cuda)
        # Mirrors the CUDA wheel configuration, except that the CMake package files stay installed.
        # Umpire is off there because its BLT build does not survive the relocation cleanly.
        cmake_args+=(
            -D NeoN_WITH_UMPIRE=OFF
            -D CMAKE_CUDA_COMPILER="${BUILD_PREFIX}/bin/nvcc"
            -D CMAKE_CUDA_HOST_COMPILER="${CXX}"
            -D CMAKE_CUDA_ARCHITECTURES="${NEON_CUDA_ARCHITECTURES}"
            -D CMAKE_CUDA_STANDARD=20
            -D CMAKE_CUDA_STANDARD_REQUIRED=ON
            -D CMAKE_CUDA_EXTENSIONS=OFF
            -D CUDAToolkit_ROOT="${PREFIX}"
            -D Kokkos_ENABLE_CUDA=ON
            -D Kokkos_ENABLE_HIP=OFF
            -D Kokkos_ENABLE_COMPILE_AS_CMAKE_LANGUAGE=ON
            -D "Kokkos_ARCH_${NEON_KOKKOS_ARCH}=ON"
        )
        export CUDAToolkit_ROOT="${PREFIX}"
        export CUDAHOSTCXX="${CXX}"
        ;;
    rocm)
        # Ginkgo's HIP backend needs rocBLAS, rocSPARSE, rocThrust and rocPRIM, none of which exist
        # on conda-forge (only hip-devel, hipcc and hip-runtime-amd do). The ROCm package therefore
        # ships Kokkos HIP without a Ginkgo solver backend.
        cmake_args+=(
            -D NeoN_WITH_UMPIRE=OFF
            -D NeoN_WITH_GINKGO=OFF
            -D CMAKE_CXX_COMPILER=hipcc
            -D CMAKE_HIP_ARCHITECTURES="${NEON_HIP_ARCHITECTURES}"
            -D Kokkos_ENABLE_CUDA=OFF
            -D Kokkos_ENABLE_HIP=ON
            -D "Kokkos_ARCH_AMD_$(printf '%s' "${NEON_HIP_ARCHITECTURES}" | tr '[:lower:]' '[:upper:]')=ON"
        )
        ;;
    *)
        echo "Unknown NEON_GPU_VARIANT: ${gpu_variant}" >&2
        exit 1
        ;;
esac

cmake -S "${SRC_DIR}" -B "${build_dir}" "${cmake_args[@]}"
cmake --build "${build_dir}" --parallel "${CPU_COUNT}"
cmake --install "${build_dir}"

# The non-scikit-build install path drops the python package in $PREFIX/lib/python/neon, which is
# not on sys.path. Move it to site-packages; libNeoN and friends stay in $PREFIX/lib, where the
# rpath rewrite in the recipe points.
python_pkg_dir="${PREFIX}/lib/python/neon"
if [[ ! -d "${python_pkg_dir}" ]]; then
    echo "Expected the bindings to be installed to ${python_pkg_dir}" >&2
    echo "Installed instead:" >&2
    find "${PREFIX}" -name "_neon*.so" -o -name "_neon*.pyd" >&2 || true
    exit 1
fi

mkdir -p "${SP_DIR}"
rm -rf "${SP_DIR}/neon"
mv "${python_pkg_dir}" "${SP_DIR}/neon"
rmdir "${PREFIX}/lib/python"

# nanobind.stubgen writes _neon.pyi next to the built extension but no install rule picks it up.
# Ship it so editors and type checkers see the same API the wheels document.
stub="${build_dir}/bindings/neon/_neon.pyi"
if [[ -f "${stub}" ]]; then
    cp "${stub}" "${SP_DIR}/neon/"
fi
