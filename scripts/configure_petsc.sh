# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 NeoN authors


#!/bin/bash

## example script how to download, configure, compile and install petsc
## see also https://petsc.org/release/install/

CURRENTDIR="$(pwd)"

# clone latest release
git clone --depth 1 --branch v3.22.2 https://gitlab.com/petsc/petsc.git petsc
cd petsc

## running version
## the problem here is still that the compiler and compiler options used later in the build of NeoN have to be used also in advance to configure petsc
## how to solve this?
./configure --with-64-bit-indices=0 --with-precision=double --with-cuda=1 --with-cuda-dir=/usr/local/cuda-12.3 --with-mpi=1 --with-fc=0 --force --with-mpi=1 --with-32bits-pci-domain=1 --prefix=optClangkokkos4dot3dot01/petsc --with-cc=mpicc --with-cxx=mpicxx --with-debugging=no --download-kokkos --download-kokkos-kernels --with-kokkos-kernels=1 --with-kokkos=1 --download-kokkos-cmake-arguments='-DKokkos_ENABLE_CUDA_CONSTEXPR=ON -DCMAKE_C_COMPILER=clang-18 -DCMAKE_CXX_COMPILER=clang++-18 -DKokkos_ENABLE_DEBUG=ON -DKokkos_ENABLE_DEBUG_BOUNDS_CHECK=ON -DKokkos_ENABLE_DEBUG_DUALVIEW_MODIFY_CHECK=ON' --download-kokkos-commit=4.3.01 --download-kokkos-kernels-commit=4.3.01

# compile petsc
make PETSC_DIR=$CURRENTDIR/petsc PETSC_ARCH=arch-linux-c-opt all

#install petsc
make PETSC_DIR=$CURRENTDIR/petsc PETSC_ARCH=arch-linux-c-opt install

export PKG_CONFIG_PATH=$CURRENTDIR/opt/petsc/lib/pkgconfig
