# SPDX-FileCopyrightText: 2026 NeoN authors
#
# SPDX-License-Identifier: Unlicense

# Two OBJECT libraries, linked by bindings/blockAMR/CMakeLists.txt (the module's only
# consumer). OBJECT libraries compile each TU exactly once and merely attribute it to a
# target, so splitting this out of _blockamr costs nothing at compile time.

# The Ginkgo-gated matrix-free solver stack: common/ krylov/ operators/ gmg/ solve/. Only
# exists when Ginkgo is available; without it bindings/blockAMR/CMakeLists.txt compiles
# ginkgo_solve_stub.cpp instead and this target is never defined.
if(NeoN_WITH_GINKGO)
  set(_blockamr_solvers_sources
      common/profiling.cpp
      common/bc.cpp
      operators/mlmg_ops.cpp
      operators/face_coeff_op.cpp
      operators/csr.cpp
      krylov/krylov.cpp
      solve/persistent.cpp
      solve/oneshot.cpp)

  add_library(blockamr_solvers OBJECT ${_blockamr_solvers_sources})
  target_link_libraries(blockamr_solvers PRIVATE amrex NeoN::NeoN Ginkgo::ginkgo)
  # Same CUDA_SEPARABLE_COMPILATION OFF as _blockamr itself (rationale kept in one place,
  # bindings/blockAMR/CMakeLists.txt near setup_target_for_cuda_compilation(_blockamr)) --
  # this stack already needs Kokkos headers (via NeoN/core/executor/executor.hpp), which
  # is only possible non-RDC. OBJECT libraries need POSITION_INDEPENDENT_CODE ON
  # explicitly: unlike _blockamr (a shared module), they are never linked directly.
  set_target_properties(blockamr_solvers PROPERTIES CUDA_SEPARABLE_COMPILATION OFF
                                                     POSITION_INDEPENDENT_CODE ON)
  if(NeoN_AMREX_GPU_BACKEND STREQUAL "CUDA")
    set_source_files_properties(${_blockamr_solvers_sources} PROPERTIES LANGUAGE CUDA)
    # AMREX_CUDA_ARCHS is the same CACHE INTERNAL variable
    # setup_target_for_cuda_compilation(_blockamr) reads for _blockamr itself
    # (AMReXTargetHelpers.cmake) -- read directly rather than copying the property off
    # _blockamr, since this directory is processed before bindings/blockAMR/CMakeLists.txt
    # defines that target.
    set_target_properties(
      blockamr_solvers
      PROPERTIES CUDA_ARCHITECTURES "${AMREX_CUDA_ARCHS}"
                 CUDA_EXTENSIONS OFF
                 CUDA_STANDARD_REQUIRED ON)
    target_compile_features(blockamr_solvers PUBLIC cuda_std_17)
  endif()
endif()

# Kernels for the Kokkos-vs-AMReX operator bench (gmgKokkos/ kokkos/ bench/) plus the
# single-TU device kernels (gmg/gmg_kernels.cpp, common/device_kernels.cpp). Compiled as
# its own object library, NOT into _blockamr directly -- but not because of an RDC fence:
# the whole module (including _blockamr itself) builds CUDA_SEPARABLE_COMPILATION OFF,
# see the rationale on _blockamr's own property in bindings/blockAMR/CMakeLists.txt.
# blockamr_kokkos (formerly blockamr_bench) remains a separate OBJECT library by
# history/inertia -- it predates that fence being dropped -- not because it needs to be
# one. Both backends' kernels belong in this target so the comparison runs under
# identical compile flags. Builds regardless of Ginkgo, exactly as blockamr_bench did.
set(_blockamr_kokkos_sources
    bench/kokkos_spike.cpp
    kokkos/runtime.cpp
    bench/operators.cpp
    common/device_kernels.cpp
    gmg/gmg_kernels.cpp
    gmgKokkos/kernels.cpp
    gmgKokkos/apply.cpp
    bench/gmg_vcycle_bench.cpp)

add_library(blockamr_kokkos OBJECT ${_blockamr_kokkos_sources})
target_link_libraries(blockamr_kokkos PRIVATE amrex NeoN::NeoN)
set_target_properties(blockamr_kokkos PROPERTIES CUDA_SEPARABLE_COMPILATION OFF
                                                 POSITION_INDEPENDENT_CODE ON)
if(NeoN_AMREX_GPU_BACKEND STREQUAL "CUDA")
  set_source_files_properties(${_blockamr_kokkos_sources} PROPERTIES LANGUAGE CUDA)
  set_target_properties(
    blockamr_kokkos
    PROPERTIES CUDA_ARCHITECTURES "${AMREX_CUDA_ARCHS}"
               CUDA_EXTENSIONS OFF
               CUDA_STANDARD_REQUIRED ON)
  target_compile_features(blockamr_kokkos PUBLIC cuda_std_17)
endif()
