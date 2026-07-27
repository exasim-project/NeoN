# SPDX-FileCopyrightText: 2026 NeoN authors
#
# SPDX-License-Identifier: Unlicense

# Set output directory based on build mode
if(DEFINED SKBUILD)
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
else()
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bindings/blockamr)
endif()

# Sources are listed once here: the same list feeds nanobind_add_module and the
# set_source_files_properties(... LANGUAGE CUDA) call below.
set(_blockamr_sources
    module.cpp
    init.cpp
    indextype.cpp
    box.cpp
    multifab.cpp
    geometry.cpp
    plotfile.cpp
    amrcore.cpp
    tagbox.cpp
    fillpatch.cpp
    solvers/linop.cpp
    stencil_kernels.cpp
    tile_layout.cpp
    bench/bindings.cpp)

# Kernels for the Kokkos-vs-AMReX operator bench. Compiled as their own object library, NOT as
# _blockamr sources -- but not because of an RDC fence: the whole module (including _blockamr
# itself) builds CUDA_SEPARABLE_COMPILATION OFF, see the rationale at line ~70 below.
# blockamr_bench remains a separate OBJECT library by history/inertia (it predates that fence
# being dropped), not because it needs to be one. Both backends' kernels belong in this target so
# the comparison runs under identical compile flags.
set(_blockamr_bench_sources
    bench/kokkos_spike.cpp
    ../../blockAmrSolvers/kokkos/runtime.cpp
    bench/operators.cpp
    ../../blockAmrSolvers/gmgKokkos/kernels.cpp
    ../../blockAmrSolvers/gmgKokkos/apply.cpp
    bench/gmg_vcycle_bench.cpp)

# Ginkgo-only sources: the matrix-free solver stack under solvers/, plus the
# shared common/, operators/ and krylov/ seams that now live in
# ../../blockAmrSolvers/ (T3).
set(_blockamr_ginkgo_sources
    ../../blockAmrSolvers/common/profiling.cpp
    ../../blockAmrSolvers/common/bc.cpp
    ../../blockAmrSolvers/operators/mlmg_ops.cpp
    ../../blockAmrSolvers/operators/face_coeff_op.cpp
    ../../blockAmrSolvers/operators/csr.cpp
    ../../blockAmrSolvers/krylov/krylov.cpp
    solvers/persistent.cpp
    solvers/ginkgo_solve.cpp)

nanobind_add_module(_blockamr NOMINSIZE ${_blockamr_sources})

# NeoN::NeoN brings Kokkos (via the NeoN_public_api INTERFACE target) for the bench/ operator
# comparison: the same cell kernels launched by amrex::ParallelFor and by Kokkos, over the same
# MultiFab memory.
target_link_libraries(_blockamr PRIVATE amrex NeoN::NeoN)

# The real ginkgo_solve.cpp (and the solvers/ stack it drives) only compiles against Ginkgo; without
# it a stub raising RuntimeError is built instead, so module.cpp stays unconditional.
if(NeoN_WITH_GINKGO)
  target_sources(_blockamr PRIVATE ${_blockamr_ginkgo_sources})
  target_link_libraries(_blockamr PRIVATE Ginkgo::ginkgo)
else()
  target_sources(_blockamr PRIVATE solvers/ginkgo_solve_stub.cpp)
endif()

# AMReX helper: propagate CUDA target & source properties (same as pyamrex)
if(NeoN_AMREX_GPU_BACKEND STREQUAL "CUDA")
  setup_target_for_cuda_compilation(_blockamr)
  target_compile_features(_blockamr PUBLIC cuda_std_17)
  # CUDA_SEPARABLE_COMPILATION OFF undoes what setup_target_for_cuda_compilation just
  # turned on, deliberately. Kokkos here is built with
  # Kokkos_ENABLE_CUDA_RELOCATABLE_DEVICE_CODE=OFF, and desul's guard
  # (tpls/desul/include/desul/atomics/Macros.hpp) is a preprocessor check on
  # __CUDACC_RDC__ -- so under -rdc=true ANY Kokkos header fails to compile on
  # inclusion, not just one that instantiates an atomic.
  #
  # This is why the fence is gone on the solvers/ side too, not just here: the executor
  # argument there is a NeoN::Executor, and NeoN/core/executor/executor.hpp includes
  # NeoN/core/memory/kokkos.hpp, so solvers/ already needs Kokkos headers to compile.
  # Dropping RDC for the whole module (this property) is what makes that possible.
  # Execution-verified for this revision: a CUDA configure+build of this target and
  # blockamr_bench produces zero "-rdc=true" in build.ninja, and solvers/*.cpp compiles
  # clean as CUDA non-RDC. If AMReX ever needs a device symbol resolved across
  # translation units this breaks at LINK time, loudly, not silently.
  set_target_properties(_blockamr PROPERTIES CUDA_EXTENSIONS OFF CUDA_STANDARD_REQUIRED ON
                                             CUDA_SEPARABLE_COMPILATION OFF)
  # Mark all .cpp sources to be compiled as CUDA
  set_source_files_properties(${_blockamr_sources} PROPERTIES LANGUAGE CUDA)
  if(NeoN_WITH_GINKGO)
    set_source_files_properties(${_blockamr_ginkgo_sources} PROPERTIES LANGUAGE CUDA)
  else()
    set_source_files_properties(solvers/ginkgo_solve_stub.cpp PROPERTIES LANGUAGE CUDA)
  endif()
endif()

# The bench object library: same CUDA language, architecture and CUDA_SEPARABLE_COMPILATION
# OFF as _blockamr (see the rationale on _blockamr's own property, above) -- kept as a
# separate OBJECT library by history/inertia rather than because Kokkos needs a fence from
# the rest of the module; the whole module builds non-RDC now.
add_library(blockamr_bench OBJECT ${_blockamr_bench_sources})
target_link_libraries(blockamr_bench PRIVATE amrex NeoN::NeoN)
set_target_properties(blockamr_bench PROPERTIES CUDA_SEPARABLE_COMPILATION OFF
                                                POSITION_INDEPENDENT_CODE ON)
if(NeoN_AMREX_GPU_BACKEND STREQUAL "CUDA")
  set_source_files_properties(${_blockamr_bench_sources} PROPERTIES LANGUAGE CUDA)
  get_target_property(_blockamr_cuda_archs _blockamr CUDA_ARCHITECTURES)
  set_target_properties(
    blockamr_bench
    PROPERTIES CUDA_ARCHITECTURES "${_blockamr_cuda_archs}"
               CUDA_EXTENSIONS OFF
               CUDA_STANDARD_REQUIRED ON)
  target_compile_features(blockamr_bench PUBLIC cuda_std_17)
endif()
target_link_libraries(_blockamr PRIVATE blockamr_bench)

# Set RPATH so _blockamr.so can find libamrex at ../lib/ (blockamr is a top-level package:
# site-packages/blockamr/_blockamr.so -> ../lib).
set_target_properties(_blockamr PROPERTIES INSTALL_RPATH "$ORIGIN/../lib" BUILD_WITH_INSTALL_RPATH
                                                                          FALSE)

# Installation
if(DEFINED SKBUILD)
  install(TARGETS _blockamr LIBRARY DESTINATION blockamr COMPONENT python)
  # CMAKE_CURRENT_SOURCE_DIR (this dir = src/bindings/blockAMR) not CMAKE_SOURCE_DIR: when NeoN is
  # add_subdirectory()'d into neofoam, CMAKE_SOURCE_DIR is the neofoam root and the blockamr .py
  # sources are missed.
  install(
    DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../../blockamr/
    DESTINATION blockamr
    COMPONENT python
    FILES_MATCHING
    PATTERN "*.py")
  # Install AMReX shared lib alongside the wheel
  install(TARGETS amrex_3d LIBRARY DESTINATION lib COMPONENT python)
  # Install Ginkgo shared libs alongside the wheel (under SKBUILD Ginkgo is always the CPM build —
  # see cmake/CxxThirdParty.cmake)
  if(NeoN_WITH_GINKGO)
    install(
      TARGETS ginkgo
              ginkgo_device
              ginkgo_omp
              ginkgo_cuda
              ginkgo_hip
              ginkgo_dpcpp
              ginkgo_reference
      LIBRARY DESTINATION lib COMPONENT python)
  endif()
else()
  install(TARGETS _blockamr LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/python/blockamr)
endif()
