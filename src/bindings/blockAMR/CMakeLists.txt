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
    linop.cpp
    stencil_kernels.cpp
    tile_layout.cpp
    bench_bindings.cpp)

# The registrar for the Ginkgo-gated matrix-free solver stack (blockAmrSolvers/{common,
# krylov,operators,gmg,solve}/, built by blockAmrSolvers/CMakeLists.txt as the
# blockamr_solvers OBJECT library and linked below).
set(_blockamr_ginkgo_sources ginkgo_solve.cpp)

nanobind_add_module(_blockamr NOMINSIZE ${_blockamr_sources})

# NeoN::NeoN brings Kokkos (via the NeoN_public_api INTERFACE target) for the
# blockAmrSolvers/bench/ operator comparison: the same cell kernels launched by
# amrex::ParallelFor and by Kokkos, over the same MultiFab memory.
target_link_libraries(_blockamr PRIVATE amrex NeoN::NeoN)

# The real ginkgo_solve.cpp (and the blockamr_solvers stack it drives) only compiles against
# Ginkgo; without it a stub raising RuntimeError is built instead, so module.cpp stays
# unconditional.
if(NeoN_WITH_GINKGO)
  target_sources(_blockamr PRIVATE ${_blockamr_ginkgo_sources})
  target_link_libraries(_blockamr PRIVATE Ginkgo::ginkgo blockamr_solvers)
else()
  target_sources(_blockamr PRIVATE ginkgo_solve_stub.cpp)
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
  # This is why the fence is gone on the blockAmrSolvers/ side too, not just here: the executor
  # argument there is a NeoN::Executor, and NeoN/core/executor/executor.hpp includes
  # NeoN/core/memory/kokkos.hpp, so blockAmrSolvers/ already needs Kokkos headers to compile.
  # Dropping RDC for the whole module (this property) is what makes that possible.
  # Execution-verified for this revision: a CUDA configure+build of this target and
  # blockamr_kokkos produces zero "-rdc=true" in build.ninja, and blockAmrSolvers/**/*.cpp
  # compiles clean as CUDA non-RDC. If AMReX ever needs a device symbol resolved across
  # translation units this breaks at LINK time, loudly, not silently.
  set_target_properties(_blockamr PROPERTIES CUDA_EXTENSIONS OFF CUDA_STANDARD_REQUIRED ON
                                             CUDA_SEPARABLE_COMPILATION OFF)
  # Mark all .cpp sources to be compiled as CUDA
  set_source_files_properties(${_blockamr_sources} PROPERTIES LANGUAGE CUDA)
  if(NeoN_WITH_GINKGO)
    set_source_files_properties(${_blockamr_ginkgo_sources} PROPERTIES LANGUAGE CUDA)
  else()
    set_source_files_properties(ginkgo_solve_stub.cpp PROPERTIES LANGUAGE CUDA)
  endif()
endif()

# blockamr_kokkos (gmgKokkos/ kokkos/ bench/) is defined by blockAmrSolvers/CMakeLists.txt
# as its own OBJECT library, same CUDA_SEPARABLE_COMPILATION OFF as _blockamr (see the
# rationale on _blockamr's own property, above) -- kept as a separate OBJECT library by
# history/inertia rather than because Kokkos needs a fence from the rest of the module;
# the whole module builds non-RDC now.
target_link_libraries(_blockamr PRIVATE blockamr_kokkos)

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
