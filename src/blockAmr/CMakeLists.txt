# SPDX-FileCopyrightText: 2026 NeoN authors
#
# SPDX-License-Identifier: Unlicense

# The blockAmr component: the block-structured AMReX engine, its matrix-free solver stack, and the
# _blockamr Python module. Self-contained -- headers in include/NeoN/blockAmr/, sources here, and
# this file is the component's only entry point. The whole component is gated on NeoN_WITH_BLOCKAMR
# by the root CMakeLists.txt; NeoN core never refers to it.

# --- Object libraries ---
#
# Both compile each TU exactly once and merely attribute it to a target, so splitting them out of
# _blockamr costs nothing at compile time. They are defined before _blockamr in this same file,
# which is all the ordering guarantee the link below needs.

# The Ginkgo-gated matrix-free solver stack (core/ linearAlgebra/). Only exists when Ginkgo is
# available; without it _blockamr compiles bindings/linearAlgebraStub.cpp instead and this target is
# never defined.
if(NeoN_WITH_GINKGO)
  set(_blockamr_solvers_sources
      core/profiling.cpp
      core/bc.cpp
      operators/laplacian.cpp
      linearAlgebra/ginkgo/adapt.cpp
      linearAlgebra/matrixFree/mlmgOps.cpp
      linearAlgebra/matrixFree/faceCoeffOp.cpp
      linearAlgebra/krylov/krylov.cpp
      linearAlgebra/precond.cpp
      linearAlgebra/solve/persistent.cpp
      linearAlgebra/solve/oneshot.cpp)

  add_library(blockamr_solvers OBJECT ${_blockamr_solvers_sources})
  target_link_libraries(blockamr_solvers PRIVATE amrex NeoN::NeoN Ginkgo::ginkgo)
  # Same CUDA_SEPARABLE_COMPILATION OFF as _blockamr itself (rationale kept in one place, near
  # setup_target_for_cuda_compilation(_blockamr) below) -- this stack already needs Kokkos headers
  # (via NeoN/core/executor/executor.hpp), which is only possible non-RDC. OBJECT libraries need
  # POSITION_INDEPENDENT_CODE ON explicitly: unlike _blockamr (a shared module), they are never
  # linked directly.
  set_target_properties(blockamr_solvers PROPERTIES CUDA_SEPARABLE_COMPILATION OFF
                                                    POSITION_INDEPENDENT_CODE ON)
  if(NeoN_AMREX_GPU_BACKEND STREQUAL "CUDA")
    set_source_files_properties(${_blockamr_solvers_sources} PROPERTIES LANGUAGE CUDA)
    # AMREX_CUDA_ARCHS is the same CACHE INTERNAL variable
    # setup_target_for_cuda_compilation(_blockamr) reads for _blockamr itself
    # (AMReXTargetHelpers.cmake) -- read directly rather than copying the property off _blockamr,
    # which is not defined yet at this point in the file.
    set_target_properties(
      blockamr_solvers
      PROPERTIES CUDA_ARCHITECTURES "${AMREX_CUDA_ARCHS}"
                 CUDA_EXTENSIONS OFF
                 CUDA_STANDARD_REQUIRED ON)
    target_compile_features(blockamr_solvers PUBLIC cuda_std_17)
  endif()
endif()

# Kernels for the Kokkos-vs-AMReX operator bench (linearAlgebra/gmgKokkos/ core/ bench/) plus the
# single-TU device kernels. Compiled as its own object library, NOT into _blockamr directly -- but
# not because of an RDC fence: the whole component builds CUDA_SEPARABLE_COMPILATION OFF, see the
# rationale on _blockamr's own property below. blockamr_kokkos (formerly blockamr_bench) remains a
# separate OBJECT library by history/inertia -- it predates that fence being dropped -- not because
# it needs to be one. Both backends' kernels belong in this target so the comparison runs under
# identical compile flags. Builds regardless of Ginkgo, exactly as blockamr_bench did.
set(_blockamr_kokkos_sources
    bench/kokkosSpike.cpp
    core/runtime.cpp
    bench/operators.cpp
    core/deviceKernels.cpp
    linearAlgebra/gmg/gmgKernels.cpp
    linearAlgebra/gmgKokkos/kernels.cpp
    linearAlgebra/gmgKokkos/apply.cpp
    bench/gmgVcycleBench.cpp)

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

# --- The _blockamr Python module ---

# Set output directory based on build mode
if(DEFINED SKBUILD)
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
else()
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bindings/blockamr)
endif()

# Sources are listed once here: the same list feeds nanobind_add_module and the
# set_source_files_properties(... LANGUAGE CUDA) call below.
set(_blockamr_sources
    bindings/module.cpp
    bindings/init.cpp
    bindings/indextype.cpp
    bindings/box.cpp
    bindings/multifab.cpp
    bindings/geometry.cpp
    bindings/plotfile.cpp
    bindings/amrcore.cpp
    bindings/tagbox.cpp
    bindings/fillpatch.cpp
    bindings/linop.cpp
    bindings/stencilKernels.cpp
    bindings/tileLayout.cpp
    bindings/benchBindings.cpp)

# The registrar for the Ginkgo-gated matrix-free solver stack built above as the blockamr_solvers
# OBJECT library and linked below.
set(_blockamr_ginkgo_sources bindings/linearAlgebra.cpp)

nanobind_add_module(_blockamr NOMINSIZE ${_blockamr_sources})

# NeoN::NeoN brings Kokkos (via the NeoN_public_api INTERFACE target) for the bench/ operator
# comparison: the same cell kernels launched by amrex::ParallelFor and by Kokkos, over the same
# MultiFab memory.
target_link_libraries(_blockamr PRIVATE amrex NeoN::NeoN)

# The real bindings/linearAlgebra.cpp (and the blockamr_solvers stack it drives) only compiles
# against Ginkgo; without it a stub raising RuntimeError is built instead, so bindings/module.cpp
# stays unconditional.
if(NeoN_WITH_GINKGO)
  target_sources(_blockamr PRIVATE ${_blockamr_ginkgo_sources})
  target_link_libraries(_blockamr PRIVATE Ginkgo::ginkgo blockamr_solvers)
else()
  target_sources(_blockamr PRIVATE bindings/linearAlgebraStub.cpp)
endif()

# AMReX helper: propagate CUDA target & source properties (same as pyamrex)
if(NeoN_AMREX_GPU_BACKEND STREQUAL "CUDA")
  setup_target_for_cuda_compilation(_blockamr)
  target_compile_features(_blockamr PUBLIC cuda_std_17)
  # CUDA_SEPARABLE_COMPILATION OFF undoes what setup_target_for_cuda_compilation just turned on,
  # deliberately. Kokkos here is built with Kokkos_ENABLE_CUDA_RELOCATABLE_DEVICE_CODE=OFF, and
  # desul's guard (tpls/desul/include/desul/atomics/Macros.hpp) is a preprocessor check on
  # __CUDACC_RDC__ -- so under -rdc=true ANY Kokkos header fails to compile on inclusion, not just
  # one that instantiates an atomic.
  #
  # This is why the fence is gone on the solver-stack side too, not just here: the executor argument
  # there is a NeoN::Executor, and NeoN/core/executor/executor.hpp includes
  # NeoN/core/memory/kokkos.hpp, so linearAlgebra/ already needs Kokkos headers to compile. Dropping
  # RDC for the whole component (this property) is what makes that possible. Execution-verified for
  # this revision: a CUDA configure+build of this target and blockamr_kokkos produces zero
  # "-rdc=true" in build.ninja, and the component's sources compile clean as CUDA non-RDC. If AMReX
  # ever needs a device symbol resolved across translation units this breaks at LINK time, loudly,
  # not silently.
  set_target_properties(
    _blockamr
    PROPERTIES CUDA_EXTENSIONS OFF
               CUDA_STANDARD_REQUIRED ON
               CUDA_SEPARABLE_COMPILATION OFF)
  # Mark all .cpp sources to be compiled as CUDA
  set_source_files_properties(${_blockamr_sources} PROPERTIES LANGUAGE CUDA)
  if(NeoN_WITH_GINKGO)
    set_source_files_properties(${_blockamr_ginkgo_sources} PROPERTIES LANGUAGE CUDA)
  else()
    set_source_files_properties(bindings/linearAlgebraStub.cpp PROPERTIES LANGUAGE CUDA)
  endif()
endif()

target_link_libraries(_blockamr PRIVATE blockamr_kokkos)

# Set RPATH so _blockamr.so can find libamrex at ../lib/ (blockamr is a top-level package:
# site-packages/blockamr/_blockamr.so -> ../lib).
set_target_properties(_blockamr PROPERTIES INSTALL_RPATH "$ORIGIN/../lib" BUILD_WITH_INSTALL_RPATH
                                                                          FALSE)

# Installation
if(DEFINED SKBUILD)
  install(TARGETS _blockamr LIBRARY DESTINATION blockamr COMPONENT python)
  # CMAKE_CURRENT_SOURCE_DIR (this dir = src/blockAmr) not CMAKE_SOURCE_DIR: when NeoN is
  # add_subdirectory()'d into neofoam, CMAKE_SOURCE_DIR is the neofoam root and the blockamr .py
  # sources are missed.
  install(
    DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/python/blockamr/
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
