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

# Kokkos runtime (initialize/finalize alongside AMReX).
set(_blockamr_kokkos_sources core/runtime.cpp)

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
    bindings/fillpatch.cpp
    bindings/stencilKernels.cpp
)

nanobind_add_module(_blockamr NOMINSIZE ${_blockamr_sources})

# NeoN::NeoN brings Kokkos (via the NeoN_public_api INTERFACE target) for the bench/ operator
# comparison: the same cell kernels launched by amrex::ParallelFor and by Kokkos, over the same
# MultiFab memory.
target_link_libraries(_blockamr PRIVATE amrex NeoN::NeoN)

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
endif()

target_link_libraries(_blockamr PRIVATE blockamr_kokkos)

# Set RPATH so _blockamr.so can find libamrex at ../lib/ (blockamr is a top-level package:
# site-packages/blockamr/_blockamr.so -> ../lib).
set_target_properties(_blockamr PROPERTIES INSTALL_RPATH "$ORIGIN/../lib")
set_target_properties(_blockamr PROPERTIES BUILD_WITH_INSTALL_RPATH FALSE)

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
else()
  install(TARGETS _blockamr LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/python/blockamr)
endif()
