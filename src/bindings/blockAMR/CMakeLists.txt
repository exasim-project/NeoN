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
    tile_layout.cpp)

# Ginkgo-only sources: the matrix-free solver stack under solvers/.
set(_blockamr_ginkgo_sources
    solvers/profiling.cpp
    solvers/bc.cpp
    solvers/mlmg_ops.cpp
    solvers/face_coeff_op.cpp
    solvers/krylov.cpp
    solvers/csr.cpp
    solvers/persistent.cpp
    solvers/ginkgo_solve.cpp)

nanobind_add_module(_blockamr NOMINSIZE ${_blockamr_sources})

target_link_libraries(_blockamr PRIVATE amrex)

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
  set_target_properties(_blockamr PROPERTIES CUDA_EXTENSIONS OFF CUDA_STANDARD_REQUIRED ON)
  # Mark all .cpp sources to be compiled as CUDA
  set_source_files_properties(${_blockamr_sources} PROPERTIES LANGUAGE CUDA)
  if(NeoN_WITH_GINKGO)
    set_source_files_properties(${_blockamr_ginkgo_sources} PROPERTIES LANGUAGE CUDA)
  else()
    set_source_files_properties(solvers/ginkgo_solve_stub.cpp PROPERTIES LANGUAGE CUDA)
  endif()
endif()

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
