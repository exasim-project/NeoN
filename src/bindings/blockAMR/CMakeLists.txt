# SPDX-FileCopyrightText: 2026 NeoN authors
#
# SPDX-License-Identifier: Unlicense

# Set output directory based on build mode
if(DEFINED SKBUILD)
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
else()
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bindings/blockamr)
endif()

nanobind_add_module(
  _blockamr
  NOMINSIZE
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
  ibm/cell_type.cpp
  ibm/ghost_cell.cpp
  schemes/boundary/wall_frame.cpp
  tile_layout.cpp
  wall_table.cpp
  band_table.cpp)

target_link_libraries(_blockamr PRIVATE amrex)

# ibm/ghost_cell.cpp is pinned BITWISE against its numpy peer (src/blockamr/ibm/ghost_cell.py;
# plans/IBM/tasks.md B31, review.md Q29(d) -- the ULP fallback is refused). Its arithmetic is + - *
# / fabs floor max only, so the only way the compiler can change the result is by contracting a
# multiply-add into an fma: one rounding where numpy does two. GCC defaults to -ffp-contract=fast
# and nvcc to --fmad=true (re-asserted by AMReX's --use_fast_math), so contraction is turned off
# here, scoped to this ONE source so no other translation unit's numerics move. See the file header
# for the rejected alternatives.
set(_ghost_cell_fp_flags
    "$<$<COMPILE_LANGUAGE:CXX>:-ffp-contract=off>" "$<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>"
    "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-ffp-contract=off>")
set_source_files_properties(ibm/ghost_cell.cpp PROPERTIES COMPILE_OPTIONS "${_ghost_cell_fp_flags}")

# AMReX helper: propagate CUDA target & source properties (same as pyamrex)
if(NeoN_AMREX_GPU_BACKEND STREQUAL "CUDA")
  setup_target_for_cuda_compilation(_blockamr)
  target_compile_features(_blockamr PUBLIC cuda_std_17)
  set_target_properties(_blockamr PROPERTIES CUDA_EXTENSIONS OFF CUDA_STANDARD_REQUIRED ON)
  # Mark all .cpp sources to be compiled as CUDA
  set_source_files_properties(
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
    ibm/cell_type.cpp
    ibm/ghost_cell.cpp
    schemes/boundary/wall_frame.cpp
    tile_layout.cpp
    wall_table.cpp
    band_table.cpp
    PROPERTIES LANGUAGE CUDA)
endif()

# Set RPATH so _blockamr.so can find libamrex at ../lib/ (blockamr is a top-level package:
# site-packages/blockamr/_blockamr.so -> ../lib).
set_target_properties(_blockamr PROPERTIES INSTALL_RPATH "$ORIGIN/../lib")
set_target_properties(_blockamr PROPERTIES BUILD_WITH_INSTALL_RPATH FALSE)

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
else()
  install(TARGETS _blockamr LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/python/blockamr)
endif()
