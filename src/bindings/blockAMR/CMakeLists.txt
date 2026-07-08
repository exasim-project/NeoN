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
  tile_layout.cpp
  eb2.cpp
  ebfactory.cpp)

target_link_libraries(_blockamr PRIVATE amrex)

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
    tile_layout.cpp
    eb2.cpp
    ebfactory.cpp
    PROPERTIES LANGUAGE CUDA)
endif()

# Set RPATH so _blockamr.so can find libamrex at ../../lib/
set_target_properties(_blockamr PROPERTIES INSTALL_RPATH "$ORIGIN/../../lib" BUILD_WITH_INSTALL_RPATH
                                                                              FALSE)

# Installation
if(DEFINED SKBUILD)
  install(TARGETS _blockamr LIBRARY DESTINATION neon/blockamr COMPONENT python)
  install(
    DIRECTORY ${CMAKE_SOURCE_DIR}/src/neon/blockamr/
    DESTINATION neon/blockamr
    COMPONENT python
    FILES_MATCHING
    PATTERN "*.py")
  # Install AMReX shared lib alongside the wheel
  install(TARGETS amrex_3d LIBRARY DESTINATION lib COMPONENT python)
else()
  install(TARGETS _blockamr LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/python/neon/blockamr)
endif()
