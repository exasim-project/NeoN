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
  schemes/stencil_kernels.cpp
  ibm/cell_type.cpp
  ibm/ghost_cell.cpp
  schemes/boundary/wall_frame.cpp
  schemes/boundary/robin_closure.cpp
  schemes/boundary/laplacian_ghost_cell.cpp
  schemes/boundary/div_ghost_cell.cpp
  schemes/boundary/grad_ghost_cell.cpp
  tile_layout.cpp
  wall_table.cpp
  band_table.cpp)

target_link_libraries(_blockamr PRIVATE amrex)

# Four sources are pinned BITWISE against a numpy peer and therefore may not have a multiply-add
# contracted into an fma -- one rounding where numpy does two:
#
# * ibm/ghost_cell.cpp against src/blockamr/ibm/ghost_cell.py (tasks.md B31, review.md Q29(d) -- the
#   ULP fallback is refused). Its arithmetic is + - * / fabs floor max only.
# * schemes/boundary/robin_closure.cpp against wall_closure in
#   src/blockamr/schemes/boundary/ghost_cell.py (tasks.md B30b, review.md Q41/Q43). The contraction
#   site is `beta - alpha * distance`, and it moves bits wherever beta ~ alpha*d.
# * schemes/boundary/laplacian_ghost_cell.cpp against _closed_flux_rows in the same v1 module
#   (tasks.md B32). It inlines robin.H's closure AND accumulates its own donor sums, diagonal and
#   constant -- three multiply-accumulate shapes of its own -- so the flag is load-bearing for this
#   TU's bar rather than merely inherited.
# * schemes/boundary/div_ghost_cell.cpp against _face_balance_rows in the same v1 module (tasks.md
#   B33). It is the FIRST caller of robin.H's atLinear/atConstant, whose `value + d*grad` is a
#   contraction site of its own, and it adds `dG = sdf + step*dx*n`, a face-neighbour accumulator
#   bank, the donor sums, the diagonal and the constant.
# * schemes/boundary/grad_ghost_cell.cpp against the SAME _face_balance_rows in the same v1 module
#   (tasks.md B34), reached from GhostCellGrad.rows with axes=(0,), flux=1 and weight_self=0.5. It
#   is robin.H's FOURTH includer and atLinear/atConstant's second caller, and it carries the same
#   accumulator shapes on one axis.
#
# GCC defaults to -ffp-contract=fast and nvcc to --fmad=true (re-asserted by AMReX's
# --use_fast_math), so contraction is turned off here, scoped to these sources so no other
# translation unit's numerics move. See each file header for the rejected alternatives.
#
# NOTE: schemes/boundary/robin.H is a HEADER -- its arithmetic is inlined with the INCLUDING TU's
# flags. Any further TU that includes it and claims v1 parity must be added to this list.
#
# review.md Q36, RULED at B32: the list stays PER-FILE OPT-IN and the directory-level IEEE posture
# is refuted in part. schemes/stencil_kernels.cpp must NOT take these flags (Q42 -- they move
# A1/B41/A4/A8's recorded accuracy digits) and schemes/boundary/wall_frame.cpp shipped without them
# by its own recorded decision, so a directory posture is really directory-wide flags plus an
# exclusion list -- and the two fail in opposite directions. Forgetting an entry HERE turns that
# TU's own bitwise rows red, loudly; forgetting one on an exclusion list moves a tolerance row's
# digits and nothing notices. The list's known weakness -- one edit away from being silently
# incomplete -- is closed mechanically instead: test_ibm_laplacian_ghost_cell.py parses the call
# below and asserts that every includer of robin.H or ibm/ghost_cell.H appears in it.
set(_ghost_cell_fp_flags
    "$<$<COMPILE_LANGUAGE:CXX>:-ffp-contract=off>" "$<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>"
    "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-ffp-contract=off>")
set_source_files_properties(
  ibm/ghost_cell.cpp schemes/boundary/robin_closure.cpp schemes/boundary/laplacian_ghost_cell.cpp
  schemes/boundary/div_ghost_cell.cpp schemes/boundary/grad_ghost_cell.cpp
  PROPERTIES COMPILE_OPTIONS "${_ghost_cell_fp_flags}")

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
    schemes/stencil_kernels.cpp
    ibm/cell_type.cpp
    ibm/ghost_cell.cpp
    schemes/boundary/wall_frame.cpp
    schemes/boundary/robin_closure.cpp
    schemes/boundary/laplacian_ghost_cell.cpp
    schemes/boundary/div_ghost_cell.cpp
    schemes/boundary/grad_ghost_cell.cpp
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
