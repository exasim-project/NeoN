# SPDX-FileCopyrightText: 2026 NeoN authors
#
# SPDX-License-Identifier: Unlicense

# Applies the Ginkgo patches idempotently.
#
# FetchContent re-runs PATCH_COMMAND whenever an existing build tree is reconfigured, and a plain
# `git apply` fails the second time round ("patch does not apply"), taking the whole configure with
# it. Restoring the checkout first makes the step repeatable: the tree is put back to the pinned
# tag, then the patches are applied to it. Probing with `git apply --reverse --check` instead does
# not work here because the patches are cumulative and git cannot reverse a stack in one pass.
#
# Expects GINKGO_PATCH_DIR. The patch step runs with the Ginkgo source tree as its working
# directory, which git inherits.

# Order matters: the fine-op reuse cache builds on the local stack. Disable the fine-op cache at
# runtime with GINKGO_PGM_REUSE_FINE_OP=0.
set(_patches ${GINKGO_PATCH_DIR}/ginkgo_local_stack.patch
             ${GINKGO_PATCH_DIR}/ginkgo_pgm_fine_op_cache.patch)

find_package(Git REQUIRED)

execute_process(
  COMMAND ${GIT_EXECUTABLE} checkout -- .
  RESULT_VARIABLE _rc
  ERROR_VARIABLE _err)

if(NOT _rc EQUAL 0)
  message(FATAL_ERROR "Failed to restore the Ginkgo checkout before patching:\n${_err}")
endif()

execute_process(
  COMMAND ${GIT_EXECUTABLE} apply ${_patches}
  RESULT_VARIABLE _rc
  ERROR_VARIABLE _err)

if(NOT _rc EQUAL 0)
  string(REPLACE ";" "\n  " _patch_list "${_patches}")
  message(FATAL_ERROR "Failed to apply Ginkgo patches:\n  ${_patch_list}\n${_err}")
endif()

message(STATUS "Applied Ginkgo patches")
