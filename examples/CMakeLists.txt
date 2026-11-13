# SPDX-FileCopyrightText: 2026 NeoN authors
#
# SPDX-License-Identifier: Unlicense

# every example is also added as a test, which fails if the results are wrong
function(NeoN_example EXAMPLE)
  add_executable(neon_example_${EXAMPLE} "${EXAMPLE}.cpp")
  target_link_libraries(neon_example_${EXAMPLE} PRIVATE NeoN_warnings NeoN_options NeoN)
  set(EXAMPLE_DIR ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/examples)
  set_target_properties(neon_example_${EXAMPLE} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${EXAMPLE_DIR})
  add_test(
    NAME example_${EXAMPLE}
    COMMAND neon_example_${EXAMPLE}
    WORKING_DIRECTORY ${EXAMPLE_DIR})
endfunction()

# needs a linear solver
if(NeoN_WITH_GINKGO)
  neon_example(steadyDiffusion)
endif()
