# SPDX-FileCopyrightText: 2025 NeoN authors
#
# SPDX-License-Identifier: MIT
# SPDX-License-Identifier: Unlicense

cmake_path(GET CMAKE_LIBRARY_OUTPUT_DIRECTORY PARENT_PATH PYGINKGO_TESTS_WORK_DIR)

# * Function to generate bindings unit test.
function(neon_bindings_test TEST)
  # Correct variable according to https://cmake.org/cmake/help/latest/module/FindPython.html
  # Printing on test skip: https://stackoverflow.com/a/64776498
  add_test(
    NAME neon_bindings_test_${TEST}
    COMMAND ${Python_EXECUTABLE} -m pytest -vv -rs ${CMAKE_CURRENT_SOURCE_DIR}/${TEST}.py
    WORKING_DIRECTORY ${PYGINKGO_TESTS_WORK_DIR})
endfunction()

neon_bindings_test(executor)
