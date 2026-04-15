# SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
#
# SPDX-License-Identifier: Unlicense

find_package(Python COMPONENTS Interpreter Development)
set(PYTHON_EXECUTABLE ${Python_EXECUTABLE})

# * Function to generate bindings unit test.
function(neon_bindings_test TEST)
  # Correct variable according to https://cmake.org/cmake/help/latest/module/FindPython.html
  # Printing on test skip: https://stackoverflow.com/a/64776498
  add_test(
    NAME neon_bindings_test_${TEST}
    COMMAND ${Python_EXECUTABLE} -m pytest -vv -rs ${CMAKE_CURRENT_SOURCE_DIR}/test_${TEST}.py
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bindings/)
endfunction()

if(NeoN_BUILD_PYTHON_BINDINGS)
  add_test(
    NAME neon_bindings_tests
    COMMAND ${Python_EXECUTABLE} -m pytest -vv -rs ${CMAKE_CURRENT_SOURCE_DIR}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bindings)
  set_tests_properties(
    neon_bindings_tests PROPERTIES ENVIRONMENT
                                   "NEON_BINDINGS_PATH=${CMAKE_BINARY_DIR}/bindings/neon")
endif()
