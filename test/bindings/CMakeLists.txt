# SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
#
# SPDX-License-Identifier: Unlicense

if(NeoN_BUILD_PYTHON_BINDINGS)
  add_test(
    NAME neon_bindings_tests
    COMMAND ${Python_EXECUTABLE} -m pytest -vv -rs ${CMAKE_CURRENT_SOURCE_DIR}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bindings)
  set_tests_properties(
    neon_bindings_tests PROPERTIES ENVIRONMENT
                                   "NEON_BINDINGS_PATH=${CMAKE_BINARY_DIR}/bindings/neon")
endif()
