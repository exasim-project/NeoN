# SPDX-FileCopyrightText: 2023 - 2025 NeoN authors
#
# SPDX-License-Identifier: Unlicense

find_package(Doxygen REQUIRED)
find_package(Sphinx REQUIRED)

# Macro: NeoN_BUILD_DOCS build documentation with doxygen and sphinx
macro(NeoN_BUILD_DOCS)

  set(SPHINX_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/doc)
  set(SPHINX_BUILD ${CMAKE_CURRENT_BINARY_DIR}/docs_build)

  add_custom_target(
    sphinx ALL
    COMMAND ${SPHINX_EXECUTABLE} -b html ${SPHINX_SOURCE} ${SPHINX_BUILD}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Generating documentation with Sphinx")

  # add one target to build both doxygen and sphinx
  add_custom_target(doc COMMENT "build doc with doxygen and sphinx")
  add_dependencies(doc sphinx)

endmacro()
