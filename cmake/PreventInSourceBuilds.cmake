# SPDX-License-Identifier: Unlicense
# SPDX-FileCopyrightText: 2023 Jason Turner
# SPDX-FileCopyrightText: 2023 NeoN authors

# cmake-format: off
##############################################################################
# This function will prevent in-source builds                                #
# from here                                                                  #
# https://github.com/cpp-best-practices/cmake_template                       #
##############################################################################
# cmake-format: on

function(myproject_assure_out_of_source_builds)
  # make sure the user doesn't play dirty with symlinks
  get_filename_component(srcdir "${CMAKE_SOURCE_DIR}" REALPATH)
  get_filename_component(bindir "${CMAKE_BINARY_DIR}" REALPATH)

  # disallow in-source builds
  if("${srcdir}" STREQUAL "${bindir}")
    message("######################################################")
    message("Warning: in-source builds are disabled")
    message("Please create a separate build directory and run cmake from there")
    message("######################################################")
    message(FATAL_ERROR "Quitting configuration")
  endif()
endfunction()

myproject_assure_out_of_source_builds()
