# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

#[=======================================================================[.rst:
FindJulia
---------

Finds the Julia C API (``julia.h`` and ``libjulia``) of a Julia installation.

The installation prefix is taken from, in this order:

1. the cache variable ``JULIA_DIR`` (``cmake -DJULIA_DIR=/path/to/julia-1.11.x``),
2. ``Julia_ROOT`` (variable or environment, the usual ``find_package`` convention),
3. the environment variables ``JULIA_DIR`` and ``JULIA_HOME``,
4. the prefix of a ``julia`` executable on ``PATH`` (resolved through symlinks),
5. as a last resort, ``julia -e 'print(Sys.BINDIR)'`` — only for launcher shims such as
   juliaup's, whose location says nothing about the installation.

Imported target::

  Julia::Julia    julia.h include directory and libjulia

Result variables::

  Julia_FOUND, Julia_VERSION, Julia_INCLUDE_DIR, Julia_LIBRARY, Julia_PREFIX

No rpath is added: CMake records the directory of the linked library in the build rpath, and
installs pick it up through ``CMAKE_INSTALL_RPATH_USE_LINK_PATH``.
#]=======================================================================]

set(JULIA_DIR
    ""
    CACHE PATH "Julia installation prefix (contains include/julia and lib)")

# find_path/find_library cache their results; forget them when JULIA_DIR changes so that the
# variable stays authoritative across reconfigures
if(NOT "${JULIA_DIR}" STREQUAL "${_Julia_JULIA_DIR_LAST}")
  unset(Julia_INCLUDE_DIR CACHE)
  unset(Julia_LIBRARY CACHE)
endif()
set(_Julia_JULIA_DIR_LAST
    "${JULIA_DIR}"
    CACHE INTERNAL "JULIA_DIR of the last configure")

set(_julia_hints "${JULIA_DIR}" "${Julia_ROOT}" "$ENV{Julia_ROOT}" "$ENV{JULIA_DIR}" "$ENV{JULIA_HOME}")
list(FILTER _julia_hints EXCLUDE REGEX "^$")

# an explicit prefix is searched exclusively; the fallbacks below also search the default paths
set(_julia_search_opts "")
if(_julia_hints)
  set(_julia_search_opts NO_DEFAULT_PATH)
endif()

if(NOT _julia_hints)
  find_program(Julia_EXECUTABLE NAMES julia)
  if(Julia_EXECUTABLE)
    get_filename_component(_julia_real "${Julia_EXECUTABLE}" REALPATH)
    get_filename_component(_julia_bin "${_julia_real}" DIRECTORY)
    get_filename_component(_julia_prefix "${_julia_bin}" DIRECTORY)
    if(EXISTS "${_julia_prefix}/include/julia/julia.h")
      list(APPEND _julia_hints "${_julia_prefix}")
    else()
      # a launcher shim (e.g. juliaup): ask the launcher where the real installation is
      execute_process(
        COMMAND "${Julia_EXECUTABLE}" --startup-file=no -e "print(Sys.BINDIR)"
        OUTPUT_VARIABLE _julia_sys_bindir
        RESULT_VARIABLE _julia_res
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET)
      if(_julia_res EQUAL 0 AND _julia_sys_bindir)
        get_filename_component(_julia_prefix "${_julia_sys_bindir}" DIRECTORY)
        list(APPEND _julia_hints "${_julia_prefix}")
        message(STATUS "FindJulia: resolved launcher ${Julia_EXECUTABLE} to ${_julia_prefix}; "
                       "set JULIA_DIR to skip this step")
      endif()
    endif()
  endif()
endif()

find_path(
  Julia_INCLUDE_DIR
  NAMES julia.h
  HINTS ${_julia_hints}
  PATH_SUFFIXES include/julia ${_julia_search_opts})

find_library(
  Julia_LIBRARY
  NAMES julia
  HINTS ${_julia_hints}
  PATH_SUFFIXES lib lib64 ${_julia_search_opts})

if(Julia_INCLUDE_DIR AND EXISTS "${Julia_INCLUDE_DIR}/julia_version.h")
  file(STRINGS "${Julia_INCLUDE_DIR}/julia_version.h" _julia_version_line
       REGEX "#define JULIA_VERSION_STRING")
  string(REGEX REPLACE ".*\"([^\"]+)\".*" "\\1" Julia_VERSION "${_julia_version_line}")
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  Julia
  REQUIRED_VARS Julia_LIBRARY Julia_INCLUDE_DIR
  VERSION_VAR Julia_VERSION
  REASON_FAILURE_MESSAGE "set JULIA_DIR to the Julia installation prefix")

if(Julia_FOUND)
  get_filename_component(Julia_PREFIX "${Julia_INCLUDE_DIR}/../.." ABSOLUTE)
  if(NOT TARGET Julia::Julia)
    add_library(Julia::Julia UNKNOWN IMPORTED)
    set_target_properties(
      Julia::Julia PROPERTIES IMPORTED_LOCATION "${Julia_LIBRARY}"
                              INTERFACE_INCLUDE_DIRECTORIES "${Julia_INCLUDE_DIR}")
  endif()
endif()

mark_as_advanced(Julia_INCLUDE_DIR Julia_LIBRARY Julia_EXECUTABLE)
unset(_julia_hints)
unset(_julia_search_opts)
