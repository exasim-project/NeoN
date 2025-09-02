# SPDX-FileCopyrightText: 2023 - 2025 NeoN authors
#
# SPDX-License-Identifier: Unlicense

cmake_minimum_required(VERSION 3.22.0)

project(
  NeoN
  LANGUAGES C CXX
  VERSION 0.1.0
  DESCRIPTION "A framework for CFD software")

set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD 20)

# Add the cmake folder so the find_package command finds custom packages
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

# Set the output directories for all binaries and libraries
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
# Tell cmake we want it to automate generating export symbols for the dll
if(WIN32)
  set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
  if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc")
  endif()
endif()

# Eventhough we compile NeoN as SHARED library we need to explicitly enable PIC for all targets
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(NeoN_DEFINE_DP_SCALAR "double precision scalar" ON)
option(NeoN_DEFINE_DP_LABEL "double precision label" OFF)
option(NeoN_DEFINE_US_IDX "double precision unsigned indices" OFF)

option(NeoN_DEVEL_TOOLS "Add development tools to the build system" OFF)

# Build components
option(NeoN_BUILD_TESTS "Build the unit tests" OFF)
option(NeoN_BUILD_BENCHMARKS "Build benchmarks" OFF)
option(NeoN_BUILD_DOC "Build documentation" OFF)

# Third party dependencies
option(NeoN_WITH_ADIOS2 "Build NeoN with ADIOS2 support" OFF)
option(NeoN_WITH_SUNDIALS "Build NeoN with Sundials support [currently required]" ON)
option(NeoN_WITH_GINKGO "Enable using Ginkgo as linear algebra backend." ON)
option(NeoN_WITH_PETSC "Enable using Petsc as linear algebra backend." OFF)
option(NeoN_WITH_OMP "Prefer OMP over Kokkos Threads" OFF)
option(NeoN_WITH_THREADS "Prefer Kokkos Threads over OMP" ON)

option(NeoN_ENABLE_SANITIZE_ADDRESS "Enable address sanitizer" OFF)
option(NeoN_ENABLE_SANITIZE_LEAK "Enable leak sanitizer" OFF)
option(NeoN_ENABLE_SANITIZE_UB "Enable undefined behaviour sanitizer" OFF)
option(NeoN_ENABLE_SANITIZE_THREAD "Enable thread sanitizer" OFF)
option(NeoN_ENABLE_SANITIZE_MEMORY "Enable memory sanitizer" OFF)
option(NeoN_ENABLE_CPP_CHECK "Enable cpp check static analyzer" OFF)
option(NeoN_ENABLE_CLANG_TIDY "Enable clang tidy static analyzer" OFF)
option(NeoN_ENABLE_PCH "Enable precompiled header" OFF)
option(NeoN_ENABLE_IWYU "Enable iwyu checks" OFF)
option(NeoN_ENABLE_MPI "Enable MPI" ON)
option(NeoN_ENABLE_MPI_WITH_THREAD_SUPPORT "Enable MPI with threading support" OFF)
option(NeoN_ENABLE_WARNINGS "Treat compiler warnings as errors" OFF)
mark_as_advanced(NeoN_ENABLE_WARNINGS)
option(NeoN_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)
mark_as_advanced(NeoN_WARNINGS_AS_ERRORS)

if(WIN32)
  set(NeoN_LIB_TYPE "STATIC")
else()
  set(NeoN_LIB_TYPE "SHARED")
endif()
set(NeoN_LIB_TYPE
    ${NeoN_LIB_TYPE}
    PARENT_SCOPE)

if(NOT DEFINED CPM_USE_LOCAL_PACKAGES)
  message(STATUS "Set CPM_USE_LOCAL_PACKAGES=ON by default.")
  set(CPM_USE_LOCAL_PACKAGES ON)
endif()

if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "Setting build type to 'Release' as none was specified.")
  set(CMAKE_BUILD_TYPE
      Release
      CACHE STRING "Choose the type of build." FORCE)
endif()

if(NOT DEFINED CMAKE_EXPORT_COMPILE_COMMANDS)
  message(STATUS "Enabling generation of compilation database.")
  set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
endif()

include(cmake/CxxThirdParty.cmake)
include(cmake/PreventInSourceBuilds.cmake)
include(cmake/StandardProjectSettings.cmake)
include(cmake/CompilerWarnings.cmake)
include(cmake/Sanitizer.cmake)
include(cmake/StaticAnalyzers.cmake)

add_library(NeoN_options INTERFACE)
add_library(NeoN_warnings INTERFACE)
if(NeoN_ENABLE_WARNINGS)
  neon_set_project_warnings(NeoN_warnings ${NeoN_WARNINGS_AS_ERRORS} "" "" "")
endif()

if(NeoN_BUILD_PCH)
  target_precompile_headers(NeoN_options INTERFACE <vector> <string> <utility>)
endif()

if(NeoN_ENABLE_CPP_CHECK)
  neon_enable_cppcheck(OFF "")
endif()

if(NeoN_ENABLE_CLANG_TIDY)
  neon_enable_clang_tidy(OFF "")
endif()

add_subdirectory(include)
add_subdirectory(src)
if(NeoN_BUILD_TESTS)
  enable_testing()
  add_subdirectory(test)
endif()

if(NeoN_BUILD_BENCHMARKS)
  enable_testing()
  add_subdirectory(benchmarks)
endif()

if(NeoN_BUILD_DOC)
  include(cmake/Docs.cmake)
  neon_build_docs()
endif()

if(NeoN_DEVEL_TOOLS)
  find_program(PRE_COMMIT pre-commit)
  if(NOT PRE_COMMIT)
    message(
      FATAL_ERROR
        "The pre-commit command was not found. "
        "It is necessary if you want to commit changes to NeoN. "
        "If that is not the case, set NeoN_DEVEL_TOOLS=OFF. "
        "Otherwise install pre-commit via pipx (or pip) using:\n"
        "    pipx install pre-commit")
  endif()

  execute_process(
    COMMAND "${PRE_COMMIT}" "install"
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    RESULT_VARIABLE pre-commit-result
    OUTPUT_VARIABLE pre-commit-output
    ERROR_VARIABLE pre-commit-error)
  if(pre-commit-result)
    message(
      FATAL_ERROR "Failed to install the git hooks via pre-commit."
                  "Please check the error message:\n" "${pre-commit-output}\n${pre-commit-error}")
  endif()
  if(pre-commit-output MATCHES "^Running in migration mode with existing hooks")
    message(WARNING "An existing git hook was encountered during `pre-commit install`."
                    "The old git hook will also be executed."
                    "Consider removing it with `pre-commit install -f`")
  elseif(NOT pre-commit-output MATCHES "^pre-commit installed at")
    message(WARNING "`pre-commit install` did not exit normally."
                    "Please check the output message:\n" "${pre-commit-output}")
  endif()

  add_custom_target(
    pre-commit
    COMMENT "execute precommit"
    COMMAND bash -c "${PRE_COMMIT} run"
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    VERBATIM)
endif()

if(NeoN_ENABLE_IWYU)
  find_program(iwyu_path NAMES include-what-you-use iwyu REQUIRED)
  set(IWYU_PATH_AND_OPTIONS ${iwyu_path})
  set_property(TARGET NeoN PROPERTY CXX_INCLUDE_WHAT_YOU_USE ${IWYU_PATH_AND_OPTIONS})
endif()

include(cmake/banner.cmake)

install(
  TARGETS NeoN NeoN_public_api NeoN_options NeoN_warnings
  EXPORT NeoNTargets
  INCLUDES
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(
  EXPORT NeoNTargets
  FILE NeoNTargets.cmake
  NAMESPACE NeoN::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/NeoN)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/include/NeoN/NeoN.hpp
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/NeoN)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/NeoN/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/NeoN)

include(CMakePackageConfigHelpers)
configure_package_config_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/NeoNConfig.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/NeoNConfig.cmake"
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/NeoN)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/NeoNConfig.cmake"
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/NeoN)

# install internal ginkgo version
if(ginkgo_POPULATED)
  install(
    TARGETS ginkgo
    EXPORT GinkgoTargets
    INCLUDES
    DESTINATION include)
  install(EXPORT GinkgoTargets NAMESPACE Ginkgo::)
endif()

if(nlohmann_json_POPULATED)
  install(
    TARGETS nlohmann_json
    EXPORT nlohmannTargets
    INCLUDES
    DESTINATION include)
  install(EXPORT nlohmannTargets NAMESPACE nlohmann_json::)
endif()

# install internal cpptrace version
if(cpptrace_POPULATED)
  install(
    TARGETS cpptrace
    EXPORT CpptraceTargets
    INCLUDES
    DESTINATION include)
  install(EXPORT CpptraceTargets NAMESPACE cpptrace::)
endif()

# install internal sundials version if(sundials_POPULATED) install( TARGETS cvode nvec arkode
# sundials_core_shared EXPORT SundialsTargets INCLUDES DESTINATION include) install(EXPORT
# SundialsTargets NAMESPACE SUNDIALS::) endif()

if(NOT kokkos_POPULATED)
  export(EXPORT NeoNTargets FILE "${CMAKE_CURRENT_BINARY_DIR}/NeoNTargets.cmake")
else()
  message(WARNING "Unable to create NeoNTargets.cmake with bundled Kokkos")
endif()
