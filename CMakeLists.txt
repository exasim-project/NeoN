# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: Unlicense

cmake_minimum_required(VERSION 3.22.0)

# Re-run configuration when the version source of truth changes, otherwise the CMake project version
# and generated neon.__version__ would go stale.
set_property(
  DIRECTORY
  APPEND
  PROPERTY CMAKE_CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/pyproject.toml")

file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/pyproject.toml" NeoN_PACKAGE_VERSION_LINE
     REGEX "^version = \"[^\"]+\"")
if(NOT NeoN_PACKAGE_VERSION_LINE)
  message(FATAL_ERROR "Could not read project.version from pyproject.toml")
endif()
string(REGEX REPLACE "^version = \"([^\"]+)\".*" "\\1" NeoN_PACKAGE_VERSION
                     "${NeoN_PACKAGE_VERSION_LINE}")
string(REGEX MATCH "^[0-9]+(\\.[0-9]+)*" NeoN_CMAKE_VERSION "${NeoN_PACKAGE_VERSION}")
if(NOT NeoN_CMAKE_VERSION)
  message(FATAL_ERROR "Package version '${NeoN_PACKAGE_VERSION}' has no numeric CMake version")
endif()

project(
  NeoN
  LANGUAGES C CXX
  VERSION ${NeoN_CMAKE_VERSION}
  DESCRIPTION "A framework for CFD software")

set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD 20)
if(APPLE)
  set(CMAKE_MACOSX_RPATH 1)
endif()
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")

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
option(NeoN_BUILD_PYTHON_BINDINGS "Build python bindings" OFF)
# Link _neon against a shared libnanobind instead of the default static one. Only needed when the
# bindings must exchange nanobind types with another extension module built against the same
# libnanobind (e.g. pybFoam, so create_adapter_run_time accepts a Foam::Time). Off by default: a
# shared runtime has to be installed alongside _neon, and wheel repair tools (auditwheel, delocate,
# delvewheel) vendor it under a mangled SONAME, which defeats the sharing it exists for.
option(NeoN_NANOBIND_SHARED "Link the python bindings against a shared libnanobind" OFF)
# Take nanobind from the active Python environment instead of fetching NeoN_NANOBIND_VERSION via
# CPM. Useful together with NeoN_NANOBIND_SHARED: pybFoam builds against that same copy, so both
# modules share one libnanobind ABI by construction rather than by a version pin.
option(NeoN_EXTERNAL_NANOBIND "Use the nanobind from the active Python environment" OFF)
option(NeoN_INSTALL_CMAKE_PACKAGE "Install CMake package config files and exported targets" ON)

# Third party dependencies
option(NeoN_WITH_KOKKOS "Enable using Kokkos as platform portability layer." ON)
option(NeoN_WITH_UMPIRE "Enable using Umpire as memory manager [Experimental]." ON)
option(NeoN_WITH_ADIOS2 "Build NeoN with ADIOS2 support" OFF)
option(NeoN_WITH_SUNDIALS "Build NeoN with Sundials support." OFF)
option(NeoN_WITH_GINKGO "Enable using Ginkgo as linear algebra backend." ON)
option(NeoN_WITH_PETSC "Enable using Petsc as linear algebra backend." OFF)
option(NeoN_WITH_OMP "Prefer OMP over Kokkos Threads" OFF)
option(NeoN_WITH_SPDLOG "Use spdlog for logging" OFF)
option(NeoN_WITH_THREADS "Prefer Kokkos Threads over OMP" ON)
option(NeoN_WITH_MPI "Enable MPI" ON)

option(NeoN_ENABLE_SANITIZE_ADDRESS "Enable address sanitizer" OFF)
option(NeoN_ENABLE_SANITIZE_LEAK "Enable leak sanitizer" OFF)
option(NeoN_ENABLE_SANITIZE_UB "Enable undefined behaviour sanitizer" OFF)
option(NeoN_ENABLE_SANITIZE_THREAD "Enable thread sanitizer" OFF)
option(NeoN_ENABLE_SANITIZE_MEMORY "Enable memory sanitizer" OFF)
option(NeoN_ENABLE_CPP_CHECK "Enable cpp check static analyzer" OFF)
option(NeoN_ENABLE_CLANG_TIDY "Enable clang tidy static analyzer" OFF)
option(NeoN_ENABLE_PCH "Enable precompiled header" OFF)
option(NeoN_ENABLE_IWYU "Enable iwyu checks" OFF)
# NeoN_ENABLE_MPI_WITH_THREAD_SUPPORT is defined in cmake/DetectMpiThreadSupport.cmake (included via
# cmake/CxxThirdParty.cmake when NeoN_WITH_MPI is ON) so that its default value reflects the runtime
# capability of the local MPI installation.
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

# Detect if building with scikit-build-core
if(DEFINED SKBUILD)
  message(STATUS "Building with scikit-build-core")
  set(NeoN_BUILD_PYTHON_BINDINGS
      ON
      CACHE BOOL "" FORCE)
  set(BUILD_SHARED_LIBS
      ON
      CACHE BOOL "" FORCE)
  # Configure RPATH for Python extension modules
  set(CMAKE_SKIP_BUILD_RPATH FALSE)
  set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
  set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
  # Set RPATH so _neon.so can find libNeoN.so at ../lib/
  set(CMAKE_INSTALL_RPATH "$ORIGIN/../lib")
endif()

include(cmake/CxxThirdParty.cmake)

# Umpire (and its bundled camp) are built and installed by their own BLT-based CMake, which sets
# each target's INSTALL_RPATH to ${CMAKE_INSTALL_PREFIX}/lib. Under scikit-build that prefix is the
# wheel-staging temp directory, which no longer exists after install, so at import time libumpire.so
# cannot find its direct dependency libcamp.so ("libcamp.so: cannot open shared object file"). A
# per-target override is the only thing that beats BLT's own RPATH — the global CMAKE_INSTALL_RPATH
# set above does not reach these CPM targets (it does reach NeoN's own libraries). The
# loader-relative entries keep the install relocatable and reach camp whether it sits beside umpire
# or is installed to the package-root lib/. On macOS BLT also points CMAKE_INSTALL_NAME_DIR at that
# same stale prefix, so overriding the RPATH alone would change nothing: the install name has to
# become @rpath-relative before any RPATH entry is consulted.
if(DEFINED SKBUILD)
  if(APPLE)
    set(_neon_cpm_dep_rpath "@loader_path;@loader_path/../lib;@loader_path/../../lib")
  else()
    set(_neon_cpm_dep_rpath "$ORIGIN;$ORIGIN/../lib;$ORIGIN/../../lib")
  endif()
  # umpire_device only exists for CUDA/HIP builds with UMPIRE_ENABLE_DEVICE_ALLOCATOR, and
  # INSTALL_NAME_DIR is ignored on non-Apple platforms, so both are safe to set unconditionally.
  foreach(_neon_cpm_dep umpire umpire_device camp)
    if(TARGET ${_neon_cpm_dep})
      set_target_properties(
        ${_neon_cpm_dep}
        PROPERTIES INSTALL_RPATH "${_neon_cpm_dep_rpath}"
                   INSTALL_RPATH_USE_LINK_PATH FALSE
                   INSTALL_NAME_DIR "@rpath")
    endif()
  endforeach()
endif()

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

# Promote nvcc's "constexpr __host__ function called from a __host__ __device__ function"
# diagnostics (20013 / 20015) to hard errors on CUDA builds. With Kokkos_ENABLE_CUDA_CONSTEXPR=ON
# (forced in cmake/AutoEnableDevice.cmake) these never fire. If that flag is ever disabled, nvcc
# would otherwise SILENTLY miscompile device-side element access through View/std::span operator[]
# (writes vanish on the GPU, results are garbage). Failing the build is far safer than shipping
# wrong numerics. -Xcudafe needs each number in its own flag; SHELL: keeps the repeated tokens from
# being de-duplicated by CMake.
if(Kokkos_ENABLE_CUDA)
  set(_neon_cuda_diag_errors "SHELL:-Xcudafe --diag_error=20013"
                             "SHELL:-Xcudafe --diag_error=20015")
  # Always apply to CUDA-language translation units (nvcc compiles .cu files directly).
  target_compile_options(NeoN_warnings
                         INTERFACE "$<$<COMPILE_LANGUAGE:CUDA>:${_neon_cuda_diag_errors}>")
  # Apply to CXX only when nvcc_wrapper is the CXX compiler; plain g++/clang++ reject -Xcudafe.
  if(CMAKE_CXX_COMPILER MATCHES "nvcc_wrapper")
    target_compile_options(NeoN_warnings
                           INTERFACE "$<$<COMPILE_LANGUAGE:CXX>:${_neon_cuda_diag_errors}>")
  endif()
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
  set(IWYU_PATH_AND_OPTIONS ${iwyu_path} -Xiwyu --mapping_file=${PROJECT_SOURCE_DIR}/cmake/iwyu.imp)
  set_property(TARGET NeoN PROPERTY CXX_INCLUDE_WHAT_YOU_USE ${IWYU_PATH_AND_OPTIONS})
  # IWYU 8.21 (clang-19) segfaults on volume/processor.cpp due to the template complexity of the
  # processor boundary hierarchy (scalar/Vec3/Tensor instantiations); a segfault cannot be fixed by
  # cleaning includes, so the file is excluded. surface/processor.cpp lints fine, so only the volume
  # variant is excluded. The property must be set in the NeoN target's directory scope
  # (TARGET_DIRECTORY) with an absolute path, or it silently fails to attach and IWYU runs anyway.
  set_source_files_properties(
    ${PROJECT_SOURCE_DIR}/src/finiteVolume/cellCentred/boundary/volume/processor.cpp
    TARGET_DIRECTORY NeoN PROPERTIES SKIP_LINTING ON)
endif()

include(cmake/banner.cmake)

if(NeoN_INSTALL_CMAKE_PACKAGE)
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

  # umpire installs its own targets when fetched via CPM

  # install internal cpptrace version
  if(cpptrace_POPULATED)
    install(
      TARGETS cpptrace
      EXPORT CpptraceTargets
      INCLUDES
      DESTINATION include)
    install(EXPORT CpptraceTargets NAMESPACE cpptrace::)
  endif()

  # install internal spdlog version TODO why does spdlog_POPULATED not work?
  if(spdlog_ADDED)
    install(
      TARGETS spdlog_header_only
      EXPORT SpdlogTargets
      INCLUDES
      DESTINATION include)
    if(fmt_ADDED)
      install(TARGETS fmt-header-only EXPORT SpdlogTargets)
    endif()
    install(
      EXPORT SpdlogTargets
      DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/spdlog
      NAMESPACE spdlog::)
  endif()

  if(fmt_ADDED)
    install(TARGETS fmt-header-only EXPORT NeoNTargets)
    install(DIRECTORY "${fmt_SOURCE_DIR}/include/" DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
  endif()

  # install internal sundials version if(sundials_POPULATED) install( TARGETS cvode nvec arkode
  # sundials_core_shared EXPORT SundialsTargets INCLUDES DESTINATION include) install(EXPORT
  # SundialsTargets NAMESPACE SUNDIALS::) endif()

  if(NOT kokkos_POPULATED AND NOT umpire_ADDED)
    export(EXPORT NeoNTargets FILE "${CMAKE_CURRENT_BINARY_DIR}/NeoNTargets.cmake")
  else()
    message(WARNING "Unable to create NeoNTargets.cmake with bundled Kokkos or Umpire")
  endif()
endif()
