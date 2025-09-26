# SPDX-FileCopyrightText: 2023 - 2025 NeoN authors
#
# SPDX-License-Identifier: Unlicense

# set(FETCHCONTENT_BASE_DIR ${CMAKE_BINARY_DIR}/cmake_packages)

set(NeoN_KOKKOS_CHECKOUT_VERSION
    "4.3.00"
    CACHE STRING "Use specific version of Kokkos")
mark_as_advanced(NeoN_KOKKOS_CHECKOUT_VERSION)
if(NeoN_ENABLE_MPI_SUPPORT)
  if(WIN32)
    message(FATAL_ERROR "NeoN_ENABLE_MPI_SUPPORT not supported on Windows")
  endif()
  find_package(MPI 3.1 REQUIRED)
endif()

if(${NeoN_WITH_PETSC})
  find_package(MPI) # make it REQUIRED, if you want
  message("${MPI_CXX_COMPILER}")
  message("${MPI_C_COMPILER}")
  message("${MPI_FOUND}")
  if(MPI_FOUND)
    set(PETSC_CXX_COMPILER ${MPI_CXX_COMPILER})
    set(PETSC_C_COMPILER ${MPI_C_COMPILER})
  else()
    set(PETSC_CXX_COMPILER ${CMAKE_CXX_COMPILER})
    set(PETSC_C_COMPILER ${CMAKE_CXX_COMPILER})
  endif()
  find_package(PkgConfig REQUIRED)
  pkg_search_module(PETSc REQUIRED IMPORTED_TARGET PETSc)
  set(Kokkos_ROOT ${PETSc_PREFIX})
  set(CMAKE_CUDA_STANDARD 17)
  include(cmake/AutoEnableDevice.cmake)

  set(Kokkos_ROOT_DIR ${PETSc_PREFIX})
  message("${PETSc_PREFIX}")
endif()

find_package(Kokkos ${NeoN_KOKKOS_CHECKOUT_VERSION} QUIET)

if(NOT Kokkos_FOUND)
  include(FetchContent)
  include(cmake/AutoEnableDevice.cmake)

  FetchContent_Declare(
    Kokkos
    SYSTEM QUITE
    GIT_SHALLOW ON
    GIT_REPOSITORY "https://github.com/kokkos/kokkos.git"
    GIT_TAG ${NeoN_KOKKOS_CHECKOUT_VERSION})

  FetchContent_MakeAvailable(Kokkos)
else()
  message(STATUS "Found Kokkos ${NeoN_KOKKOS_CHECKOUT_VERSION}")
endif()

include(cmake/CPM.cmake)

cpmaddpackage(
  NAME
  cpptrace
  URL
  https://github.com/jeremy-rifkin/cpptrace/archive/refs/tags/v0.7.3.zip
  VERSION
  0.7.3
  SYSTEM)

if(${NeoN_WITH_ADIOS2})

  set(ADIOS2_KOKKOS_PATCH git apply ${CMAKE_CURRENT_SOURCE_DIR}/cmake/patches/adios2_kokkos.patch)

  set(ADIOS2_OPTIONS
      "BUILD_TYPE ${CMAKE_BUILD_TYPE}"
      "ADIOS2_USE_Kokkos ON"
      "Kokkos_DIR ${Kokkos_BINARY_DIR}"
      "ADIOS2_USE_Fortran OFF"
      "ADIOS2_USE_Python OFF"
      "ADIOS2_USE_MHS OFF"
      "ADIOS2_USE_SST OFF"
      "ADIOS2_BUILD_EXAMPLES OFF"
      "BUILD_TESTING OFF"
      "ADIOS2_USE_Profiling OFF")

  if(WIN32)
    list(APPEND ADIOS2_OPTIONS "BUILD_STATIC_LIBS ON")
    list(APPEND ADIOS2_OPTIONS "BUILD_SHARED_LIBS OFF")
  else()
    list(APPEND ADIOS2_OPTIONS "BUILD_STATIC_LIBS OFF")
    list(APPEND ADIOS2_OPTIONS "BUILD_SHARED_LIBS ON")
  endif()

  # Checking for patched and cached ADIOS2 will become obsolete after this issue in CPM has been
  # fixed: https://github.com/cpm-cmake/CPM.cmake/issues/618
  if(NOT ADIOS2_PATCHED)
    set(ADIOS2_PATCHED
        TRUE
        CACHE INTERNAL "Whether ADIOS2 has been fetched and patched in CPM cache.")
    cpmaddpackage(
      NAME
      adios2
      GITHUB_REPOSITORY
      ornladios/ADIOS2
      PATCH_COMMAND
      ${ADIOS2_KOKKOS_PATCH}
      VERSION
      2.10.2
      OPTIONS
      ${ADIOS2_OPTIONS}
      ${ADIOS2_CUDA_OPTIONS}
      SYSTEM)
  else()
    cpmaddpackage(
      NAME
      adios2
      GITHUB_REPOSITORY
      ornladios/ADIOS2
      VERSION
      2.10.2
      OPTIONS
      ${ADIOS2_OPTIONS}
      ${ADIOS2_CUDA_OPTIONS}
      SYSTEM)
  endif()
endif()

if(${NeoN_WITH_SUNDIALS})

  set(SUNDIALS_OPTIONS
      "BUILD_TESTING OFF"
      "EXAMPLES_INSTALL OFF"
      "BUILD_ARKODE ON"
      "BUILD_CVODE OFF"
      "BUILD_CVODES OFF"
      "BUILD_IDA OFF"
      "BUILD_IDAS OFF"
      "BUILD_KINSOL OFF"
      "BUILD_CPODES OFF")

  if(WIN32)
    list(APPEND SUNDIALS_OPTIONS "BUILD_STATIC_LIBS ON")
    list(APPEND SUNDIALS_OPTIONS "BUILD_SHARED_LIBS OFF")
  else()
    list(APPEND SUNDIALS_OPTIONS "BUILD_STATIC_LIBS OFF")
    list(APPEND SUNDIALS_OPTIONS "BUILD_SHARED_LIBS ON")
  endif()

  if(Kokkos_ENABLE_CUDA)
    set(CMAKE_CUDA_STANDARD 20)
    set(SUNDIALS_CUDA_OPTIONS "ENABLE_CUDA ON" "SUNDIALS_BUILD_KOKKOS ON")
  else()
    set(SUNDIALS_CUDA_OPTIONS "ENABLE_CUDA OFF" "SUNDIALS_BUILD_KOKKOS ON")
  endif()

  cpmaddpackage(
    NAME
    SUNDIALS
    GITHUB_REPOSITORY
    LLNL/sundials
    VERSION
    7.3.0
    SYSTEM
    YES
    OPTIONS
    ${SUNDIALS_OPTIONS}
    ${SUNDIALS_CUDA_OPTIONS})
endif()

# currently not used cpmaddpackage( NAME spdlog URL
# https://github.com/gabime/spdlog/archive/refs/tags/v1.13.0.zip VERSION 1.13.0 SYSTEM)

# currently not used cpmaddpackage( NAME cxxopts URL
# https://github.com/jarro2783/cxxopts/archive/refs/tags/v3.2.0.zip VERSION 3.2.0 SYSTEM)

if(${NeoN_WITH_GINKGO})
  # nlohmann json is only needed in combination with ginkgo. Getting the package via cpm+github
  # takes ages. After pulling only the .zip no nlohmann_json::nlohman_json target is found. Thus we
  # always prepare system install version of nlohmann_json if present.
  find_package(nlohmann_json 3.11.3 QUIET)
  if(NOT nlohmann_json_FOUND)
    cpmaddpackage(
      NAME
      nlohmann_json
      VERSION
      3.11.3
      GITHUB_REPOSITORY
      nlohmann/json
      SYSTEM
      YES
      OPTIONS
      "JSON_Install ON")
  endif()

  cpmaddpackage(
    NAME
    Ginkgo
    VERSION
    1.10.0
    GITHUB_REPOSITORY
    ginkgo-project/ginkgo
    GIT_TAG
    0b50e390e15d36fe5432e6584049fd3f880584f1
    SYSTEM
    YES
    OPTIONS
    "GINKGO_BUILD_TESTS OFF"
    "GINKGO_BUILD_BENCHMARKS OFF"
    "GINKGO_BUILD_EXAMPLES OFF"
    "GINKGO_BUILD_OMP ${NeoN_WITH_OMP}"
    "GINKGO_ENABLE_HALF OFF"
    "GINKGO_BUILD_MPI OFF"
    "GINKGO_BUILD_CUDA ${Kokkos_ENABLE_CUDA}"
    "GINKGO_BUILD_HIP ${Kokkos_ENABLE_HIP}")
endif()

if(NeoN_BUILD_TESTS OR NeoN_BUILD_BENCHMARKS)
  cpmaddpackage(
    NAME
    Catch2
    URL
    https://github.com/catchorg/Catch2/archive/refs/tags/v3.4.0.zip
    VERSION
    3.4.0
    SYSTEM
    YES)
endif()
