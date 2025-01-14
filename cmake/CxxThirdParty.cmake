# SPDX-License-Identifier: Unlicense
# SPDX-FileCopyrightText: 2023 NeoFOAM authors

set(NEOFOAM_KOKKOS_CHECKOUT_VERSION
    "4.3.00"
    CACHE STRING "Use specific version of Kokkos")
mark_as_advanced(NEOFOAM_KOKKOS_CHECKOUT_VERSION)
if(NEOFOAM_ENABLE_MPI_SUPPORT)
  if(WIN32)
    message(FATAL_ERROR "NEOFOAM_ENABLE_MPI_SUPPORT not supported on Windows")
  endif()
  find_package(MPI 3.1 REQUIRED)
endif()

find_package(Kokkos ${NEOFOAM_KOKKOS_CHECKOUT_VERSION} QUIET)

if(NOT ${Kokkos_FOUND})
  include(FetchContent)
  include(cmake/AutoEnableDevice.cmake)

  FetchContent_Declare(
    Kokkos
    SYSTEM QUITE
    GIT_SHALLOW ON
    GIT_REPOSITORY "https://github.com/kokkos/kokkos.git"
    GIT_TAG ${NEOFOAM_KOKKOS_CHECKOUT_VERSION})

  FetchContent_MakeAvailable(Kokkos)
else()
  message(STATUS "Found Kokkos ${NEOFOAM_KOKKOS_CHECKOUT_VERSION}")
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

cpmaddpackage(
  NAME
  nlohmann_json
  VERSION
  3.11.3
  URL
  https://github.com/nlohmann/json/releases/download/v3.11.3/include.zip
  SYSTEM)

if(${NEOFOAM_WITH_PETSC})
  include(ExternalProject)
  if(CMAKE_CUDA_COMPILER)
    set(PETSC_ENABLE_CUDA 1)
  else()
    set(PETSC_ENABLE_CUDA 0)
  endif()

  cmake_path(GET CMAKE_CUDA_COMPILER PARENT_PATH CUDA_PARTENT_PATH)
  cmake_path(GET CUDA_PARTENT_PATH PARENT_PATH CUDA_PARENT_PARENT_PATH)
  find_package(MPI)

  if(MPI_FOUND)
    set(PETSC_ENABLE_MPI 1)
    set(PETSC_CXX_COMPILER ${MPI_CXX_COMPILER})
    set(PETSC_C_COMPILER ${MPI_C_COMPILER})
  else()
    set(PETSC_ENABLE_MPI 0)
    set(PETSC_CXX_COMPILER ${CMAKE_CXX_COMPILER})
    set(PETSC_C_COMPILER ${CMAKE_CXX_COMPILER})
  endif()

  ExternalProject_Add(
    petsc
    GIT_SHALLOW ON
    GIT_REPOSITORY "https://gitlab.com/petsc/petsc.git"
    GIT_TAG v3.22.2
    PREFIX ${CMAKE_BINARY_DIR}/petsc
    BUILD_IN_SOURCE YES
    CONFIGURE_COMMAND
      ./configure --with-64-bit-indices=0 --with-precision=double --with-cuda=${PETSC_ENABLE_CUDA}
      --with-cuda-dir=${CUDA_PARENT_PARENT_PATH} --with-mpi=${PETSC_ENABLE_MPI} --with-fc=0 --force
      --with-32bits-pci-domain=1 --with-cc=${PETSC_C_COMPILER} --with-cxx=${PETSC_CXX_COMPILER}
      --with-debugging=no --prefix=${CMAKE_BINARY_DIR}/petsc/opt/petsc
    BUILD_COMMAND make PETSC_DIR=${CMAKE_BINARY_DIR}/petsc/src/petsc PETSC_ARCH=arch-linux-c-opt all
    INSTALL_COMMAND make PETSC_DIR=${CMAKE_BINARY_DIR}/petsc/src/petsc PETSC_ARCH=arch-linux-c-opt
                    install
    BUILD_BYPRODUCTS ${CMAKE_BINARY_DIR}/petsc/opt/petsc/lib/libpetsc.so)

endif()

if(${NEOFOAM_WITH_SUNDIALS})

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
    set(SUNDIALS_CUDA_OPTIONS "ENABLE_CUDA ON" "SUNDIALS_BUILD_KOKKOS ON")
  else()
    set(SUNDIALS_CUDA_OPTIONS "ENABLE_CUDA OFF" "SUNDIALS_BUILD_KOKKOS ON")
  endif()

  cpmaddpackage(
    NAME
    sundials
    GITHUB_REPOSITORY
    LLNL/sundials
    VERSION
    7.1.1
    OPTIONS
    ${SUNDIALS_OPTIONS}
    ${SUNDIALS_CUDA_OPTIONS}
    SYSTEM)
endif()

cpmaddpackage(
  NAME
  spdlog
  URL
  https://github.com/gabime/spdlog/archive/refs/tags/v1.13.0.zip
  VERSION
  1.13.0
  SYSTEM)

cpmaddpackage(
  NAME
  cxxopts
  URL
  https://github.com/jarro2783/cxxopts/archive/refs/tags/v3.2.0.zip
  VERSION
  3.2.0
  SYSTEM)

if(NEOFOAM_BUILD_TESTS OR NEOFOAM_BUILD_BENCHMARKS)
  cpmaddpackage(
    NAME
    Catch2
    URL
    https://github.com/catchorg/Catch2/archive/refs/tags/v3.4.0.zip
    VERSION
    3.4.0
    SYSTEM)
endif()
