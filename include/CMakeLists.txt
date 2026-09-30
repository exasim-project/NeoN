# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: Unlicense

# NeoN's usage requirements come in two layers (see doc/splitCompilation.rst):
#
# * NeoN_config_api: include directories, every compile definition that affects class layout or API
#   (NF_WITH_MPI_SUPPORT, NeoN_DP_SCALAR, ...) and the host-only header dependencies (fmt, spdlog,
#   cpptrace, MPI, Umpire, nlohmann_json). It deliberately carries no Kokkos, so host-only
#   translation units, possibly built by a different compiler at a newer C++ standard, see exactly
#   the same class definitions as the device half.
# * NeoN_public_api: NeoN_config_api plus Kokkos and the device-side libraries (Ginkgo, SUNDIALS,
#   PETSc). This is what consumers of NeoN link against, unchanged from before.
add_library(NeoN_config_api INTERFACE)
add_library(NeoN::NeoN_config_api ALIAS NeoN_config_api)
add_library(NeoN_public_api INTERFACE) # dummy target
add_library(NeoN::NeoN_public_api ALIAS NeoN_public_api) # dummy target
target_link_libraries(NeoN_public_api INTERFACE NeoN_config_api)

# set_property( TARGET NeoN_public_api APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES
# "${CMAKE_CURRENT_BINARY_DIR}")

target_include_directories(
  NeoN_config_api INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
                            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}> $<INSTALL_INTERFACE:.>)
if(NeoN_WITH_KOKKOS)
  target_compile_definitions(NeoN_config_api INTERFACE NN_WITH_KOKKOS=1)
  target_link_libraries(NeoN_public_api INTERFACE Kokkos::kokkos)
  if(Kokkos_ENABLE_CUDA)
    target_compile_definitions(NeoN_config_api INTERFACE NEON_ENABLE_CUDA=1)
  endif()
  if(Kokkos_ENABLE_HIP)
    target_compile_definitions(NeoN_config_api INTERFACE NEON_ENABLE_HIP=1)
  endif()
  if(Kokkos_ENABLE_CYCLE)
    target_compile_definitions(NeoN_config_api INTERFACE NEON_ENABLE_CYCLE=1)
  endif()
  if(Kokkos_ENABLE_OPENMP)
    target_compile_definitions(NeoN_config_api INTERFACE NEON_ENABLE_OPENMP=1)
  endif()
  if(Kokkos_ENABLE_THREADS)
    target_compile_definitions(NeoN_config_api INTERFACE NEON_ENABLE_THREADS=1)
  endif()
else()
  target_compile_definitions(NeoN_config_api INTERFACE NN_WITH_KOKKOS=0)
endif()

if(NeoN_WITH_UMPIRE)
  target_link_libraries(NeoN_config_api INTERFACE umpire)
  target_compile_definitions(NeoN_config_api INTERFACE NF_WITH_UMPIRE=1)
else()
  target_compile_definitions(NeoN_config_api INTERFACE NF_WITH_UMPIRE=0)
endif()

if(NeoN_WITH_GINKGO)
  target_link_libraries(NeoN_config_api INTERFACE nlohmann_json::nlohmann_json)
  target_link_libraries(NeoN_public_api INTERFACE Ginkgo::ginkgo)
  target_compile_definitions(NeoN_config_api INTERFACE NF_WITH_GINKGO=1)
else()
  target_compile_definitions(NeoN_config_api INTERFACE NF_WITH_GINKGO=0)
endif()

if(NeoN_WITH_SUNDIALS)
  target_compile_definitions(NeoN_config_api INTERFACE NN_WITH_SUNDIALS=1)
  target_link_libraries(NeoN_public_api INTERFACE SUNDIALS::arkode SUNDIALS::nvecserial
                                                  SUNDIALS::core)
else()
  target_compile_definitions(NeoN_config_api INTERFACE NN_WITH_SUNDIALS=0)
endif()

if(NeoN_WITH_SPDLOG)
  target_link_libraries(NeoN_config_api INTERFACE spdlog::spdlog_header_only)
  target_compile_definitions(NeoN_config_api INTERFACE NF_WITH_SPDLOG=1)
else()
  target_compile_definitions(NeoN_config_api INTERFACE NF_WITH_SPDLOG=0)
endif()

target_link_libraries(NeoN_config_api INTERFACE cpptrace::cpptrace)
target_link_libraries(NeoN_config_api INTERFACE fmt::fmt-header-only)

if(NeoN_WITH_PETSC)
  add_definitions(${PETSc_DEFINITIONS})
  target_link_libraries(NeoN_public_api INTERFACE ${PETSc_LIBRARIES})
  target_compile_definitions(NeoN_config_api INTERFACE NF_WITH_PETSC=1)
else()
  target_compile_definitions(NeoN_config_api INTERFACE NF_WITH_PETSC=0)
endif()

if(${CMAKE_BUILD_TYPE} MATCHES Debug)
  target_compile_definitions(NeoN_config_api INTERFACE NF_DEBUG)
endif()

if(${CMAKE_BUILD_TYPE} MATCHES RelWithDebInfo)
  target_compile_definitions(NeoN_config_api INTERFACE NF_DEBUG_INFO)
endif()

if(NeoN_DEFINE_DP_SCALAR)
  target_compile_definitions(NeoN_config_api INTERFACE NeoN_DP_SCALAR=1)
endif()
if(NeoN_DEFINE_DP_LABEL)
  target_compile_definitions(NeoN_config_api INTERFACE NeoN_DP_LABEL=1)
endif()
if(NeoN_DEFINE_US_IDX)
  target_compile_definitions(NeoN_config_api INTERFACE NeoN_US_IDX=1)
endif()

if(NeoN_WITH_MPI)
  target_compile_definitions(NeoN_config_api INTERFACE NF_WITH_MPI_SUPPORT=1)
  target_link_libraries(NeoN_config_api INTERFACE MPI::MPI_CXX)
  if(NeoN_ENABLE_MPI_WITH_THREAD_SUPPORT)
    target_compile_definitions(NeoN_config_api INTERFACE NF_REQUIRE_MPI_THREAD_SUPPORT=1)
  endif()
endif()

# Get list of some *.hpp files in folder include
file(
  GLOB_RECURSE include_files
  RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}/
  *.hpp)

# Convert the list of files into #includes
foreach(include_file ${include_files})
  set(include_statements "${include_statements}#include \"${include_file}\"\n")
endforeach()

configure_file(${CMAKE_CURRENT_LIST_DIR}/NeoN/NeoN.hpp.in ${CMAKE_CURRENT_BINARY_DIR}/NeoN/NeoN.hpp)
