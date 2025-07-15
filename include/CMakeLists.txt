# SPDX-FileCopyrightText: 2023 - 2025 NeoN authors
#
# SPDX-License-Identifier: Unlicense

add_library(NeoN_public_api INTERFACE) # dummy target
add_library(NeoN::NeoN_public_api ALIAS NeoN_public_api) # dummy target

# set_property( TARGET NeoN_public_api APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES
# "${CMAKE_CURRENT_BINARY_DIR}")

target_include_directories(
  NeoN_public_api INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
                            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}> $<INSTALL_INTERFACE:.>)

if(NeoN_WITH_GINKGO)
  target_link_libraries(NeoN_public_api INTERFACE Ginkgo::ginkgo)
  target_compile_definitions(NeoN_public_api INTERFACE NF_WITH_GINKGO=1)
  # set_property( TARGET Ginkgo::ginkgo APPEND PROPERTY INTERFACE_COMPILE_DEFINITIONS )
else()
  target_compile_definitions(NeoN_public_api INTERFACE NF_WITH_GINKGO=0)
endif()

if(NeoN_WITH_SUNDIALS)
  target_compile_definitions(NeoN_public_api INTERFACE NN_WITH_SUNDIALS=1)
  target_link_libraries(NeoN_public_api INTERFACE SUNDIALS::arkode SUNDIALS::nvecserial
                                                  SUNDIALS::core)
else()
  target_compile_definitions(NeoN_public_api INTERFACE NN_WITH_SUNDIALS=0)
endif()

target_link_libraries(NeoN_public_api INTERFACE cpptrace::cpptrace Kokkos::kokkos)

if(NeoN_WITH_PETSC)
  add_definitions(${PETSc_DEFINITIONS})
  target_link_libraries(NeoN_public_api INTERFACE ${PETSc_LIBRARIES})
  target_compile_definitions(NeoN_public_api INTERFACE NF_WITH_PETSC=1)
else()
  target_compile_definitions(NeoN_public_api INTERFACE NF_WITH_PETSC=0)
endif()

if(${CMAKE_BUILD_TYPE} MATCHES Debug)
  target_compile_definitions(NeoN_public_api INTERFACE NF_DEBUG)
endif()

if(${CMAKE_BUILD_TYPE} MATCHES RelWithDebInfo)
  target_compile_definitions(NeoN_public_api INTERFACE NF_DEBUG_INFO)
endif()

if(NeoN_DEFINE_DP_SCALAR)
  target_compile_definitions(NeoN_public_api INTERFACE NeoN_DP_SCALAR=1)
endif()
if(NeoN_DEFINE_DP_LABEL)
  target_compile_definitions(NeoN_public_api INTERFACE NeoN_DP_LABEL=1)
endif()
if(NeoN_DEFINE_US_IDX)
  target_compile_definitions(NeoN_public_api INTERFACE NeoN_US_IDX=1)
endif()

if(NeoN_ENABLE_MPI_SUPPORT)
  target_compile_definitions(NeoN_public_api INTERFACE NF_WITH_MPI_SUPPORT=1)
  if(NeoN_ENABLE_MPI_WITH_THREAD_SUPPORT)
    target_compile_definitions(NeoN_public_api INTERFACE NF_REQUIRE_MPI_THREAD_SUPPORT=1)
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
