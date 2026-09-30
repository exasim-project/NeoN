# SPDX-FileCopyrightText: 2026 NeoN authors
#
# SPDX-License-Identifier: Unlicense

# neon_add_host_object_library(<name> SOURCES <src>...)
#
# Creates an OBJECT library for host-only translation units (see doc/splitCompilation.rst):
#
# * compiled at NeoN_HOST_CXX_STANDARD instead of CMAKE_CXX_STANDARD,
# * sees NeoN_config_api (include paths and all layout-relevant definitions) but not Kokkos, so it
#   carries no -DKOKKOS_DEPENDENCE and kokkos_launch_compiler leaves it with the host compiler,
# * compiled with NEON_HOST_ONLY_TU=1 and force-includes src/hostOnlyGuard.hpp, which poisons the
#   identifier `Kokkos` so any Kokkos header reached from such a TU is a hard error.
#
# Link the objects into the owning target with target_sources(<owner> PRIVATE
# $<TARGET_OBJECTS:<name>>). Sources must never be marked LANGUAGE CUDA.
function(neon_add_host_object_library name)
  cmake_parse_arguments(PARSE_ARGV 1 ARG "" "" "SOURCES")

  add_library(${name} OBJECT ${ARG_SOURCES})
  set_target_properties(
    ${name}
    PROPERTIES CXX_STANDARD ${NeoN_HOST_CXX_STANDARD}
               CXX_STANDARD_REQUIRED ON
               CXX_EXTENSIONS OFF
               POSITION_INDEPENDENT_CODE ON)
  target_link_libraries(${name} PRIVATE NeoN_config_api NeoN_warnings NeoN_options)
  target_compile_definitions(${name} PRIVATE NEON_HOST_ONLY_TU=1)
  # MSVC has no equivalent of #pragma GCC poison; there the missing Kokkos include path is the only
  # guard.
  target_compile_options(
    ${name}
    PRIVATE
      "$<$<CXX_COMPILER_ID:GNU,Clang,AppleClang,IntelLLVM>:SHELL:-include ${NeoN_HOST_ONLY_GUARD}>")
endfunction()

if(NOT NeoN_HOST_CXX_STANDARD MATCHES "^(20|23|26)$")
  message(
    FATAL_ERROR "NeoN_HOST_CXX_STANDARD must be 20, 23 or 26, got '${NeoN_HOST_CXX_STANDARD}'")
endif()

if(NOT NeoN_HOST_CXX_STANDARD STREQUAL "20" AND CMAKE_CXX_COMPILER MATCHES "nvcc_wrapper")
  message(
    FATAL_ERROR
      "NeoN_HOST_CXX_STANDARD=${NeoN_HOST_CXX_STANDARD} needs a host C++ compiler, but "
      "CMAKE_CXX_COMPILER is nvcc_wrapper, which only accepts C++20. Build Kokkos and NeoN with "
      "Kokkos_ENABLE_COMPILE_AS_CMAKE_LANGUAGE=ON, set CMAKE_CXX_COMPILER to the host compiler and "
      "CMAKE_CUDA_HOST_COMPILER to a compiler nvcc supports. See doc/splitCompilation.rst.")
endif()

set(NeoN_HOST_ONLY_GUARD "${CMAKE_CURRENT_LIST_DIR}/../src/hostOnlyGuard.hpp")
message(STATUS "NeoN host-only TUs: C++${NeoN_HOST_CXX_STANDARD} with ${CMAKE_CXX_COMPILER_ID} "
               "${CMAKE_CXX_COMPILER_VERSION}")
