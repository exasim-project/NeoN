# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: Unlicense

if(CMAKE_VERSION VERSION_LESS 3.18)
  set(DEV_MODULE Development)
else()
  set(DEV_MODULE Development.Module)
endif()

# convention is to put bindings .so file in a project name folder
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bindings/neon)

find_package(
  Python 3.8
  COMPONENTS Interpreter ${DEV_MODULE}
  REQUIRED)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE
      Release
      CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel"
                                               "RelWithDebInfo")
endif()

# Set output directory based on build mode
if(DEFINED SKBUILD)
  # scikit-build-core manages installation paths
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
else()
  # Standalone build - convention is to put bindings .so file in a project name folder
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bindings/neon)
endif()

nanobind_add_module(
  neon
  neon.cpp
  executors.cpp
  scalar.cpp
  vec3.cpp
  vectors.cpp
  containerFreeFunctions.cpp
  boundaryMesh.cpp
  unstructuredMesh.cpp
  surfaceField.cpp
  volumeField.cpp
  surfaceInterpolation.cpp
  coNum.cpp
  database/document.cpp
  database/collection.cpp
  database/database.cpp)

# Link against the NeoN library to access executor classes
target_link_libraries(neon PRIVATE NeoN::NeoN)

# Installation
if(DEFINED SKBUILD)
  # Install for scikit-build (into neon/ package)
  install(TARGETS neon LIBRARY DESTINATION neon COMPONENT python)
else()
  # Standalone installation
  install(TARGETS neon LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/python/neon)
endif()
