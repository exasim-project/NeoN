# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: Unlicense

# convention is to put bindings .so file in a project name folder
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bindings/neon)

message(STATUS "Python_EXECUTABLE = ${Python_EXECUTABLE}")
message(STATUS "Python_INCLUDE_DIRS = ${Python_INCLUDE_DIRS}")
message(STATUS "Python_LIBRARIES = ${Python_LIBRARIES}")

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
  _neon
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
  inputs.cpp
  coNum.cpp
  initialization.cpp
  linearAlgebra.cpp
  dsl.cpp
  database/document.cpp
  database/collection.cpp
  database/database.cpp)

# Link against the NeoN library to access executor classes
target_link_libraries(_neon PRIVATE NeoN::NeoN)

# Generate Python type stubs (.pyi) after building the module
add_custom_command(
  TARGET _neon
  POST_BUILD
  COMMAND ${Python_EXECUTABLE} -m nanobind.stubgen -m _neon -O ${CMAKE_LIBRARY_OUTPUT_DIRECTORY} -i
          ${CMAKE_LIBRARY_OUTPUT_DIRECTORY} -D
  COMMENT "Generating type hints (_neon.pyi) for neon..."
  VERBATIM)

# Generate __init__.py file
file(
  WRITE ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/__init__.py
  "# SPDX-FileCopyrightText: 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

\"\"\"
NeoN - A framework for CFD software

Python bindings for the NeoN CFD framework.

Note: Type stubs (_neon.pyi) are automatically generated during the build process
using nanobind-stubgen. This enables IDE autocomplete and type checking.
\"\"\"

__version__ = \"0.1.0\"

# Import the C++ extension module
try:
    from ._neon import *
    # Explicitly import module attributes that aren't exported by *
    from ._neon import __has_serial__, __has_cpu__, __has_gpu__
except ImportError as e:
    raise ImportError(
        \"Failed to import NeoN C++ extension module. \"
        \"Make sure the package is properly installed. \"
        f\"Error: {e}\"
    ) from e

__all__ = [\"__version__\", \"__has_serial__\", \"__has_cpu__\", \"__has_gpu__\"]
")

# Copy Python package files to build directory
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/../neon/py.typed
               ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/py.typed COPYONLY)

# Installation
if(DEFINED SKBUILD)
  # Install for scikit-build (into neon/ package)
  install(TARGETS _neon LIBRARY DESTINATION neon COMPONENT python)
  install(FILES ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/__init__.py DESTINATION neon)
  install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/../neon/py.typed DESTINATION neon)
else()
  # Standalone installation
  install(TARGETS _neon LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/python/neon)
  install(FILES ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/__init__.py
          DESTINATION ${CMAKE_INSTALL_LIBDIR}/python/neon)
  install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/../neon/py.typed
          DESTINATION ${CMAKE_INSTALL_LIBDIR}/python/neon)
endif()
