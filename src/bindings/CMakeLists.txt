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

set(NeoN_PYTHON_BINDING_SRCS
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

if(NeoN_ENABLE_CUDA AND Kokkos_ENABLE_COMPILE_AS_CMAKE_LANGUAGE)
  # CUDA-enabled Kokkos headers require translation units that include them to use CUDA mode. Only
  # in cmake-language mode; traditional mode compiles these as C++ via nvcc_wrapper.
  set_source_files_properties(${NeoN_PYTHON_BINDING_SRCS} PROPERTIES LANGUAGE CUDA)
endif()

# NB_SHARED is opt-in (see NeoN_NANOBIND_SHARED in the top-level CMakeLists): it is only needed to
# exchange nanobind types with another extension module, and it makes _neon depend on a libnanobind
# that must then be installed and found at runtime.
if(NeoN_NANOBIND_SHARED)
  set(NeoN_NANOBIND_LINKAGE NB_SHARED)
endif()

nanobind_add_module(_neon ${NeoN_NANOBIND_LINKAGE} NOMINSIZE ${NeoN_PYTHON_BINDING_SRCS})

if(NeoN_NANOBIND_SHARED AND NOT WIN32)
  # nanobind neither installs its shared library nor sets an RPATH, so the installed _neon would
  # keep a DT_NEEDED that resolves nowhere. Install libnanobind next to _neon and look for it there.
  # Appended so the RPATH inherited from CMAKE_INSTALL_RPATH (libNeoN) is kept. Windows has no
  # RPATH; the DLL is found because it is installed into the same directory.
  if(APPLE)
    set_property(
      TARGET _neon
      APPEND
      PROPERTY INSTALL_RPATH "@loader_path")
  else()
    set_property(
      TARGET _neon
      APPEND
      PROPERTY INSTALL_RPATH "$ORIGIN")
  endif()
endif()

# Link against the NeoN library to access executor classes
target_link_libraries(_neon PRIVATE NeoN::NeoN)

# Ensure the POST_BUILD stubgen step can dlopen all transitive deps (libNeoN.so, libumpire.so,
# libcamp.so, libginkgo*, kokkos, ...) from the build tree. They all land in libNeoN's output
# directory (the top-level CMAKE_LIBRARY_OUTPUT_DIRECTORY, which this file overrides locally for
# _neon only), so a single RPATH entry pointing at it suffices.
set_target_properties(_neon PROPERTIES BUILD_RPATH "$<TARGET_FILE_DIR:NeoN::NeoN>")

if(NOT NeoN_ENABLE_CUDA AND NOT WIN32)
  # Stub generation imports _neon. Skipped for CUDA (may build without an NVIDIA driver present) and
  # on Windows (the extension's dependent DLLs are not on the loader search path at build time, so
  # the import fails).
  add_custom_command(
    TARGET _neon
    POST_BUILD
    COMMAND ${Python_EXECUTABLE} -m nanobind.stubgen -m _neon -O ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}
            -i ${CMAKE_LIBRARY_OUTPUT_DIRECTORY} -D
    COMMENT "Generating type hints (_neon.pyi) for neon..."
    VERBATIM)
endif()

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

__version__ = \"${NeoN_PACKAGE_VERSION}\"

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
  if(NeoN_NANOBIND_SHARED)
    install(
      TARGETS nanobind
      LIBRARY DESTINATION neon COMPONENT python
      RUNTIME DESTINATION neon COMPONENT python)
  endif()
  install(FILES ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/__init__.py DESTINATION neon)
  install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/../neon/py.typed DESTINATION neon)
else()
  # Standalone installation
  install(TARGETS _neon LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/python/neon)
  if(NeoN_NANOBIND_SHARED)
    install(
      TARGETS nanobind
      LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/python/neon
      RUNTIME DESTINATION ${CMAKE_INSTALL_LIBDIR}/python/neon)
  endif()
  install(FILES ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/__init__.py
          DESTINATION ${CMAKE_INSTALL_LIBDIR}/python/neon)
  install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/../neon/py.typed
          DESTINATION ${CMAKE_INSTALL_LIBDIR}/python/neon)
endif()
