# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: Unlicense

add_library(NeoN ${NeoN_LIB_TYPE})
add_library(NeoN::NeoN ALIAS NeoN) # dummy target

# Bind libNeoN's calls to its own template instantiations instead of a consumer's copy; see
# detail::visitExecutor in core/parallelAlgorithms.hpp (exasim-project/NeoFOAM#410). Data, e.g. the
# runtime selection tables, stays shared process-wide.
if(NeoN_LIB_TYPE STREQUAL "SHARED" AND NOT APPLE)
  target_link_options(NeoN PRIVATE "LINKER:-Bsymbolic-functions")
endif()

include(GNUInstallDirs)

set(NeoN_SRCS
    "core/vector/vector.cpp"
    "core/vector/vectorFreeFunctions.cpp"
    "core/database/fieldCollection.cpp"
    "core/database/oldTimeCollection.cpp"
    "core/backendInfo.cpp"
    "core/initialization.cpp"
    "executor/executor.cpp"
    "core/memory/kokkos.cpp"
    "dsl/coeff.cpp"
    "dsl/explicit.cpp"
    "dsl/spatialOperator.cpp"
    "dsl/temporalOperator.cpp"
    "linearAlgebra/utilities.cpp"
    "linearAlgebra/matrix.cpp"
    "linearAlgebra/ginkgo/ginkgo.cpp"
    "linearAlgebra/ginkgo/ginkgoL1Stop.cpp"
    "linearAlgebra/faceToMatrixAddress.cpp"
    "linearAlgebra/cooSparsityPattern.cpp"
    "linearAlgebra/csrSparsityPattern.cpp"
    "linearAlgebra/meshIterationStrategies.cpp"
    "mesh/unstructured/unstructuredMesh.cpp"
    "mesh/unstructured/uniformMeshDataGenerator.cpp"
    "finiteVolume/cellCentred/stencil/geometryScheme.cpp"
    "finiteVolume/cellCentred/stencil/basicGeometryScheme.cpp"
    "finiteVolume/cellCentred/stencil/cellToFaceStencil.cpp"
    "finiteVolume/cellCentred/boundary/boundary.cpp"
    "finiteVolume/cellCentred/boundary/boundaryContext.cpp"
    "finiteVolume/cellCentred/boundary/surface/processor.cpp"
    "finiteVolume/cellCentred/boundary/volume/processor.cpp"
    "finiteVolume/cellCentred/operators/ddtOperator.cpp"
    "finiteVolume/cellCentred/fields/volumeField.cpp"
    "finiteVolume/cellCentred/operators/gaussGreenGrad.cpp"
    "finiteVolume/cellCentred/operators/cellLimitedGrad.cpp"
    "finiteVolume/cellCentred/operators/gaussGreenDiv.cpp"
    "finiteVolume/cellCentred/operators/boundedDiv.cpp"
    "finiteVolume/cellCentred/operators/gaussGreenLaplacian.cpp"
    "finiteVolume/cellCentred/operators/gaussGreenDivLaplacian.cpp"
    "finiteVolume/cellCentred/operators/gaussGreenDdtDivLaplacian.cpp"
    "finiteVolume/cellCentred/operators/sourceTerm.cpp"
    "finiteVolume/cellCentred/operators/surfaceIntegrate.cpp"
    "finiteVolume/cellCentred/operators/reconstruct.cpp"
    "finiteVolume/cellCentred/interpolation/linear.cpp"
    "finiteVolume/cellCentred/interpolation/upwind.cpp"
    "finiteVolume/cellCentred/interpolation/linearUpwind.cpp"
    "finiteVolume/cellCentred/interpolation/limitedLinear.cpp"
    "finiteVolume/cellCentred/faceNormalGradient/corrected.cpp"
    "finiteVolume/cellCentred/faceNormalGradient/limitedCorrected.cpp"
    "finiteVolume/cellCentred/faceNormalGradient/uncorrected.cpp"
    "finiteVolume/cellCentred/auxiliary/coNum.cpp"
    "timeIntegration/timeIntegration.cpp"
    "timeIntegration/rungeKutta.cpp")

if(NeoN_WITH_MPI)
  list(APPEND NeoN_SRCS "mesh/unstructured/communicator.cpp"
       "mesh/unstructured/distributedUnstructuredMesh.cpp"
       "linearAlgebra/ginkgo/ginkgoDistributed.cpp")
endif()

# Host-only translation units (see doc/splitCompilation.rst). They include no Kokkos header, launch
# no kernels and are compiled with NeoN_HOST_CXX_STANDARD as an object library linked into NeoN. The
# device TUs above stay at CMAKE_CXX_STANDARD (C++20), the level nvcc supports.
set(NeoN_HOST_SRCS
    "core/primitives/vec3.cpp"
    "core/primitives/tensor.cpp"
    "core/primitives/symmTensor.cpp"
    "core/time.cpp"
    "core/database/database.cpp"
    "core/database/collection.cpp"
    "core/database/document.cpp"
    "core/memory/umpire.cpp"
    "core/dictionary.cpp"
    "core/demangle.cpp"
    "core/tokenList.cpp"
    "core/logging.cpp"
    "executor/CPUExecutor.cpp"
    "executor/GPUExecutor.cpp"
    "executor/serialExecutor.cpp"
    "mesh/unstructured/boundaryMesh.cpp")

if(NeoN_WITH_MPI)
  list(APPEND NeoN_HOST_SRCS "core/mpi/halfDuplexCommBuffer.cpp")
endif()

# Only mark sources as the CUDA language in cmake-language mode. In traditional Kokkos mode
# (nvcc_wrapper) these compile as C++ and Kokkos supplies the CUDA flags (--expt-extended-lambda,
# --expt-relaxed-constexpr); forcing LANGUAGE CUDA there bypasses the CUDA language setup and those
# flags.
if(NeoN_ENABLE_CUDA AND Kokkos_ENABLE_COMPILE_AS_CMAKE_LANGUAGE)
  set_source_files_properties(${NeoN_SRCS} PROPERTIES LANGUAGE CUDA)
endif()

target_sources(NeoN PRIVATE ${NeoN_SRCS})

neon_add_host_object_library(NeoN_host SOURCES ${NeoN_HOST_SRCS})
target_sources(NeoN PRIVATE $<TARGET_OBJECTS:NeoN_host>)

if(NeoN_BUILD_PYTHON_BINDINGS)
  add_subdirectory(bindings)
endif()

include(${CMAKE_SOURCE_DIR}/cmake/Sanitizer.cmake)
foreach(_neon_target NeoN NeoN_host)
  enable_sanitizers(${_neon_target} NeoN_ENABLE_SANITIZE_ADDRESS NeoN_ENABLE_SANITIZE_LEAK
                    NeoN_ENABLE_SANITIZE_UB NeoN_ENABLE_SANITIZE_THREAD NeoN_ENABLE_SANITIZE_MEMORY)
endforeach()

target_link_libraries(NeoN PRIVATE NeoN_warnings NeoN_options)
target_link_libraries(NeoN PUBLIC NeoN_public_api)

if(WIN32)
  set_target_properties(
    NeoN
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/$<0:>
               LIBRARY_OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/$<0:>
               ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY}/$<0:>)
endif()
