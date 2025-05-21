# SPDX-License-Identifier: Unlicense
# SPDX-FileCopyrightText: 2023 NeoN authors

add_library(NeoN ${NeoN_LIB_TYPE})

include(GNUInstallDirs)

if(NeoN_ENABLE_CUDA)
  set_source_files_properties(${NeoN_SRCS} PROPERTIES LANGUAGE CUDA)
endif()

target_sources(
  NeoN
  PRIVATE "core/primitives/vec3.cpp"
          "core/time.cpp"
          "core/vector/vector.cpp"
          "core/vector/vectorFreeFunctions.cpp"
          "core/database/database.cpp"
          "core/database/collection.cpp"
          "core/database/document.cpp"
          "core/database/fieldCollection.cpp"
          "core/database/oldTimeCollection.cpp"
          "core/dictionary.cpp"
          "core/demangle.cpp"
          "core/tokenList.cpp"
          "dsl/coeff.cpp"
          "dsl/explicit.cpp"
          "dsl/spatialOperator.cpp"
          "dsl/temporalOperator.cpp"
          "executor/CPUExecutor.cpp"
          "executor/GPUExecutor.cpp"
          "executor/serialExecutor.cpp"
          "linearAlgebra/utilities.cpp"
          "linearAlgebra/ginkgo.cpp"
          "mesh/unstructured/boundaryMesh.cpp"
          "mesh/unstructured/unstructuredMesh.cpp"
          "linearAlgebra/sparsityPattern.cpp"
          "finiteVolume/cellCentred/stencil/stencilDataBase.cpp"
          "finiteVolume/cellCentred/stencil/geometryScheme.cpp"
          "finiteVolume/cellCentred/stencil/basicGeometryScheme.cpp"
          "finiteVolume/cellCentred/stencil/cellToFaceStencil.cpp"
          "finiteVolume/cellCentred/boundary/boundary.cpp"
          "finiteVolume/cellCentred/operators/ddtOperator.cpp"
          "finiteVolume/cellCentred/fields/volumeField.cpp"
          "finiteVolume/cellCentred/operators/gaussGreenGrad.cpp"
          "finiteVolume/cellCentred/operators/gaussGreenDiv.cpp"
          "finiteVolume/cellCentred/operators/gaussGreenLaplacian.cpp"
          "finiteVolume/cellCentred/operators/sourceTerm.cpp"
          "finiteVolume/cellCentred/operators/surfaceIntegrate.cpp"
          "finiteVolume/cellCentred/interpolation/linear.cpp"
          "finiteVolume/cellCentred/interpolation/upwind.cpp"
          "finiteVolume/cellCentred/faceNormalGradient/uncorrected.cpp"
          "finiteVolume/cellCentred/auxiliary/coNum.cpp"
          "timeIntegration/timeIntegration.cpp"
          "timeIntegration/rungeKutta.cpp")

if(NeoN_ENABLE_MPI_SUPPORT)
  target_sources(NeoN PRIVATE "core/mpi/halfDuplexCommBuffer.cpp"
                              "mesh/unstructured/communicator.cpp")
endif()

include(${CMAKE_SOURCE_DIR}/cmake/Sanitizer.cmake)
enable_sanitizers(NeoN NeoN_ENABLE_SANITIZE_ADDRESS NeoN_ENABLE_SANITIZE_LEAK
                  NeoN_ENABLE_SANITIZE_UB NeoN_ENABLE_SANITIZE_THREAD NeoN_ENABLE_SANITIZE_MEMORY)

target_link_libraries(NeoN PRIVATE NeoN_warnings NeoN_options Kokkos::kokkos)
target_link_libraries(NeoN PUBLIC Kokkos::kokkos sundials_core sundials_arkode sundials_nvecserial
                                  cpptrace::cpptrace)

target_link_libraries(NeoN PUBLIC NeoN_public_api)

if(NeoN_ENABLE_MPI_SUPPORT)
  target_link_libraries(NeoN PUBLIC MPI::MPI_CXX)
endif()
if(WIN32)
  set_target_properties(
    NeoN
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/$<0:>
               LIBRARY_OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/$<0:>
               ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY}/$<0:>)
endif()
