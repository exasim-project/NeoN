# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: Unlicense

function(NeoN_benchmark BENCH)

  add_executable(bench_${BENCH} "${BENCH}.cpp")
  target_link_libraries(bench_${BENCH} PRIVATE Catch2::Catch2 NeoN)
  target_include_directories(bench_${BENCH} PRIVATE "${CMAKE_SOURCE_DIR}")

  if(WIN32)
    set_target_properties(
      bench_${BENCH}
      PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/benchmarks/
                 LIBRARY_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/benchmarks/
                 ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/benchmarks/)
  else()
    set_property(TARGET bench_${BENCH} PROPERTY RUNTIME_OUTPUT_DIRECTORY
                                                ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/benchmarks)
  endif()

  if(NOT DEFINED "NeoN_WORKING_DIRECTORY")
    set(NeoN_WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/benchmarks)
  endif()
  add_test(
    NAME bench_${BENCH}
    COMMAND sh -c "./bench_${BENCH} -r xml > ${BENCH}.xml"
    WORKING_DIRECTORY ${NeoN_WORKING_DIRECTORY})
endfunction()

function(NeoN_mpi_benchmark BENCH)
  set(options "")
  set(oneValueKeywords "")
  set(multiValueKeywords "MPI_SIZES")
  cmake_parse_arguments("NeoN" "${options}" "${oneValueKeywords}" "${multiValueKeywords}" ${ARGN})

  if(NOT DEFINED NeoN_MPI_SIZES)
    set(NeoN_MPI_SIZES 4)
  endif()

  set(MPI_REPORTER_DIR "${CMAKE_SOURCE_DIR}/test/catch2")

  add_executable(
    bench_${BENCH} "${BENCH}.cpp" "${MPI_REPORTER_DIR}/mpiReporter.cpp"
                   "${MPI_REPORTER_DIR}/mpiGlobals.cpp" "${MPI_REPORTER_DIR}/mpiSerialization.cpp")
  target_link_libraries(bench_${BENCH} PRIVATE Catch2::Catch2 NeoN MPI::MPI_CXX cpptrace::cpptrace)
  target_include_directories(bench_${BENCH} PRIVATE "${CMAKE_SOURCE_DIR}" "${MPI_REPORTER_DIR}")

  if(WIN32)
    set_target_properties(
      bench_${BENCH}
      PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/benchmarks/
                 LIBRARY_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/benchmarks/
                 ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/benchmarks/)
  else()
    set_property(TARGET bench_${BENCH} PROPERTY RUNTIME_OUTPUT_DIRECTORY
                                                ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/benchmarks)
  endif()

  if(NOT DEFINED "NeoN_WORKING_DIRECTORY")
    set(NeoN_WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/benchmarks)
  endif()

  foreach(NP IN LISTS NeoN_MPI_SIZES)
    add_test(
      NAME bench_${BENCH}_${NP}ranks
      COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${NP} ./bench_${BENCH} -r mpi
      WORKING_DIRECTORY ${NeoN_WORKING_DIRECTORY})
    set_tests_properties(bench_${BENCH}_${NP}ranks PROPERTIES PROCESSORS ${NP})
    if(${MPIEXEC_MAX_NUMPROCS} LESS ${NP})
      message(
        WARNING
          "Benchmark <${BENCH}> requested <${NP}> processors, "
          "but only <${MPIEXEC_MAX_NUMPROCS}> are available. Disabling bench_${BENCH}_${NP}ranks")
      set_tests_properties(bench_${BENCH}_${NP}ranks PROPERTIES DISABLED TRUE)
    endif()
  endforeach()
endfunction()

add_subdirectory(fields)
add_subdirectory(memory)
add_subdirectory(dsl)
add_subdirectory(linearAlgebra)
add_subdirectory(finiteVolume/cellCentred/operator)
add_subdirectory(finiteVolume/cellCentred/interpolation)
add_subdirectory(distributed)
