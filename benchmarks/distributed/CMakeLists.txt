# SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
#
# SPDX-License-Identifier: Unlicense

if(NeoN_WITH_MPI)
  set(MPI_REPORTER_DIR "${CMAKE_SOURCE_DIR}/test/catch2")
  set(DIST_COMMON_SOURCES
      "${CMAKE_CURRENT_SOURCE_DIR}/common.cpp" "${MPI_REPORTER_DIR}/mpiReporter.cpp"
      "${MPI_REPORTER_DIR}/mpiGlobals.cpp" "${MPI_REPORTER_DIR}/mpiSerialization.cpp")
  set(DIST_MPI_SIZES 1 2 4 8 9)
  set(DIST_LINK_LIBS Catch2::Catch2 NeoN MPI::MPI_CXX cpptrace::cpptrace)

  foreach(BENCH weakScaling strongScaling)
    add_executable(bench_distributed${BENCH} "${CMAKE_CURRENT_SOURCE_DIR}/${BENCH}.cpp"
                                             ${DIST_COMMON_SOURCES})
    target_link_libraries(bench_distributed${BENCH} PRIVATE ${DIST_LINK_LIBS})
    target_include_directories(
      bench_distributed${BENCH} PRIVATE "${CMAKE_SOURCE_DIR}" "${MPI_REPORTER_DIR}"
                                        "${CMAKE_CURRENT_SOURCE_DIR}")
    set_property(TARGET bench_distributed${BENCH}
                 PROPERTY RUNTIME_OUTPUT_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/benchmarks")
    foreach(NP IN LISTS DIST_MPI_SIZES)
      add_test(
        NAME bench_distributed${BENCH}_${NP}ranks
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${NP} ./bench_distributed${BENCH} -r
                mpi
        WORKING_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/benchmarks")
      set_tests_properties(bench_distributed${BENCH}_${NP}ranks PROPERTIES PROCESSORS ${NP})
      if(${MPIEXEC_MAX_NUMPROCS} LESS ${NP})
        message(WARNING "Benchmark <distributed${BENCH}> requested <${NP}> processors "
                        "but only <${MPIEXEC_MAX_NUMPROCS}> are available — "
                        "disabling bench_distributed${BENCH}_${NP}ranks")
        set_tests_properties(bench_distributed${BENCH}_${NP}ranks PROPERTIES DISABLED TRUE)
      endif()
    endforeach()
  endforeach()
endif()
