# SPDX-FileCopyrightText: 2023 - 2025 NeoN authors
#
# SPDX-License-Identifier: MIT

list(APPEND CMAKE_MODULE_PATH ${Catch2_SOURCE_DIR}/extras)

# has to be first since it adds the main target
add_subdirectory(catch2)
set(TEST_CATCH2_DIR ${CMAKE_CURRENT_LIST_DIR}/catch2)
message("TEST_CATCH2_DIR ${TEST_CATCH2_DIR}")

# include "NeoN/core/parallelAlgorithms.hpp" This function creates unit tests. It provides the
# following keywords:
#
# * MPI_SIZE: the number of MPI processors to be used, defaults to 1 if not set
# * COMMAND: the test command (same behavior as for CMake's add_test), defaults to the test name
# * WORKING_DIRECTORY: the working directory for the test, defaults to CMAKE_BINARY_DIR/bin
function(NeoN_unit_test TEST)
  set(options "")
  set(oneValueKeywords "MPI_SIZE" "COMMAND" "WORKING_DIRECTORY")
  set(multiValueKeywords "")
  cmake_parse_arguments("NeoN" "${options}" "${oneValueKeywords}" "${multiValueKeywords}" ${ARGN})
  if(NOT DEFINED "NeoN_WORKING_DIRECTORY")
    set(NeoN_WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/tests)
  endif()
  if(NOT DEFINED "NeoN_COMMAND")
    set(NeoN_COMMAND nn_${TEST})
  endif()

  add_executable(${NeoN_COMMAND} "${TEST}.cpp")
  install(TARGETS ${NeoN_COMMAND})

  if(NeoN_WITH_PETSC)
    target_link_libraries(${NeoN_COMMAND} PRIVATE ${PETSc_LIBRARIES})
    target_link_libraries(${NeoN_COMMAND} PRIVATE ${PETSc_LIBRARIES}
                                                  ${PETSc_LIBRARY_DIRS}/libkokkoscore.so mpi)
    target_include_directories(${NeoN_COMMAND} PRIVATE ${PETSc_INCLUDE_DIRS})
    target_link_directories(${NeoN_COMMAND} PRIVATE ${PETSc_LIBRARY_DIRS})
    link_directories(${PETSc_LIBRARY_DIRS})
  endif()
  target_include_directories(${NeoN_COMMAND} PRIVATE "${TEST_CATCH2_DIR}")
  set_target_properties(${NeoN_COMMAND} PROPERTIES OUTPUT_NAME ${TEST})

  target_link_libraries(${NeoN_COMMAND} PRIVATE NeoN_warnings NeoN_options NeoN cpptrace::cpptrace)
  if(NOT NeoN_MPI_SIZE)
    target_link_libraries(${NeoN_COMMAND} PRIVATE NeoN_catch_main)
  else()
    target_link_libraries(${NeoN_COMMAND} PRIVATE NeoN_catch_main_mpi)
  endif()

  if(NeoN_WITH_SUNDIALS)
    target_link_libraries(${NeoN_COMMAND} PRIVATE SUNDIALS::arkode)

  endif()
  if(WIN32)
    set_target_properties(
      ${NeoN_COMMAND}
      PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${NeoN_WORKING_DIRECTORY}/$<0:>
                 LIBRARY_OUTPUT_DIRECTORY ${NeoN_WORKING_DIRECTORY}/$<0:>
                 ARCHIVE_OUTPUT_DIRECTORY ${NeoN_WORKING_DIRECTORY}/$<0:>)
  else()
    # cmake-lint: disable=C0307
    set_target_properties(${NeoN_COMMAND} PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                                     ${NeoN_WORKING_DIRECTORY})
  endif()

  if(NOT NeoN_MPI_SIZE)
    add_test(
      NAME ${TEST}
      COMMAND ${NeoN_COMMAND}
      WORKING_DIRECTORY ${NeoN_WORKING_DIRECTORY})
  else()
    add_test(
      NAME ${TEST}
      COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${NeoN_MPI_SIZE} ${NeoN_COMMAND}
      WORKING_DIRECTORY ${NeoN_WORKING_DIRECTORY})
    set_tests_properties(${TEST} PROPERTIES TIMEOUT 10)
    set_tests_properties(${TEST} PROPERTIES PROCESSORS ${NeoN_MPI_SIZE})
    if(${MPIEXEC_MAX_NUMPROCS} LESS NeoN_MPI_SIZE)
      message(WARNING "Test <${TEST}> requested <${NeoN_MPI_SIZE}> processors, "
                      "but only <${numberOfProcessors}> are available. Disabling test " "<${TEST}>")
      set_tests_properties(${TEST} PROPERTIES DISABLED TRUE)
    endif()
  endif()
endfunction()

add_subdirectory(core)
add_subdirectory(dsl)
add_subdirectory(fields)
add_subdirectory(finiteVolume)
add_subdirectory(linearAlgebra)
add_subdirectory(mesh)
add_subdirectory(timeIntegration)
