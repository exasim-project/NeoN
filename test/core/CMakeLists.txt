# SPDX-License-Identifier: Unlicense
# SPDX-FileCopyrightText: 2024 NeoN authors

if(NeoN_ENABLE_MPI_SUPPORT)
  add_subdirectory(mpi)
endif()

add_subdirectory(database)
add_subdirectory(primitives)
add_subdirectory(vector)
neon_unit_test(array)
neon_unit_test(dictionary)
neon_unit_test(tokenList)
neon_unit_test(input)
neon_unit_test(executor)
neon_unit_test(parallelAlgorithms)
neon_unit_test(view)
neon_unit_test(segmentedVector)

add_executable(runTimeSelectionFactory "runTimeSelectionFactory.cpp")
target_link_libraries(runTimeSelectionFactory PRIVATE Catch2::Catch2WithMain cpptrace::cpptrace
                                                      NeoN)
if(WIN32)
  set_target_properties(
    runTimeSelectionFactory
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/tests/$<0:>
               LIBRARY_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/tests/$<0:>
               ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/tests/$<0:>)

else()
  set_property(TARGET runTimeSelectionFactory PROPERTY RUNTIME_OUTPUT_DIRECTORY
                                                       ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/tests)
  target_include_directories(runTimeSelectionFactory PRIVATE "${CMAKE_CURRENT_LIST_DIR}/../catch2")
endif()

add_test(
  NAME runTimeSelectionFactory
  COMMAND runTimeSelectionFactory
  WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/tests)
