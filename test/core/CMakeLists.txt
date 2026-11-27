# SPDX-FileCopyrightText: 2024 - 2026 NeoN authors
#
# SPDX-License-Identifier: Unlicense

if(NeoN_WITH_MPI)
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
                                                      NeoN::NeoN)
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

# The runtime-selection tables must be shared with a library built with hidden (inline) visibility,
# like the Python bindings or a conda-forge build (#588). NeoN is a static library on Windows, so a
# DLL would carry its own copy of the registry there.
if(NOT WIN32)
  add_library(neon_test_shared_registry_plugin SHARED "sharedRegistry/plugin.cpp")
  target_link_libraries(neon_test_shared_registry_plugin PRIVATE NeoN::NeoN)
  set_target_properties(neon_test_shared_registry_plugin PROPERTIES CXX_VISIBILITY_PRESET hidden
                                                                    VISIBILITY_INLINES_HIDDEN ON)
  install(TARGETS neon_test_shared_registry_plugin)
  neon_unit_test(sharedRegistry/sharedRegistry COMMAND neon_test_sharedRegistry)
  target_link_libraries(neon_test_sharedRegistry PRIVATE neon_test_shared_registry_plugin)
endif()
