# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: Unlicense

if(NeoN_WITH_VTK)
  neon_unit_test(cgnsMeshReader)
  neon_unit_test(cgnsMeshWriter)
  neon_unit_test(vtuMeshWriter)
  neon_unit_test(cgnsBCPreservation)
  neon_unit_test(cgnsGeometricQuantities)
  neon_unit_test(cgnsMixedCells)
  neon_unit_test(meshConverter)
  neon_unit_test(vtkHdfMeshWriter)
  neon_unit_test(vtkHdfMeshReader)

  add_executable(nn_cgnsRoundTrip tools/cgnsRoundTrip.cpp)
  target_link_libraries(nn_cgnsRoundTrip PRIVATE NeoN NeoN_warnings NeoN_options)
  set_target_properties(
    nn_cgnsRoundTrip PROPERTIES OUTPUT_NAME cgnsRoundTrip RUNTIME_OUTPUT_DIRECTORY
                                                          ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/tests)

  add_executable(nn_cgnsToVtu tools/cgnsToVtu.cpp)
  target_link_libraries(nn_cgnsToVtu PRIVATE NeoN NeoN_warnings NeoN_options)
  set_target_properties(
    nn_cgnsToVtu PROPERTIES OUTPUT_NAME cgnsToVtu RUNTIME_OUTPUT_DIRECTORY
                                                  ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/tests)

  file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/meshFiles
       DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/tests)
endif()
