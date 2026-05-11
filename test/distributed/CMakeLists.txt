# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: Unlicense

if(NeoN_WITH_MPI)
  # Build executables and register 3-rank CTest entries (default names).
  neon_unit_test(operator MPI_SIZE 3)
  neon_unit_test(partitioning MPI_SIZE 3)

  # 2-rank entries (ENABLED): NeoN distributed tests use synthetic fixtures (no external decomp
  # needed) so they can run at any rank count.
  foreach(TEST operator partitioning)
    add_test(
      NAME ${TEST}_mpi2
      COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 neon_test_${TEST}
      WORKING_DIRECTORY ${NeoN_WORKING_DIRECTORY})
    set_tests_properties(${TEST}_mpi2 PROPERTIES TIMEOUT 10 PROCESSORS 2)
  endforeach()

  # 4-rank entries (DISABLED): placeholder for Phase 5 once 4-rank fixtures exist.
  foreach(TEST operator partitioning)
    add_test(
      NAME ${TEST}_mpi4
      COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4 neon_test_${TEST}
      WORKING_DIRECTORY ${NeoN_WORKING_DIRECTORY})
    set_tests_properties(${TEST}_mpi4 PROPERTIES TIMEOUT 10 PROCESSORS 4 DISABLED TRUE)
  endforeach()
endif()
