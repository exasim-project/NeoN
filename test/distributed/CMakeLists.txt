# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: Unlicense

if(NeoN_WITH_MPI)
  neon_unit_test(partitioning MPI_SIZE 3)
  neon_unit_test(partitioning2D MPI_SIZE 4)
  neon_unit_test(operator MPI_SIZE 3)
endif()
