# SPDX-FileCopyrightText: 2024 - 2026 NeoN authors
#
# SPDX-License-Identifier: Unlicense

neon_unit_test(coNum)

if(NeoN_WITH_MPI)
  neon_unit_test(coNum_distributed MPI_SIZE 3)
endif()
