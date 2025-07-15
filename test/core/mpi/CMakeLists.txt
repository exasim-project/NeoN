# SPDX-FileCopyrightText: 2023 - 2025 NeoN authors
#
# SPDX-License-Identifier: MIT

if(NeoN_ENABLE_MPI_SUPPORT)
  neon_unit_test(fullDuplexCommBuffer MPI_SIZE 3)
  neon_unit_test(halfDuplexCommBuffer MPI_SIZE 3)
  neon_unit_test(operators MPI_SIZE 3)
endif()
