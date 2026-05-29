# SPDX-FileCopyrightText: 2024 - 2026 NeoN authors
#
# SPDX-License-Identifier: Unlicense

neon_unit_test(basicGeometryScheme)

if(NeoN_WITH_MPI)
  neon_unit_test(basicGeometryScheme_distributed MPI_SIZE 3)
endif()
