# SPDX-FileCopyrightText: 2024 - 2026 NeoN authors
#
# SPDX-License-Identifier: Unlicense

neon_unit_test(ddtOperator)
neon_unit_test(laplacianOperator)
neon_unit_test(cellLimitedGrad)
neon_unit_test(gaussGreenDiv)
neon_unit_test(boundedDiv)
neon_unit_test(gaussGreenDivLaplacian)
neon_unit_test(sourceTerm)
neon_unit_test(reconstruct)

if(NeoN_WITH_MPI)
  neon_unit_test(distributedReconstruct MPI_SIZE 3)
endif()
