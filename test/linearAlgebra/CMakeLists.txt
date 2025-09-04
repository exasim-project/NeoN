# SPDX-FileCopyrightText: 2024 - 2025 NeoN authors
#
# SPDX-License-Identifier: Unlicense

neon_unit_test(CSRMatrix)
neon_unit_test(linearSystem)
neon_unit_test(sparsityPattern)
neon_unit_test(utilities)

# the following tests currently require Ginkgo
if(NeoN_WITH_GINKGO)
  neon_unit_test(ginkgo)
endif()

if(NeoN_WITH_PETSC)
  neon_unit_test(petsc)
  neon_unit_test(petscSolverContext)
endif()
