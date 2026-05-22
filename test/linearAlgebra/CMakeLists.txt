# SPDX-FileCopyrightText: 2024 - 2026 NeoN authors
#
# SPDX-License-Identifier: Unlicense

neon_unit_test(matrix)
neon_unit_test(linearSystem)
neon_unit_test(cooSparsityPattern)
neon_unit_test(csrSparsityPattern)
neon_unit_test(faceToMatrixAddress)
neon_unit_test(utilities)
neon_unit_test(matrixIterator)

# the following tests currently require Ginkgo
if(NeoN_WITH_GINKGO)
  neon_unit_test(ginkgo)
endif()

if(NeoN_WITH_PETSC)
  neon_unit_test(petsc)
endif()
