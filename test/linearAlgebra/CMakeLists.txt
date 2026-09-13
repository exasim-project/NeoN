# SPDX-FileCopyrightText: 2024 - 2026 NeoN authors
#
# SPDX-License-Identifier: Unlicense

neon_unit_test(matrix)
neon_unit_test(matrixIterator)
neon_unit_test(linearSystem)
neon_unit_test(cooSparsityPattern)
neon_unit_test(csrSparsityPattern)
neon_unit_test(faceToMatrixAddress)
neon_unit_test(sharedSparsity)
neon_unit_test(utilities)

# the following tests currently require Ginkgo
if(NeoN_WITH_GINKGO)
  neon_unit_test(ginkgo)
  # scale_correction needs a Ginkgo that provides it (see NeoN_GINKGO_FORK in CxxThirdParty.cmake);
  # otherwise the configFile fails to parse at runtime.
  if(NeoN_GINKGO_FORK)
    neon_unit_test(ginkgoScaleCorrection)
  endif()
endif()

if(NeoN_WITH_PETSC)
  neon_unit_test(petsc)
endif()
