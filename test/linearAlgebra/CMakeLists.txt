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
  neon_unit_test(ginkgoMergedPgm)
  neon_unit_test(ginkgoScaleCorrection)
  # Only when Ginkgo shares NeoN's Umpire pool (see cmake/CxxThirdParty.cmake)
  if(NeoN_GINKGO_WITH_UMPIRE AND (Kokkos_ENABLE_CUDA OR Kokkos_ENABLE_HIP))
    neon_unit_test(ginkgoUmpirePool)
  endif()
endif()

if(NeoN_WITH_PETSC)
  neon_unit_test(petsc)
endif()
