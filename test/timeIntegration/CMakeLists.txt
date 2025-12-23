# SPDX-FileCopyrightText: 2024 - 2025 NeoN authors
#
# SPDX-License-Identifier: Unlicense

neon_unit_test(explicitTimeIntegration)
neon_unit_test(implicitTimeIntegration)
if(NOT WIN32 AND NeoN_WITH_SUNDIALS)
  neon_unit_test(rungeKutta)
endif()
neon_unit_test(ddtFluxCorr)
