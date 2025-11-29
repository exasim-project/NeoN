# SPDX-FileCopyrightText: 2024 - 2025 NeoN authors
#
# SPDX-License-Identifier: Unlicense

neon_unit_test(timeIntegration)
neon_unit_test(implicitTimeIntegration)
neon_unit_test(ddtCorr)
if(NOT WIN32 AND NeoN_WITH_SUNDIALS)
  neon_unit_test(rungeKutta)
endif()
