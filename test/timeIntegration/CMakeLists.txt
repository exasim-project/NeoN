# SPDX-License-Identifier: Unlicense
# SPDX-FileCopyrightText: 2024 NeoN authors

neon_unit_test(timeIntegration)
neon_unit_test(implicitTimeIntegration)
if(NOT WIN32)
  neon_unit_test(rungeKutta)
endif()
