# SPDX-License-Identifier: Unlicense
# SPDX-FileCopyrightText: 2024 NeoN authors

neon_unit_test(euler)
if(NOT WIN32)
  neon_unit_test(rungeKutta)
endif()
