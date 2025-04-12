# SPDX-License-Identifier: Unlicense
# SPDX-FileCopyrightText: 2023-2024 NeoN authors

if(NeoN_ENABLE_MPI_SUPPORT)
  neon_unit_test(communicator MPI_SIZE 3)
endif()

neon_unit_test(unstructuredMesh)
