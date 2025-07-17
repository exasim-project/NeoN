# SPDX-FileCopyrightText: 2023 - 2025 NeoN authors
#
# SPDX-License-Identifier: Unlicense

if(NeoN_ENABLE_MPI_SUPPORT)
  neon_unit_test(communicator MPI_SIZE 3)
endif()

neon_unit_test(unstructuredMesh)
