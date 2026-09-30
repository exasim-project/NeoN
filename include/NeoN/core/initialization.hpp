// SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include "NeoN/core/logging.hpp"

namespace NeoN
{

/* @brief Initialize Kokkos, the terminate handler and the default logging pattern.
 *
 * Call after MPI has been initialized by the host application, if MPI is used.
 */
void initialize(int argc, char* argv[]);

/* @brief Finalize Kokkos. */
void finalize();
} // namespace NeoN
