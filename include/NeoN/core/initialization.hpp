// SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include "NeoN/core/logging.hpp"

#include <Kokkos_Core.hpp>
#include <chrono>

#include <cpptrace/cpptrace.hpp>

#ifdef NF_WITH_MPI_SUPPORT
#include "NeoN/core/mpi/environment.hpp"
#endif

namespace NeoN
{

#ifdef NF_WITH_MPI_SUPPORT
inline void initialize(int argc, char* argv[])
{
    // NOTE: Kokkos::initialize(argc, argv) returns void, so the builder methods
    // (set_print_configuration / set_map_device_id_by) cannot be chained onto it.
    // Plain init also avoids Kokkos printing its configuration on every rank.
    Kokkos::initialize(argc, argv);

    cpptrace::register_terminate_handler();
    mpi::Environment mpiEnviron;
    Logging::setNeonDefaultPattern(mpiEnviron);
}

inline void finalize()
{
    Logging::info("Finalizing NeoN");
    Kokkos::finalize();
}
#else
inline void initialize(int argc, char* argv[])
{
    // See note above: Kokkos::initialize(argc, argv) returns void.
    Kokkos::initialize(argc, argv);
    Logging::setNeonDefaultPattern(mpiEnviron);
}

inline void finalize()
{
    Logging::info("Finalizing NeoN");
    Kokkos::finalize();
}
#endif


}
