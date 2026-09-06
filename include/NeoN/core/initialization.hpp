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

inline void initialize(int argc, char* argv[])
{
    // NOTE: Kokkos::initialize(argc, argv) returns void, so the builder methods
    // (set_print_configuration / set_map_device_id_by) cannot be chained onto it.
    // Plain init also avoids Kokkos printing its configuration on every rank.
    // Kokkos::initialize(argc, argv);
    auto kokkosSettings = Kokkos::InitializationSettings();
#ifdef NF_WITH_MPI_SUPPORT
    {
        const NeoN::mpi::Environment mpiEnv;
        const int nDev = Kokkos::num_devices();
        if (nDev > 0) kokkosSettings.set_device_id(mpiEnv.rank() % nDev);
    }
#else
    kokkosSettings.set_map_device_id_by("random");
#endif
    Kokkos::initialize(kokkosSettings);

    cpptrace::register_terminate_handler();
    // setNeonDefaultPattern() reads the MPI rank internally (when MPI is up) to
    // mute non-root ranks; it works with or without MPI support. Call it after the
    // host has initialized MPI so the rank is known.
    Logging::setNeonDefaultPattern();
}

inline void finalize()
{
    Logging::info("Finalizing NeoN");
    Kokkos::finalize();
}


}
