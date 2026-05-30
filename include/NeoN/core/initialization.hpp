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
    Kokkos::initialize(argc, argv).set_print_configuration(true).set_map_device_id_by("mpi_rank");

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
    Kokkos::initialize(argc, argv).set_print_configuration(true);
    Logging::setNeonDefaultPattern(mpiEnviron);
}

inline void finalize()
{
    Logging::info("Finalizing NeoN");
    Kokkos::finalize();
}
#endif


}
