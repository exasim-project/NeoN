// SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include "NeoN/core/logging.hpp"

#include <Kokkos_Core.hpp>
#include <chrono>

#ifdef NF_WITH_MPI_SUPPORT
#include <mpi.h>
#endif


namespace NeoN
{

inline void initialize(int argc, char* argv[])
{
#ifdef NF_WITH_MPI_SUPPORT
    int mpiInitialized = 0;
    MPI_Initialized(&mpiInitialized);
    if (!mpiInitialized)
    {
#ifdef NF_REQUIRE_MPI_THREAD_SUPPORT
        int provided;
        MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
#else
        MPI_Init(&argc, &argv);
#endif
    }
#endif
    Kokkos::initialize(argc, argv);

    Logging::setNeonDefaultPattern();
}

inline void finalize()
{
    Logging::info("Finalizing NeoN");
    Kokkos::finalize();
#ifdef NF_WITH_MPI_SUPPORT
    int mpiFinalized = 0;
    MPI_Finalized(&mpiFinalized);
    if (!mpiFinalized)
    {
        MPI_Finalize();
    }
#endif
}

}
