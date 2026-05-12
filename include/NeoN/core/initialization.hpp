// SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include "NeoN/core/logging.hpp"
#include "NeoN/core/mpi/environment.hpp"

#include <Kokkos_Core.hpp>
#include <chrono>

#include <cpptrace/cpptrace.hpp>

namespace NeoN
{

inline void initialize(int argc, char* argv[])
{
    mpi::Environment mpiEnv;

    // Bind each MPI rank to a distinct GPU on multi-GPU nodes.
    //
    // Without explicit binding every rank defaults to device 0, leaving all
    // other GPUs idle and causing CUDA/HIP context contention on GPU 0.
    // We use a shared-memory sub-communicator (MPI_COMM_TYPE_SHARED) to obtain
    // the local rank within the node, then map:
    //   device_id = local_rank % num_devices
    //
    // Skipped when:
    //   - MPI is not initialised (serial run or pre-MPI test harness), OR
    //   - only one device is visible (single-GPU node or CPU-only build) —
    //     Kokkos::num_devices() returns 0 for CPU-only backends.
    //   - KOKKOS_DEVICE_ID or --kokkos-device-id is already set by the user,
    //     in which case Kokkos::initialize(argc, argv) below will honour it.
#if defined(NF_WITH_MPI_SUPPORT)
    if (mpiEnv.isInitialized() && Kokkos::num_devices() > 1
        && std::getenv("KOKKOS_DEVICE_ID") == nullptr)
    {
        MPI_Comm nodeComm;
        MPI_Comm_split_type(
            mpiEnv.comm(),
            MPI_COMM_TYPE_SHARED,
            static_cast<int>(mpiEnv.rank()),
            MPI_INFO_NULL,
            &nodeComm
        );
        int localRank = 0;
        MPI_Comm_rank(nodeComm, &localRank);
        MPI_Comm_free(&nodeComm);

        Kokkos::InitializationSettings settings;
        settings.set_device_id(localRank % Kokkos::num_devices());
        Kokkos::initialize(settings);
    }
    else
#endif
    {
        Kokkos::initialize(argc, argv);
    }

    cpptrace::register_terminate_handler();
    Logging::setNeonDefaultPattern(mpiEnv);
}

inline void finalize()
{
    Logging::info("Finalizing NeoN");
    Kokkos::finalize();
}
}
