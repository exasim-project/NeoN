// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

// See kokkos_runtime.hpp. Split out of bench/kokkos_spike.cpp, which keeps only
// the feasibility spike (FabView, kokkosSelftest, kokkosMfSum). This TU is
// compiled WITHOUT relocatable device code, like the rest of the module (see
// CMakeLists.txt for why _blockamr itself is also non-RDC).

#include "kokkos_runtime.hpp"

#include <Kokkos_Core.hpp>

#include "launch.hpp"

namespace blockamr::bench
{

void kokkosInitialize()
{
    if (!Kokkos::is_initialized() && !Kokkos::is_finalized())
    {
        Kokkos::initialize(Kokkos::InitializationSettings());
    }
}

void kokkosFinalize()
{
    if (Kokkos::is_initialized())
    {
        // The kokkos_stream backend's execution space instances own cudaStreams
        // (ManageStream::yes), so they have to go before finalize -- and before
        // amrex::Finalize tears the CUDA context down.
        releaseStreamPool();
        Kokkos::finalize();
    }
}

bool kokkosInitialized() { return Kokkos::is_initialized(); }

bool kokkosFinalized() { return Kokkos::is_finalized(); }

} // namespace blockamr::bench
