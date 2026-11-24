// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

// See runtime.hpp. Compiled WITHOUT relocatable device code, like the rest of the module
// (CMakeLists.txt has the rationale).

#include "NeoN/blockAmr/core/runtime.hpp"

#include <Kokkos_Core.hpp>

#include "NeoN/blockAmr/core/launch.hpp"

namespace blockamr
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
        // The kokkos_stream backend's space instances own cudaStreams (ManageStream::yes), so
        // they have to go before finalize and before amrex::Finalize drops the CUDA context.
        releaseStreamPool();
        Kokkos::finalize();
    }
}

bool kokkosInitialized() { return Kokkos::is_initialized(); }

bool kokkosFinalized() { return Kokkos::is_finalized(); }

} // namespace blockamr
