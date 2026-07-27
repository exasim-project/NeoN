// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

// ---------------------------------------------------------------------------
// Kokkos lifetime, driven from blockamr.initialize()/finalize() so the ordering
// against amrex::Initialize/Finalize is enforced in one place. Split out of
// bench/kokkos_spike.cpp so that production code (init.cpp) does not have to
// include a bench-only header (kokkos_bench.hpp) to reach four entry points it
// actually depends on.
// ---------------------------------------------------------------------------

namespace blockamr::bench
{

void kokkosInitialize();
void kokkosFinalize();
bool kokkosInitialized();
bool kokkosFinalized();

} // namespace blockamr::bench
