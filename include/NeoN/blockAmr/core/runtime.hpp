// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

// Kokkos lifetime, driven from blockamr.initialize()/finalize() so the ordering
// against amrex::Initialize/Finalize is enforced in one place. Split out of
// bench/kokkosSpike.cpp so production code need not include a bench-only header.

namespace blockamr
{

void kokkosInitialize();
void kokkosFinalize();
bool kokkosInitialized();
bool kokkosFinalized();

} // namespace blockamr
