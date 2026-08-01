// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

// Kokkos lifetime, driven from blockamr.initialize()/finalize() so its ordering against
// amrex::Initialize/Finalize is enforced in one place.

namespace blockamr
{

void kokkosInitialize();
void kokkosFinalize();
bool kokkosInitialized();
bool kokkosFinalized();

} // namespace blockamr
