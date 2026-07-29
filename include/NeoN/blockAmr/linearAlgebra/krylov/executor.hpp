// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <ginkgo/ginkgo.hpp>

#include "NeoN/core/executor/executor.hpp"

#include <memory>

namespace blockamr::la
{

// The Ginkgo executor backing a NeoN one. Thin forwarder to
// NeoN::la::ginkgo::getGkoExecutor, where the lifetime rules live: one memoized
// Ginkgo executor per NeoN executor kind (a per-call one re-inits cuBLAS/cuSPARSE and
// disturbs the CUDA context at teardown), the cache released from a Kokkos finalize
// hook while the device is still alive. It also threads the Kokkos stream into
// Ginkgo, which is why blockAMR's AMReX init adopts that same stream (init.cpp): one
// stream for AMReX, Kokkos and Ginkgo means no cross-library synchronisation at the
// operator boundary.
std::shared_ptr<const gko::Executor> makeExecutor(const NeoN::Executor& executor);

} // namespace blockamr::la
