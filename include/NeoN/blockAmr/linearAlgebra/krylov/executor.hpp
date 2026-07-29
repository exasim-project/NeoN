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
// NeoN::la::ginkgo::getGkoExecutor, which is where the lifetime rules live: it
// memoizes one Ginkgo executor per NeoN executor kind (a per-call executor
// re-inits cuBLAS/cuSPARSE and disturbs the CUDA context at teardown) and
// releases the cache from a Kokkos finalize hook, while the device is alive.
//
// It also threads the Kokkos execution-space stream into Ginkgo rather than
// letting Ginkgo pick its own, which is why blockAMR's AMReX initialisation
// adopts that same stream (see init.cpp): one stream for AMReX, Kokkos and
// Ginkgo means no cross-library synchronisation at the operator boundary.
std::shared_ptr<const gko::Executor> makeExecutor(const NeoN::Executor& executor);

} // namespace blockamr::la
