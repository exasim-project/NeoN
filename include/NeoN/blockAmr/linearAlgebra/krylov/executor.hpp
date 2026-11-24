// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <ginkgo/ginkgo.hpp>

#include "NeoN/core/executor/executor.hpp"

#include <memory>

namespace blockamr::la
{

// The Ginkgo executor backing a NeoN one; a thin forwarder to
// NeoN::la::ginkgo::getGkoExecutor, which memoizes one per NeoN executor kind and frees
// the cache from a Kokkos finalize hook. It threads the Kokkos stream in, as AMReX does.
std::shared_ptr<const gko::Executor> makeExecutor(const NeoN::Executor& executor);

} // namespace blockamr::la
