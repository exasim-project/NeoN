// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "catch2_common.hpp"

#include "NeoN/NeoN.hpp"
#include "NeoN/core/memory/umpire.hpp"
#include "NeoN/linearAlgebra/ginkgo.hpp"

// Kept in its own test binary: getGkoExecutor memoizes one Ginkgo executor per executor kind, so
// the pool has to exist before the first GPU Ginkgo executor of the process is created.
TEST_CASE("Ginkgo GPU executor allocates from the Umpire DEVICE_POOL")
{
    NeoN::UmpireMempoolHandler::setupUmpirePool(NeoN::MemorySpace::GPU, 1 << 20);
    auto pool = NeoN::UmpireMempoolHandler::getUmpirePool(NeoN::MemorySpace::GPU);
    auto gkoExec = NeoN::la::ginkgo::getGkoExecutor(NeoN::GPUExecutor {});
    const auto sizeBefore = pool.getCurrentSize();
    const NeoN::sizeType numElems {1000};

    auto* ptr = gkoExec->alloc<NeoN::scalar>(numElems);
    REQUIRE(pool.getCurrentSize() >= sizeBefore + numElems * sizeof(NeoN::scalar));

    gkoExec->free(ptr);
    REQUIRE(pool.getCurrentSize() == sizeBefore);
}
