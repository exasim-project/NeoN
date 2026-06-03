// SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#define CATCH_CONFIG_RUNNER
#include "catch2_common.hpp"

#include "NeoN/core/memory/umpire.hpp"

#if NF_WITH_UMPIRE
#include "umpire/Allocator.hpp"
#endif

#include <cstddef>

// Reporting guard for the Umpire pool: the QuickPool keeps every block it grows to for the
// whole run, so the reserved footprint (actualSize) is the real device-memory ceiling and can
// sit far above the in-use bytes (currentSize). UmpireMempoolHandler exposes these so a run can
// log/inspect that ceiling; they do not (and cannot) shrink it. Exercised on the HOST pool so it
// runs without a GPU.
TEST_CASE("Umpire host pool reporting tracks reserved vs in-use")
{
#if NF_WITH_UMPIRE
    using NeoN::MemorySpace;
    using NeoN::UmpireMempoolHandler;

    constexpr MemorySpace host = MemorySpace::CPU;
    constexpr std::size_t bigAlloc = 64u * 1024u * 1024u; // 64 MiB

    // Small initial block so the large allocation below forces observable pool growth.
    UmpireMempoolHandler::setupUmpirePool(host, 64);

    auto pool = UmpireMempoolHandler::getUmpirePool(host);
    void* ptr = pool.allocate(bigAlloc);

    // While live: in-use, reserved and high-water all reflect the allocation.
    REQUIRE(UmpireMempoolHandler::currentSize(host) >= bigAlloc);
    REQUIRE(UmpireMempoolHandler::actualSize(host) >= bigAlloc);
    REQUIRE(UmpireMempoolHandler::highWatermark(host) >= bigAlloc);

    pool.deallocate(ptr);

    // After the free the block stays reserved in the pool: in-use drops to zero but reserved
    // (the ceiling) and high-water do not. This gap is exactly what the reporting surfaces.
    REQUIRE(UmpireMempoolHandler::currentSize(host) == 0);
    REQUIRE(UmpireMempoolHandler::actualSize(host) >= bigAlloc);
    REQUIRE(UmpireMempoolHandler::highWatermark(host) >= bigAlloc);
#else
    SUCCEED("NeoN built without Umpire; mempool reporting is a no-op");
#endif
}
