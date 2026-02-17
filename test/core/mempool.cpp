// SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main
#include "catch2_common.hpp"

#include "NeoN/NeoN.hpp"

TEST_CASE("Mempool")
{
    // auto exec = NeoN::createDefaultExecutor();
    // // auto exec = NeoN::GPUExecutor();
    // auto execName = executorName(exec);
    // using MemorySpace = NeoN::MemorySpaceMapping<decltype(exec)>::MemorySpace;

    // SECTION("Can construct mempool " + execName)
    // {
    //     auto& mempool = NeoN::Mempool<MemorySpace>::instance<MemorySpace>();
    //     mempool.setSize(0);
    // }

    // SECTION("Can set size " + execName)
    // {
    //     auto& mempool = NeoN::Mempool<MemorySpace>::instance<MemorySpace>();
    //     mempool.setSize(42);
    // }

    // SECTION("Mempool is singleton " + execName)
    // {
    //     auto& mempool = NeoN::Mempool<MemorySpace>::instance<MemorySpace>();
    //     REQUIRE(mempool.getSize() == 42);
    // }

    // SECTION("Can allocate and free with mempool " + execName)
    // {
    //     auto& mempool = NeoN::Mempool<MemorySpace>::instance<MemorySpace>();
    //     int numElems {5};
    //     int* ptr = reinterpret_cast<int*>(mempool.alloc(numElems * sizeof(int)));
    //     mempool.free<int>(ptr);
    // }

    // SECTION("Can get mempool statistics " + execName)
    // {
    //     auto& mempool = NeoN::Mempool<MemorySpace>::instance<MemorySpace>();
    //     auto stats = mempool.getStatistics();
    //     REQUIRE_FALSE(stats.capacity_bytes == 0);
    //     REQUIRE(stats.capacity_bytes >= 42);
    // }
}
