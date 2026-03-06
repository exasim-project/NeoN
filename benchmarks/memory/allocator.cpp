// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main
                            //
#include "NeoN/NeoN.hpp"
#include "benchmarks/catch_main.hpp"
#include "test/catch2/executorGenerator.hpp"

TEST_CASE("Vector<Vec3>::defaultAllocator", "[bench]")
{
    auto size = GENERATE(1 << 14, 1 << 15, 1 << 16, 1 << 17, 1 << 18);
    auto exec = NeoN::GPUExecutor();
    auto execName = executorName(exec);

    DYNAMIC_SECTION("" << size)
    {
        BENCHMARK(std::string(execName))
        {
            NeoN::Vector<NeoN::scalar> cpuA(exec, size);
            NeoN::fill(cpuA, 1.0);
            NeoN::Vector<NeoN::scalar> cpuB(exec, size);
            NeoN::fill(cpuB, 2.0);
            NeoN::Vector<NeoN::scalar> cpuC(exec, size);
            NeoN::fill(cpuC, 0.0);
        };
    }
}

#if NF_WITH_UMPIRE

TEST_CASE("Vector<Vec3>::umpireAllocator", "[bench]")
{
    auto size = GENERATE(1 << 14, 1 << 15, 1 << 16, 1 << 17, 1 << 18);
    auto exec = NeoN::createDefaultExecutor();
    auto execName = executorName(exec);

    DYNAMIC_SECTION("" << size)
    {
        BENCHMARK(std::string(execName))
        {
            NeoN::Vector<NeoN::scalar> cpuA(exec, size);
            NeoN::fill(cpuA, 1.0);
            NeoN::Vector<NeoN::scalar> cpuB(exec, size);
            NeoN::fill(cpuB, 2.0);
            NeoN::Vector<NeoN::scalar> cpuC(exec, size);
            NeoN::fill(cpuC, 0.0);
        };
    }
}

TEST_CASE("Vector<Vec3>::umpirePoolAllocator", "[bench]")
{
    auto size = GENERATE(1 << 14, 1 << 15, 1 << 16, 1 << 17, 1 << 18);
    auto exec = NeoN::createDefaultExecutor(std::make_unique<NeoN::UmpireAllocator>());
    auto execName = executorName(exec);
    NeoN::UmpireMempoolHandler::setupUmpirePool(NeoN::memorySpace(exec), 1024 * 1024 * 1024);

    DYNAMIC_SECTION("" << size)
    {
        BENCHMARK(std::string(execName))
        {
            NeoN::Vector<NeoN::scalar> cpuA(exec, size);
            NeoN::fill(cpuA, 1.0);
            NeoN::Vector<NeoN::scalar> cpuB(exec, size);
            NeoN::fill(cpuB, 2.0);
            NeoN::Vector<NeoN::scalar> cpuC(exec, size);
            NeoN::fill(cpuC, 0.0);
        };
    }
    NeoN::UmpireMempoolHandler::destroyUmpirePool(NeoN::memorySpace(exec));
}

#endif
