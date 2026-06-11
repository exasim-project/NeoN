// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main
                            //
#include "NeoN/NeoN.hpp"
#include "benchmarks/catch_main.hpp"
#include "test/catch2/executorGenerator.hpp"

TEMPLATE_TEST_CASE(
    "Allocator::1D", "[bench]", NeoN::Vec3
) //"Template" for consistency with other benchmarks.
{
    auto size = GENERATE(1 << 14, 1 << 15, 1 << 16, 1 << 17, 1 << 18);

    const std::string sectionName = std::to_string(size);

    DYNAMIC_SECTION(sectionName + " - Default")
    {
        auto exec = NeoN::GPUExecutor();
        BENCHMARK(executorName(exec))
        {
            NeoN::Vector<TestType> cpuA(exec, size);
            NeoN::fill(cpuA, NeoN::one<TestType>());
            NeoN::Vector<TestType> cpuB(exec, size);
            NeoN::fill(cpuB, 2 * NeoN::one<TestType>());
            NeoN::Vector<TestType> cpuC(exec, size);
            NeoN::fill(cpuC, NeoN::zero<TestType>());
        };
    }

#if NF_WITH_UMPIRE
    DYNAMIC_SECTION(sectionName + " - Umpire")
    {
        auto exec = NeoN::createDefaultExecutor();
        BENCHMARK(executorName(exec))
        {
            NeoN::Vector<TestType> cpuA(exec, size);
            NeoN::fill(cpuA, NeoN::one<TestType>());
            NeoN::Vector<TestType> cpuB(exec, size);
            NeoN::fill(cpuB, 2 * NeoN::one<TestType>());
            NeoN::Vector<TestType> cpuC(exec, size);
            NeoN::fill(cpuC, NeoN::zero<TestType>());
        };
    }

    DYNAMIC_SECTION(sectionName + " - UmpirePool")
    {
        auto exec = NeoN::createDefaultExecutor(std::make_unique<NeoN::UmpireAllocator>());
        NeoN::UmpireMempoolHandler::setupUmpirePool(NeoN::memorySpace(exec), 1024 * 1024 * 1024);
        BENCHMARK(executorName(exec))
        {
            NeoN::Vector<TestType> cpuA(exec, size);
            NeoN::fill(cpuA, NeoN::one<TestType>());
            NeoN::Vector<TestType> cpuB(exec, size);
            NeoN::fill(cpuB, 2 * NeoN::one<TestType>());
            NeoN::Vector<TestType> cpuC(exec, size);
            NeoN::fill(cpuC, NeoN::zero<TestType>());
        };
        NeoN::UmpireMempoolHandler::destroyUmpirePool(NeoN::memorySpace(exec));
    }
#endif
}
