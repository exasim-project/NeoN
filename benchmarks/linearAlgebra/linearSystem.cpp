// SPDX-FileCopyrightText: 2024 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main

#include "NeoN/NeoN.hpp"
#include "benchmarks/catch_main.hpp"
#include "test/catch2/executorGenerator.hpp"

TEMPLATE_TEST_CASE("LinearSystem::1D", "[bench]", NeoN::scalar, NeoN::Vec3)
{
    auto size = GENERATE(1 << 16, 1 << 17, 1 << 18, 1 << 19, 1 << 20);
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    NeoN::UnstructuredMesh mesh = NeoN::create1DUniformMesh(exec, size);

    const std::string sectionName = std::to_string(size);

    DYNAMIC_SECTION(sectionName + " - Construct")
    {
        BENCHMARK(std::string(execName))
        {
            auto ls = NeoN::la::createEmptyLinearSystem<TestType>(mesh);
            return ls.matrix().values().size();
        };
    }

    DYNAMIC_SECTION(sectionName + " - Reset")
    {
        auto ls = la::createEmptyLinearSystem<TestType>(mesh);
        BENCHMARK(std::string(execName)) { ls.reset(); };
    }

    DYNAMIC_SECTION(sectionName + " - CopyToHost")
    {
        auto ls = la::createEmptyLinearSystem<TestType>(mesh);
        BENCHMARK(std::string(execName)) { return ls.copyToHost(); };
    }
}
