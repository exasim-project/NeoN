// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoN authors

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main

#include "NeoN/NeoN.hpp"
#include "benchmarks/catch_main.hpp"
#include "test/catch2/executorGenerator.hpp"

TEST_CASE("Vector<scalar>::addition", "[bench]")
{
    auto size = GENERATE(1 << 16, 1 << 17, 1 << 18, 1 << 19, 1 << 20);
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    DYNAMIC_SECTION("" << size)
    {
        NeoN::Vector<NeoN::scalar> cpuA(exec, size);
        NeoN::fill(cpuA, 1.0);
        NeoN::Vector<NeoN::scalar> cpuB(exec, size);
        NeoN::fill(cpuB, 2.0);
        NeoN::Vector<NeoN::scalar> cpuC(exec, size);
        NeoN::fill(cpuC, 0.0);

        BENCHMARK(std::string(execName)) { return (cpuC = cpuA + cpuB); };
    }
}

TEST_CASE("Vector<scalar>::multiplication", "[bench]")
{
    auto size = GENERATE(1 << 16, 1 << 17, 1 << 18, 1 << 19, 1 << 20);

    auto [execName, exec] = GENERATE(allAvailableExecutor());

    DYNAMIC_SECTION("" << size)
    {
        NeoN::Vector<NeoN::scalar> cpuA(exec, size);
        NeoN::fill(cpuA, 1.0);
        NeoN::Vector<NeoN::scalar> cpuB(exec, size);
        NeoN::fill(cpuB, 2.0);
        NeoN::Vector<NeoN::scalar> cpuC(exec, size);
        NeoN::fill(cpuC, 0.0);

        BENCHMARK(std::string(execName)) { return (cpuC = cpuA * cpuB); };
    }
}
