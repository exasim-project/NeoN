// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoN authors

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_adapters.hpp>

#include "NeoN/core/executor/executor.hpp"

TEST_CASE("Executor Equality")
{
    NeoN::Executor cpuExec0(NeoN::SerialExecutor {});
    NeoN::Executor ompExec0(NeoN::CPUExecutor {});
    NeoN::Executor gpuExec0(NeoN::GPUExecutor {});

    NeoN::Executor cpuExec1(NeoN::SerialExecutor {});
    NeoN::Executor ompExec1(NeoN::CPUExecutor {});
    NeoN::Executor gpuExec1(NeoN::GPUExecutor {});

    REQUIRE(cpuExec0 == cpuExec1);
    REQUIRE(cpuExec0 != ompExec1);
    REQUIRE(cpuExec0 != gpuExec1);

    REQUIRE(ompExec0 != cpuExec1);
    REQUIRE(ompExec0 == ompExec1);
    REQUIRE(ompExec0 != gpuExec1);

    REQUIRE(gpuExec0 != cpuExec1);
    REQUIRE(gpuExec0 != ompExec1);
    REQUIRE(gpuExec0 == gpuExec1);
}
