// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_adapters.hpp>

#include "NeoN/core/executor/executor.hpp"
#include "NeoN/core/memory/umpire.hpp"
#include "NeoN/core/memory/kokkos.hpp"

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


TEST_CASE("Can use CPUExecutor wo mempool")
{
    auto exec = NeoN::CPUExecutor();
    int numElems {5};
    int* ptr = exec.template alloc<int>(numElems);
    exec.free(ptr);
}


TEST_CASE("Can use DefaultExecutor wo mempool")
{
    auto exec = NeoN::createDefaultExecutor();
    int numElems {5};
    std::visit(
        [=](auto concreteExec)
        {
            int* ptr = concreteExec.template alloc<int>(numElems);
            concreteExec.free(ptr);
        },
        exec
    );
}

#if NF_WITH_UMPIRE

TEST_CASE("Can use SerialExecutor with UmpireAllocator")
{
    auto exec = NeoN::SerialExecutor(std::make_unique<NeoN::UmpireAllocator>());
    int numElems {5};
    int* ptr = exec.template alloc<int>(numElems);
    exec.free(ptr);
}


TEST_CASE("Can use DefaultExecutor with UmpireAllocator")
{
    auto exec = NeoN::createDefaultExecutor(std::make_unique<NeoN::UmpireAllocator>());
    int numElems {5};
    std::visit(
        [=](auto concreteExec)
        {
            int* ptr = concreteExec.template alloc<int>(numElems);
            concreteExec.free(ptr);
        },
        exec
    );
}


TEST_CASE("Can use SerialExecutor with UmpirePoolAllocator")
{
    auto exec = NeoN::SerialExecutor(std::make_unique<NeoN::UmpirePoolAllocator>());
    NeoN::UmpireMempoolHandler::setupUmpirePool(NeoN::MemorySpace::CPU, 1024);

    int numElems {5};
    int* ptr = exec.template alloc<int>(numElems);
    exec.free(ptr);

    NeoN::UmpireMempoolHandler::destroyUmpirePool(NeoN::MemorySpace::CPU);
}


TEST_CASE("Can use DefaultExecutor with UmpirePoolAllocator")
{
    auto exec = NeoN::createDefaultExecutor(std::make_unique<NeoN::UmpirePoolAllocator>());
    NeoN::UmpireMempoolHandler::setupUmpirePool(NeoN::memorySpace(exec), 1024);

    int numElems {5};
    std::visit(
        [=](auto concreteExec)
        {
            int* ptr = concreteExec.template alloc<int>(numElems);
            concreteExec.free(ptr);
        },
        exec
    );

    NeoN::UmpireMempoolHandler::destroyUmpirePool(NeoN::memorySpace(exec));
}

#endif
