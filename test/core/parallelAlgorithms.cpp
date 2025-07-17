// SPDX-FileCopyrightText: 2023 - 2025 NeoN authors
//
// SPDX-License-Identifier: MIT

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main
#include "catch2_common.hpp"

#include <Kokkos_Core.hpp>

#include "NeoN/NeoN.hpp"

#include <limits>

TEST_CASE("parallelFor")
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    SECTION("parallelFor_" + execName)
    {
        NeoN::Vector<NeoN::scalar> fieldA(exec, 5);
        NeoN::fill(fieldA, 0.0);
        NeoN::Vector<NeoN::scalar> fieldB(exec, 5);
        auto viewA = fieldA.view();
        auto viewB = fieldB.view();
        NeoN::fill(fieldB, 1.0);
        NeoN::parallelFor(
            exec, {0, 5}, KOKKOS_LAMBDA(const NeoN::localIdx i) { viewA[i] = viewB[i] + 2.0; }
        );
        auto hostA = fieldA.copyToHost();
        for (auto value : hostA.view())
        {
            REQUIRE(value == 3.0);
        }
    }

    SECTION("parallelFor_Vec3" + execName)
    {
        NeoN::Vector<NeoN::Vec3> fieldA(exec, 5);
        NeoN::fill(fieldA, NeoN::Vec3(0.0, 0.0, 0.0));
        NeoN::Vector<NeoN::Vec3> fieldB(exec, 5);
        auto viewA = fieldA.view();
        auto viewB = fieldB.view();
        NeoN::fill(fieldB, NeoN::Vec3(1.0, 1.0, 1.0));
        NeoN::parallelFor(
            exec,
            {0, 5},
            KOKKOS_LAMBDA(const NeoN::localIdx i) {
                viewA[i] = viewB[i] + NeoN::Vec3(2.0, 2.0, 2.0);
            }
        );
        auto hostA = fieldA.copyToHost();
        for (auto value : hostA.view())
        {
            REQUIRE(value == NeoN::Vec3(3.0, 3.0, 3.0));
        }
    }

    SECTION("parallelFor_Vector_" + execName)
    {
        NeoN::Vector<NeoN::scalar> fieldA(exec, 5);
        NeoN::fill(fieldA, 0.0);
        NeoN::Vector<NeoN::scalar> fieldB(exec, 5);
        auto viewB = fieldB.view();
        NeoN::fill(fieldB, 1.0);
        NeoN::parallelFor(
            fieldA, KOKKOS_LAMBDA(const NeoN::localIdx i) { return viewB[i] + 2.0; }
        );
        auto hostA = fieldA.copyToHost();
        for (auto value : hostA.view())
        {
            REQUIRE(value == 3.0);
        }
    }
};


TEST_CASE("parallelReduce")
{
    NeoN::Executor exec = GENERATE(
        NeoN::Executor(NeoN::SerialExecutor {}),
        NeoN::Executor(NeoN::CPUExecutor {}),
        NeoN::Executor(NeoN::GPUExecutor {})
    );
    std::string execName = std::visit([](auto e) { return e.name(); }, exec);

    SECTION("parallelReduce_" + execName)
    {
        NeoN::Vector<NeoN::scalar> fieldA(exec, 5);
        NeoN::fill(fieldA, 0.0);
        NeoN::Vector<NeoN::scalar> fieldB(exec, 5);
        auto viewB = fieldB.view();
        NeoN::fill(fieldB, 1.0);
        NeoN::scalar sum = 0.0;
        NeoN::parallelReduce(
            exec,
            {0, 5},
            KOKKOS_LAMBDA(const NeoN::localIdx i, double& lsum) { lsum += viewB[i]; },
            sum
        );

        REQUIRE(sum == 5.0);
    }

    SECTION("parallelReduce_MaxValue" + execName)
    {
        NeoN::Vector<NeoN::scalar> fieldA(exec, 5);
        NeoN::fill(fieldA, 0.0);
        NeoN::Vector<NeoN::scalar> fieldB(exec, 5);
        auto viewB = fieldB.view();
        NeoN::fill(fieldB, 1.0);
        auto max = std::numeric_limits<NeoN::scalar>::lowest();
        Kokkos::Max<NeoN::scalar> reducer(max);
        NeoN::parallelReduce(
            exec,
            {0, 5},
            KOKKOS_LAMBDA(const NeoN::localIdx i, NeoN::scalar& lmax) {
                if (lmax < viewB[i]) lmax = viewB[i];
            },
            reducer
        );

        REQUIRE(max == 1.0);
    }

    SECTION("parallelReduce_Vector_" + execName)
    {
        NeoN::Vector<NeoN::scalar> fieldA(exec, 5);
        NeoN::fill(fieldA, 0.0);
        NeoN::Vector<NeoN::scalar> fieldB(exec, 5);
        auto viewB = fieldB.view();
        NeoN::fill(fieldB, 1.0);
        NeoN::scalar sum = 0.0;
        NeoN::parallelReduce(
            fieldA, KOKKOS_LAMBDA(const NeoN::localIdx i, double& lsum) { lsum += viewB[i]; }, sum
        );

        REQUIRE(sum == 5.0);
    }

    SECTION("parallelReduce_Vector_MaxValue" + execName)
    {
        NeoN::Vector<NeoN::scalar> fieldA(exec, 5);
        NeoN::fill(fieldA, 0.0);
        NeoN::Vector<NeoN::scalar> fieldB(exec, 5);
        auto viewB = fieldB.view();
        NeoN::fill(fieldB, 1.0);
        auto max = std::numeric_limits<NeoN::scalar>::lowest();
        Kokkos::Max<NeoN::scalar> reducer(max);
        NeoN::parallelReduce(
            fieldA,
            KOKKOS_LAMBDA(const NeoN::localIdx i, NeoN::scalar& lmax) {
                if (lmax < viewB[i]) lmax = viewB[i];
            },
            reducer
        );

        REQUIRE(max == 1.0);
    }
};

TEST_CASE("parallelScan")
{
    NeoN::Executor exec = GENERATE(NeoN::Executor(NeoN::SerialExecutor {})
                                   // NeoN::Executor(NeoN::CPUExecutor {}),
                                   // NeoN::Executor(NeoN::GPUExecutor {})
    );
    std::string execName = std::visit([](auto e) { return e.name(); }, exec);


    SECTION("parallelScan_withoutReturn" + execName)
    {
        NeoN::Vector<NeoN::localIdx> intervals(exec, {1, 2, 3, 4, 5});
        NeoN::Vector<NeoN::localIdx> segments(exec, intervals.size() + 1, 0);
        auto segView = segments.view();
        const auto intView = intervals.view();

        NeoN::parallelScan(
            exec,
            {1, segView.size()},
            KOKKOS_LAMBDA(const NeoN::localIdx i, NeoN::localIdx& update, const bool final) {
                update += intView[i - 1];
                if (final)
                {
                    segView[i] = update;
                }
            }
        );

        auto hostSegments = segments.copyToHost();
        REQUIRE(hostSegments.view()[0] == 0);
        REQUIRE(hostSegments.view()[1] == 1);
        REQUIRE(hostSegments.view()[2] == 3);
        REQUIRE(hostSegments.view()[3] == 6);
        REQUIRE(hostSegments.view()[4] == 10);
        REQUIRE(hostSegments.view()[5] == 15);

        auto hostIntervals = intervals.copyToHost();
        REQUIRE(hostIntervals.view()[0] == 1);
        REQUIRE(hostIntervals.view()[1] == 2);
        REQUIRE(hostIntervals.view()[2] == 3);
        REQUIRE(hostIntervals.view()[3] == 4);
        REQUIRE(hostIntervals.view()[4] == 5);
    }

    SECTION("parallelScan_withReturn" + execName)
    {
        NeoN::Vector<NeoN::localIdx> intervals(exec, {1, 2, 3, 4, 5});
        NeoN::Vector<NeoN::localIdx> segments(exec, intervals.size() + 1, 0);
        auto segView = segments.view();
        const auto intView = intervals.view();
        NeoN::localIdx finalValue = 0;

        NeoN::parallelScan(
            exec,
            {1, segView.size()},
            KOKKOS_LAMBDA(const NeoN::localIdx i, NeoN::localIdx& update, const bool final) {
                update += intView[i - 1];
                if (final)
                {
                    segView[i] = update;
                }
            },
            finalValue
        );

        REQUIRE(finalValue == 15);

        auto hostSegments = segments.copyToHost();
        REQUIRE(hostSegments.view()[0] == 0);
        REQUIRE(hostSegments.view()[1] == 1);
        REQUIRE(hostSegments.view()[2] == 3);
        REQUIRE(hostSegments.view()[3] == 6);
        REQUIRE(hostSegments.view()[4] == 10);
        REQUIRE(hostSegments.view()[5] == 15);

        auto hostIntervals = intervals.copyToHost();
        REQUIRE(hostIntervals.view()[0] == 1);
        REQUIRE(hostIntervals.view()[1] == 2);
        REQUIRE(hostIntervals.view()[2] == 3);
        REQUIRE(hostIntervals.view()[3] == 4);
        REQUIRE(hostIntervals.view()[4] == 5);
    }
};
