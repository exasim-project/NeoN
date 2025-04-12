// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoN authors

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
        NeoN::Field<NeoN::scalar> fieldA(exec, 5);
        NeoN::fill(fieldA, 0.0);
        NeoN::Field<NeoN::scalar> fieldB(exec, 5);
        auto viewA = fieldA.view();
        auto viewB = fieldB.view();
        NeoN::fill(fieldB, 1.0);
        NeoN::parallelFor(
            exec, {0, 5}, KOKKOS_LAMBDA(const size_t i) { viewA[i] = viewB[i] + 2.0; }
        );
        auto hostA = fieldA.copyToHost();
        for (auto value : hostA.view())
        {
            REQUIRE(value == 3.0);
        }
    }

    SECTION("parallelFor_Vector" + execName)
    {
        NeoN::Field<NeoN::Vector> fieldA(exec, 5);
        NeoN::fill(fieldA, NeoN::Vector(0.0, 0.0, 0.0));
        NeoN::Field<NeoN::Vector> fieldB(exec, 5);
        auto viewA = fieldA.view();
        auto viewB = fieldB.view();
        NeoN::fill(fieldB, NeoN::Vector(1.0, 1.0, 1.0));
        NeoN::parallelFor(
            exec,
            {0, 5},
            KOKKOS_LAMBDA(const size_t i) { viewA[i] = viewB[i] + NeoN::Vector(2.0, 2.0, 2.0); }
        );
        auto hostA = fieldA.copyToHost();
        for (auto value : hostA.view())
        {
            REQUIRE(value == NeoN::Vector(3.0, 3.0, 3.0));
        }
    }

    SECTION("parallelFor_Field_" + execName)
    {
        NeoN::Field<NeoN::scalar> fieldA(exec, 5);
        NeoN::fill(fieldA, 0.0);
        NeoN::Field<NeoN::scalar> fieldB(exec, 5);
        auto viewB = fieldB.view();
        NeoN::fill(fieldB, 1.0);
        NeoN::parallelFor(
            fieldA, KOKKOS_LAMBDA(const size_t i) { return viewB[i] + 2.0; }
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
        NeoN::Field<NeoN::scalar> fieldA(exec, 5);
        NeoN::fill(fieldA, 0.0);
        NeoN::Field<NeoN::scalar> fieldB(exec, 5);
        auto viewB = fieldB.view();
        NeoN::fill(fieldB, 1.0);
        NeoN::scalar sum = 0.0;
        NeoN::parallelReduce(
            exec, {0, 5}, KOKKOS_LAMBDA(const size_t i, double& lsum) { lsum += viewB[i]; }, sum
        );

        REQUIRE(sum == 5.0);
    }

    SECTION("parallelReduce_MaxValue" + execName)
    {
        NeoN::Field<NeoN::scalar> fieldA(exec, 5);
        NeoN::fill(fieldA, 0.0);
        NeoN::Field<NeoN::scalar> fieldB(exec, 5);
        auto viewB = fieldB.view();
        NeoN::fill(fieldB, 1.0);
        auto max = std::numeric_limits<NeoN::scalar>::lowest();
        Kokkos::Max<NeoN::scalar> reducer(max);
        NeoN::parallelReduce(
            exec,
            {0, 5},
            KOKKOS_LAMBDA(const size_t i, NeoN::scalar& lmax) {
                if (lmax < viewB[i]) lmax = viewB[i];
            },
            reducer
        );

        REQUIRE(max == 1.0);
    }

    SECTION("parallelReduce_Field_" + execName)
    {
        NeoN::Field<NeoN::scalar> fieldA(exec, 5);
        NeoN::fill(fieldA, 0.0);
        NeoN::Field<NeoN::scalar> fieldB(exec, 5);
        auto viewB = fieldB.view();
        NeoN::fill(fieldB, 1.0);
        NeoN::scalar sum = 0.0;
        NeoN::parallelReduce(
            fieldA, KOKKOS_LAMBDA(const size_t i, double& lsum) { lsum += viewB[i]; }, sum
        );

        REQUIRE(sum == 5.0);
    }

    SECTION("parallelReduce_Field_MaxValue" + execName)
    {
        NeoN::Field<NeoN::scalar> fieldA(exec, 5);
        NeoN::fill(fieldA, 0.0);
        NeoN::Field<NeoN::scalar> fieldB(exec, 5);
        auto viewB = fieldB.view();
        NeoN::fill(fieldB, 1.0);
        auto max = std::numeric_limits<NeoN::scalar>::lowest();
        Kokkos::Max<NeoN::scalar> reducer(max);
        NeoN::parallelReduce(
            fieldA,
            KOKKOS_LAMBDA(const size_t i, NeoN::scalar& lmax) {
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
        NeoN::Field<NeoN::localIdx> intervals(exec, {1, 2, 3, 4, 5});
        NeoN::Field<NeoN::localIdx> segments(exec, intervals.size() + 1, 0);
        auto segView = segments.view();
        const auto intView = intervals.view();

        NeoN::parallelScan(
            exec,
            {1, segView.size()},
            KOKKOS_LAMBDA(const std::size_t i, NeoN::localIdx& update, const bool final) {
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
        NeoN::Field<NeoN::localIdx> intervals(exec, {1, 2, 3, 4, 5});
        NeoN::Field<NeoN::localIdx> segments(exec, intervals.size() + 1, 0);
        auto segView = segments.view();
        const auto intView = intervals.view();
        NeoN::localIdx finalValue = 0;

        NeoN::parallelScan(
            exec,
            {1, segView.size()},
            KOKKOS_LAMBDA(const std::size_t i, NeoN::localIdx& update, const bool final) {
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
