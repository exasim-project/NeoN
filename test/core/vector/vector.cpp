// SPDX-FileCopyrightText: 2023 - 2025 NeoN authors
//
// SPDX-License-Identifier: MIT

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main
#include "catch2_common.hpp"

#include "NeoN/NeoN.hpp"

TEST_CASE("Vector Constructors")
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    SECTION("Copy Constructor " + execName)
    {
        NeoN::localIdx size = 10;
        NeoN::Vector<NeoN::scalar> a(exec, size);
        NeoN::fill(a, 5.0);
        NeoN::Vector<NeoN::scalar> b(a);

        REQUIRE(b.size() == size);

        auto hostB = b.copyToHost();
        for (auto value : hostB.view())
        {
            REQUIRE(value == 5.0);
        }

        NeoN::Vector<NeoN::scalar> initWith5(exec, size, 5.0);
        REQUIRE(initWith5.size() == size);

        auto hostInitWith5 = initWith5.copyToHost();
        for (auto value : hostInitWith5.view())
        {
            REQUIRE(value == 5.0);
        }
    }

    SECTION("Initialiser List Constructor " + execName)
    {
        NeoN::Vector<NeoN::label> a(exec, {1, 2, 3});
        auto hostA = a.copyToHost();
        REQUIRE(hostA.data()[0] == 1);
        REQUIRE(hostA.data()[1] == 2);
        REQUIRE(hostA.data()[2] == 3);
    }

    SECTION("Cross Exec Constructor " + execName)
    {

        NeoN::Vector<NeoN::label> a(exec, {1, 2, 3});
        NeoN::Vector<NeoN::label> b(a);

        auto hostB = b.copyToHost();

        REQUIRE(hostB.data()[0] == 1);
        REQUIRE(hostB.data()[1] == 2);
        REQUIRE(hostB.data()[2] == 3);
    }
}


TEST_CASE("Vector Operator Overloads")
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    SECTION("Vector Operator+= " + execName)
    {
        NeoN::localIdx size = 10;
        NeoN::Vector<NeoN::scalar> a(exec, size);
        NeoN::Vector<NeoN::scalar> b(exec, size);
        NeoN::fill(a, 5.0);
        NeoN::fill(b, 10.0);

        a += b;

        auto hostViewA = a.copyToHost();
        for (auto value : hostViewA.view())
        {
            REQUIRE(value == 15.0);
        }
    }

    SECTION("Vector Operator-= " + execName)
    {
        NeoN::localIdx size = 10;
        NeoN::Vector<NeoN::scalar> a(exec, size);
        NeoN::Vector<NeoN::scalar> b(exec, size);
        NeoN::fill(a, 5.0);
        NeoN::fill(b, 10.0);

        a -= b;

        auto hostA = a.copyToHost();
        for (auto value : hostA.view())
        {
            REQUIRE(value == -5.0);
        }
    }

    SECTION("Vector Operator+ " + execName)
    {
        NeoN::localIdx size = 10;
        NeoN::Vector<NeoN::scalar> a(exec, size);
        NeoN::Vector<NeoN::scalar> b(exec, size);
        NeoN::Vector<NeoN::scalar> c(exec, size);
        NeoN::fill(a, 5.0);
        NeoN::fill(b, 10.0);

        c = a + b;
        auto hostC = c.copyToHost();
        for (auto value : hostC.view())
        {
            REQUIRE(value == 15.0);
        }
    }

    SECTION("Vector Operator-" + execName)
    {
        NeoN::localIdx size = 10;
        NeoN::Vector<NeoN::scalar> a(exec, size);
        NeoN::Vector<NeoN::scalar> b(exec, size);
        NeoN::Vector<NeoN::scalar> c(exec, size);
        NeoN::fill(a, 5.0);
        NeoN::fill(b, 10.0);

        c = a - b;

        auto hostC = c.copyToHost();
        for (auto value : hostC.view())
        {
            REQUIRE(value == -5.0);
        }
    }

    SECTION("Vector Operator*=" + execName)
    {
        NeoN::localIdx size = 10;
        NeoN::Vector<NeoN::Vec3> a(exec, size);
        NeoN::fill(a, NeoN::Vec3 {5.0, 10.0, 15.0});

        a *= 2;

        auto hostA = a.copyToHost();
        for (auto value : hostA.view())
        {
            REQUIRE(value[0] == 10.0);
            REQUIRE(value[1] == 20.0);
            REQUIRE(value[2] == 30.0);
        }
    }
}

TEST_CASE("Vector Container Operations")
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    SECTION("empty, size, range" + execName)
    {
        NeoN::localIdx size = 10;
        NeoN::Vector<NeoN::scalar> a(exec, 0);
        NeoN::Vector<NeoN::scalar> b(exec, size);
        REQUIRE(a.empty() == true);
        REQUIRE(a.size() == 0);
        REQUIRE(a.range().first == 0);
        REQUIRE(a.range().second == 0);
        REQUIRE(b.empty() == false);
        REQUIRE(b.size() == size);
        REQUIRE(b.range().first == 0);
        REQUIRE(b.range().second == size);
    };

    SECTION("view" + execName)
    {
        NeoN::Vector<NeoN::label> a(exec, {1, 2, 3});
        auto hostA = a.copyToHost();

        auto view = hostA.view();
        REQUIRE(view[0] == 1);
        REQUIRE(view[1] == 2);
        REQUIRE(view[2] == 3);

        auto subview = hostA.view({1, 3});
        REQUIRE(subview[0] == 2);
        REQUIRE(subview[1] == 3);
    }

    SECTION("viewVec3" + execName)
    {
        NeoN::Vector<NeoN::Vec3> a(exec, {{1, 1, 1}, {2, 2, 2}, {3, 3, 3}});
        auto hostA = a.copyToHost();

        auto view = hostA.view();
        REQUIRE(view[0] == NeoN::Vec3(1, 1, 1));
        REQUIRE(view[1] == NeoN::Vec3(2, 2, 2));
        REQUIRE(view[2] == NeoN::Vec3(3, 3, 3));

        auto subview = hostA.view({1, 3});
        REQUIRE(subview[0] == NeoN::Vec3(2, 2, 2));
        REQUIRE(subview[1] == NeoN::Vec3(3, 3, 3));
    }

    SECTION("copyToHost " + execName)
    {

        NeoN::Vector<NeoN::label> a(exec, {1, 2, 3});

        auto hostA = a.copyToHost();
        auto hostB = a.copyToHost();

        REQUIRE(&(hostA.data()[0]) != &(hostB.data()[0]));
        REQUIRE(hostA.data()[1] == hostB.data()[1]);
    }
}

TEST_CASE("Vector Operations")
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    SECTION("Vector_" + execName)
    {
        NeoN::localIdx size = 10;
        NeoN::Vector<NeoN::scalar> a(exec, size);
        NeoN::fill(a, 5.0);

        REQUIRE(equal(a, 5.0));

        NeoN::Vector<NeoN::scalar> b(exec, size + 2);
        NeoN::fill(b, 10.0);

        a = b;
        REQUIRE(a.view().size() == size + 2);
        REQUIRE(equal(a, b));

        add(a, b);
        REQUIRE(a.view().size() == size + 2);
        REQUIRE(equal(a, 20.0));

        a = a + b;
        REQUIRE(equal(a, 30.0));

        a = a - b;
        REQUIRE(equal(a, 20.0));

        a = a * 0.1;
        REQUIRE(equal(a, 2.0));

        a = a * b;
        REQUIRE(equal(a, 20.0));

        auto sB = b.view();
        a.apply(KOKKOS_LAMBDA(const NeoN::localIdx i) { return 2 * sB[i]; });
        REQUIRE(equal(a, 20.0));
    }
}

TEST_CASE("getViews")
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    NeoN::Vector<NeoN::scalar> a(exec, 3, 1.0);
    NeoN::Vector<NeoN::scalar> b(exec, 3, 2.0);
    NeoN::Vector<NeoN::scalar> c(exec, 3, 3.0);

    auto [hostA, hostB, hostC] = NeoN::copyToHosts(a, b, c);
    auto [viewB, viewC] = NeoN::views(b, c);

    REQUIRE(hostA.view()[0] == 1.0);
    REQUIRE(hostB.view()[0] == 2.0);
    REQUIRE(hostC.view()[0] == 3.0);

    NeoN::parallelFor(
        a, KOKKOS_LAMBDA(const NeoN::localIdx i) { return viewB[i] + viewC[i]; }
    );

    auto hostD = a.copyToHost();

    for (auto value : hostD.view())
    {
        REQUIRE(value == 5.0);
    }
}
