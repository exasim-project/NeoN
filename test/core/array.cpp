// SPDX-FileCopyrightText: 2023 - 2025 NeoN authors
//
// SPDX-License-Identifier: MIT

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main
#include "catch2_common.hpp"

#include "NeoN/NeoN.hpp"

TEST_CASE("Array Constructors")
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    SECTION("Empty Constructor " + execName)
    {
        NeoN::localIdx size = 10;
        NeoN::Array<NeoN::scalar> arrayA(exec, size);
        REQUIRE(arrayA.size() == size);
    }

    SECTION("Single Value Constructor " + execName)
    {
        NeoN::localIdx size = 10;
        NeoN::scalar value = 5.0;
        NeoN::Array<NeoN::scalar> arrayA(exec, size, value);
        REQUIRE(arrayA.size() == size);
        auto hostA = arrayA.copyToHost();
        for (auto check : hostA.view())
        {
            REQUIRE(check == value);
        }
    }

    SECTION("Copy Constructor " + execName)
    {
        NeoN::localIdx size = 10;
        NeoN::Array<NeoN::scalar> arrayA(exec, size);
        NeoN::fill(arrayA, 5.0);
        NeoN::Array<NeoN::scalar> arrayB(arrayA);

        REQUIRE(arrayB.size() == size);

        auto hostB = arrayB.copyToHost();
        for (auto value : hostB.view())
        {
            REQUIRE(value == 5.0);
        }

        NeoN::Array<NeoN::scalar> initWith5(exec, size, 5.0);
        REQUIRE(initWith5.size() == size);

        auto hostInitWith5 = initWith5.copyToHost();
        for (auto value : hostInitWith5.view())
        {
            REQUIRE(value == 5.0);
        }
    }

    SECTION("Initialiser List Constructor " + execName)
    {
        NeoN::Array<NeoN::label> arrayA(exec, {1, 2, 3});
        auto hostA = arrayA.copyToHost();
        REQUIRE(hostA.data()[0] == 1);
        REQUIRE(hostA.data()[1] == 2);
        REQUIRE(hostA.data()[2] == 3);
    }
}


TEST_CASE("Array Container Operations")
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    SECTION("empty, size, range" + execName)
    {
        NeoN::localIdx size = 10;
        NeoN::Array<NeoN::scalar> arrayA(exec, 0);
        NeoN::Array<NeoN::scalar> arrayB(exec, size);
        REQUIRE(arrayA.empty() == true);
        REQUIRE(arrayA.size() == 0);
        REQUIRE(arrayA.range().first == 0);
        REQUIRE(arrayA.range().second == 0);
        REQUIRE(arrayB.empty() == false);
        REQUIRE(arrayB.size() == size);
        REQUIRE(arrayB.range().first == 0);
        REQUIRE(arrayB.range().second == size);
    };

    SECTION("view" + execName)
    {
        NeoN::Array<NeoN::label> a(exec, {1, 2, 3});
        auto hostA = a.copyToHost();

        auto view = hostA.view();
        REQUIRE(view[0] == 1);
        REQUIRE(view[1] == 2);
        REQUIRE(view[2] == 3);

        auto subview = hostA.view({1, 3});
        REQUIRE(subview[0] == 2);
        REQUIRE(subview[1] == 3);
    }

    SECTION("copyToHost " + execName)
    {
        NeoN::Array<NeoN::label> arrayA(exec, {1, 2, 3});

        auto hostA1 = arrayA.copyToHost();
        auto hostA2 = arrayA.copyToHost();

        REQUIRE(&(hostA1.data()[0]) != &(hostA2.data()[0]));
        REQUIRE(hostA1.data()[0] == hostA2.data()[0]);
        REQUIRE(hostA1.data()[1] == hostA2.data()[1]);
        REQUIRE(hostA1.data()[2] == hostA2.data()[2]);
    }
}
