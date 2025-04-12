// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023-2024 NeoN authors

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>

#include "NeoN/NeoN.hpp"

TEST_CASE("Primitives")
{

    SECTION("Vector")
    {
        SECTION("CPU")
        {
            NeoN::Vector a(1.0, 2.0, 3.0);
            REQUIRE(a(0) == 1.0);
            REQUIRE(a(1) == 2.0);
            REQUIRE(a(2) == 3.0);

            NeoN::Vector b(1.0, 2.0, 3.0);
            REQUIRE(a == b);

            NeoN::Vector c(2.0, 4.0, 6.0);

            REQUIRE(a + b == c);

            REQUIRE((a - b) == NeoN::Vector(0.0, 0.0, 0.0));

            a += b;
            REQUIRE(a == c);

            a -= b;
            REQUIRE(a == b);
            a *= 2;
            REQUIRE(a == c);
            a = b;

            REQUIRE(a == b);

            NeoN::Vector d(4.0, 8.0, 12.0);
            REQUIRE((a + a + a + a) == d);
            REQUIRE((4 * a) == d);
            REQUIRE((a * 4) == d);
            REQUIRE((a + 3 * a) == d);
            REQUIRE((a + 2 * a + a) == d);
        }
    }

    SECTION("Vector", "[Traits]")
    {

        auto one = NeoN::one<NeoN::Vector>();

        REQUIRE(one(0) == 1.0);
        REQUIRE(one(1) == 1.0);
        REQUIRE(one(2) == 1.0);

        auto zero = NeoN::zero<NeoN::Vector>();

        REQUIRE(zero(0) == 0.0);
        REQUIRE(zero(1) == 0.0);
        REQUIRE(zero(2) == 0.0);
    }
}
