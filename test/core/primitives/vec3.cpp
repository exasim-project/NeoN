// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>

#include "NeoN/NeoN.hpp"

TEST_CASE("Primitives")
{

    SECTION("Vec3")
    {
        SECTION("CPU")
        {
            NeoN::Vec3 a(1.0, 2.0, 3.0);
            REQUIRE(a(0) == 1.0);
            REQUIRE(a(1) == 2.0);
            REQUIRE(a(2) == 3.0);

            NeoN::Vec3 b(1.0, 2.0, 3.0);
            REQUIRE(a == b);

            NeoN::Vec3 c(2.0, 4.0, 6.0);

            REQUIRE(a + b == c);

            REQUIRE((a - b) == NeoN::Vec3(0.0, 0.0, 0.0));

            a += b;
            REQUIRE(a == c);

            a -= b;
            REQUIRE(a == b);
            a *= 2;
            REQUIRE(a == c);
            a = b;

            REQUIRE(a == b);

            NeoN::Vec3 d(4.0, 8.0, 12.0);
            REQUIRE((a + a + a + a) == d);
            REQUIRE((4 * a) == d);
            REQUIRE((a * 4) == d);
            REQUIRE((a + 3 * a) == d);
            REQUIRE((a + 2 * a + a) == d);
        }

        SECTION("cross product")
        {
            NeoN::Vec3 x(1.0, 0.0, 0.0);
            NeoN::Vec3 y(0.0, 1.0, 0.0);
            NeoN::Vec3 z(0.0, 0.0, 1.0);

            // right-handed, and the named form agrees with the operator
            REQUIRE((x ^ y) == z);
            REQUIRE(NeoN::cross(x, y) == z);
            REQUIRE((y ^ z) == x);
            REQUIRE((z ^ x) == y);

            // anticommutative, and parallel vectors give zero
            REQUIRE((y ^ x) == NeoN::Vec3(0.0, 0.0, -1.0));
            REQUIRE((x ^ x) == NeoN::zero<NeoN::Vec3>());

            NeoN::Vec3 u(1.0, 2.0, 3.0);
            NeoN::Vec3 v(4.0, 5.0, 6.0);
            REQUIRE((u ^ v) == NeoN::Vec3(-3.0, 6.0, -3.0));
            // orthogonal to both operands
            REQUIRE(((u ^ v) & u) == 0.0);
            REQUIRE(((u ^ v) & v) == 0.0);
        }
    }

    SECTION("Vec3", "[Traits]")
    {

        auto one = NeoN::one<NeoN::Vec3>();

        REQUIRE(one(0) == 1.0);
        REQUIRE(one(1) == 1.0);
        REQUIRE(one(2) == 1.0);

        auto zero = NeoN::zero<NeoN::Vec3>();

        REQUIRE(zero(0) == 0.0);
        REQUIRE(zero(1) == 0.0);
        REQUIRE(zero(2) == 0.0);
    }
}
