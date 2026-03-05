// SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "NeoN/NeoN.hpp"

using Catch::Approx;

TEST_CASE("Tensor")
{
    SECTION("Default constructor - all zeros")
    {
        NeoN::Tensor t;
        for (size_t i = 0; i < 9; i++)
        {
            REQUIRE(t[i] == 0.0);
        }
    }

    SECTION("9-arg constructor and component access")
    {
        NeoN::Tensor t(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0);
        REQUIRE(t.xx() == 1.0);
        REQUIRE(t.xy() == 2.0);
        REQUIRE(t.xz() == 3.0);
        REQUIRE(t.yx() == 4.0);
        REQUIRE(t.yy() == 5.0);
        REQUIRE(t.yz() == 6.0);
        REQUIRE(t.zx() == 7.0);
        REQUIRE(t.zy() == 8.0);
        REQUIRE(t.zz() == 9.0);
    }

    SECTION("Uniform scalar constructor")
    {
        NeoN::Tensor t(3.0);
        for (size_t i = 0; i < 9; i++)
        {
            REQUIRE(t[i] == 3.0);
        }
    }

    SECTION("Row-col access via operator()")
    {
        NeoN::Tensor t(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0);
        REQUIRE(t(0, 0) == 1.0);
        REQUIRE(t(0, 1) == 2.0);
        REQUIRE(t(0, 2) == 3.0);
        REQUIRE(t(1, 0) == 4.0);
        REQUIRE(t(1, 1) == 5.0);
        REQUIRE(t(1, 2) == 6.0);
        REQUIRE(t(2, 0) == 7.0);
        REQUIRE(t(2, 1) == 8.0);
        REQUIRE(t(2, 2) == 9.0);
    }

    SECTION("Equality")
    {
        NeoN::Tensor a(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0);
        NeoN::Tensor b(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0);
        REQUIRE(a == b);
    }

    SECTION("Addition")
    {
        NeoN::Tensor a(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0);
        NeoN::Tensor b(9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0);
        NeoN::Tensor c = a + b;
        NeoN::Tensor expected(10.0);
        REQUIRE(c == expected);
    }

    SECTION("Subtraction")
    {
        NeoN::Tensor a(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0);
        NeoN::Tensor b(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0);
        NeoN::Tensor c = a - b;
        NeoN::Tensor expected(0.0);
        REQUIRE(c == expected);
    }

    SECTION("Compound assignment += and -=")
    {
        NeoN::Tensor a(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0);
        NeoN::Tensor b(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0);
        a += b;
        NeoN::Tensor expected(2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0);
        REQUIRE(a == expected);

        a -= b;
        REQUIRE(a == b);
    }

    SECTION("Scalar multiplication")
    {
        NeoN::Tensor a(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0);
        NeoN::Tensor c = a * 2.0;
        NeoN::Tensor expected(2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0);
        REQUIRE(c == expected);

        NeoN::Tensor d = 2.0 * a;
        REQUIRE(d == expected);

        a *= 2.0;
        REQUIRE(a == expected);
    }

    SECTION("Scalar division")
    {
        NeoN::Tensor a(2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0);
        NeoN::Tensor c = a / 2.0;
        NeoN::Tensor expected(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0);
        REQUIRE(c == expected);
    }

    SECTION("mag (Frobenius norm)")
    {
        // Identity tensor: mag = sqrt(3)
        NeoN::Tensor I(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
        REQUIRE(NeoN::mag(I) == Approx(std::sqrt(3.0)));
    }

    SECTION("Transpose")
    {
        NeoN::Tensor a(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0);
        NeoN::Tensor aT = NeoN::T(a);
        NeoN::Tensor expected(1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 9.0);
        REQUIRE(aT == expected);
    }

    SECTION("Skew: 0.5*(T - T^T)")
    {
        NeoN::Tensor a(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0);
        NeoN::Tensor s = NeoN::skew(a);
        // skew(A) = 0.5*(A - A^T)
        // diagonal should be zero
        REQUIRE(s.xx() == 0.0);
        REQUIRE(s.yy() == 0.0);
        REQUIRE(s.zz() == 0.0);
        // off-diagonal: xy = 0.5*(2-4) = -1, yx = 0.5*(4-2) = 1
        REQUIRE(s.xy() == Approx(-1.0));
        REQUIRE(s.yx() == Approx(1.0));
    }

    SECTION("Traits")
    {
        auto z = NeoN::zero<NeoN::Tensor>();
        for (size_t i = 0; i < 9; i++)
        {
            REQUIRE(z[i] == 0.0);
        }

        auto o = NeoN::one<NeoN::Tensor>();
        for (size_t i = 0; i < 9; i++)
        {
            REQUIRE(o[i] == 1.0);
        }
    }

    SECTION("Size and data")
    {
        NeoN::Tensor t(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0);
        REQUIRE(t.size() == 9);
        REQUIRE(t.data() != nullptr);
    }
}
