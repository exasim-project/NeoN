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

TEST_CASE("SymmTensor")
{
    SECTION("Default constructor - all zeros")
    {
        NeoN::SymmTensor s;
        for (size_t i = 0; i < 6; i++)
        {
            REQUIRE(s[i] == 0.0);
        }
    }

    SECTION("6-arg constructor and component access")
    {
        NeoN::SymmTensor s(1.0, 2.0, 3.0, 4.0, 5.0, 6.0);
        REQUIRE(s.xx() == 1.0);
        REQUIRE(s.xy() == 2.0);
        REQUIRE(s.xz() == 3.0);
        REQUIRE(s.yy() == 4.0);
        REQUIRE(s.yz() == 5.0);
        REQUIRE(s.zz() == 6.0);
    }

    SECTION("Uniform scalar constructor")
    {
        NeoN::SymmTensor s(3.0);
        for (size_t i = 0; i < 6; i++)
        {
            REQUIRE(s[i] == 3.0);
        }
    }

    SECTION("Equality")
    {
        NeoN::SymmTensor a(1.0, 2.0, 3.0, 4.0, 5.0, 6.0);
        NeoN::SymmTensor b(1.0, 2.0, 3.0, 4.0, 5.0, 6.0);
        REQUIRE(a == b);
    }

    SECTION("Addition and subtraction")
    {
        NeoN::SymmTensor a(1.0, 2.0, 3.0, 4.0, 5.0, 6.0);
        NeoN::SymmTensor b(6.0, 5.0, 4.0, 3.0, 2.0, 1.0);
        NeoN::SymmTensor c = a + b;
        NeoN::SymmTensor expected(7.0);
        REQUIRE(c == expected);

        NeoN::SymmTensor d = a - a;
        NeoN::SymmTensor zero(0.0);
        REQUIRE(d == zero);
    }

    SECTION("Compound assignment += and -=")
    {
        NeoN::SymmTensor a(1.0, 2.0, 3.0, 4.0, 5.0, 6.0);
        NeoN::SymmTensor b(1.0, 2.0, 3.0, 4.0, 5.0, 6.0);
        a += b;
        NeoN::SymmTensor expected(2.0, 4.0, 6.0, 8.0, 10.0, 12.0);
        REQUIRE(a == expected);
        a -= b;
        REQUIRE(a == b);
    }

    SECTION("Scalar multiplication and division")
    {
        NeoN::SymmTensor a(1.0, 2.0, 3.0, 4.0, 5.0, 6.0);
        NeoN::SymmTensor c = a * 2.0;
        NeoN::SymmTensor expected(2.0, 4.0, 6.0, 8.0, 10.0, 12.0);
        REQUIRE(c == expected);
        REQUIRE((2.0 * a) == expected);

        a *= 2.0;
        REQUIRE(a == expected);

        NeoN::SymmTensor d = expected / 2.0;
        NeoN::SymmTensor orig(1.0, 2.0, 3.0, 4.0, 5.0, 6.0);
        REQUIRE(d == orig);
    }

    SECTION("mag")
    {
        NeoN::SymmTensor s(1.0, 0.0, 0.0, 1.0, 0.0, 1.0);
        REQUIRE(NeoN::mag(s) == Approx(std::sqrt(3.0)));
    }

    SECTION("dev (deviatoric)")
    {
        // For identity-like SymmTensor: dev(S) = S - tr(S)/3 * I
        // S = (1,0,0,1,0,1), tr=3, dev = S - I = (0,0,0,0,0,0)
        NeoN::SymmTensor s(1.0, 0.0, 0.0, 1.0, 0.0, 1.0);
        NeoN::SymmTensor d = NeoN::dev(s);
        NeoN::SymmTensor zero(0.0);
        REQUIRE(d == zero);

        // Non-trivial case: S = (3,0,0,0,0,0), tr=3, dev = (2,0,0,-1,0,-1)
        NeoN::SymmTensor s2(3.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        NeoN::SymmTensor d2 = NeoN::dev(s2);
        REQUIRE(d2.xx() == Approx(2.0));
        REQUIRE(d2.yy() == Approx(-1.0));
        REQUIRE(d2.zz() == Approx(-1.0));
    }

    SECTION("Traits")
    {
        auto z = NeoN::zero<NeoN::SymmTensor>();
        for (size_t i = 0; i < 6; i++)
        {
            REQUIRE(z[i] == 0.0);
        }

        auto o = NeoN::one<NeoN::SymmTensor>();
        for (size_t i = 0; i < 6; i++)
        {
            REQUIRE(o[i] == 1.0);
        }
    }

    SECTION("Size and data")
    {
        NeoN::SymmTensor s(1.0, 2.0, 3.0, 4.0, 5.0, 6.0);
        REQUIRE(s.size() == 6);
        REQUIRE(s.data() != nullptr);
    }
}

TEST_CASE("Tensor-SymmTensor cross-type operations")
{
    SECTION("symm(Tensor) -> SymmTensor: 0.5*(T + T^T)")
    {
        // symmetric tensor should produce itself
        NeoN::Tensor t(1.0, 2.0, 3.0, 2.0, 4.0, 5.0, 3.0, 5.0, 6.0);
        NeoN::SymmTensor s = NeoN::symm(t);
        REQUIRE(s.xx() == Approx(1.0));
        REQUIRE(s.xy() == Approx(2.0));
        REQUIRE(s.xz() == Approx(3.0));
        REQUIRE(s.yy() == Approx(4.0));
        REQUIRE(s.yz() == Approx(5.0));
        REQUIRE(s.zz() == Approx(6.0));
    }

    SECTION("twoSymm(Tensor) -> SymmTensor: T + T^T")
    {
        NeoN::Tensor t(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0);
        NeoN::SymmTensor ts = NeoN::twoSymm(t);
        // twoSymm_xx = 2*xx = 2, xy = xy+yx = 6, xz = xz+zx = 10
        // yy = 2*yy = 10, yz = yz+zy = 14, zz = 2*zz = 18
        REQUIRE(ts.xx() == Approx(2.0));
        REQUIRE(ts.xy() == Approx(6.0));
        REQUIRE(ts.xz() == Approx(10.0));
        REQUIRE(ts.yy() == Approx(10.0));
        REQUIRE(ts.yz() == Approx(14.0));
        REQUIRE(ts.zz() == Approx(18.0));
    }
}
