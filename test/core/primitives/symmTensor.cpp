// SPDX-FileCopyrightText: 2024 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#define CATCH_CONFIG_RUNNER
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "NeoN/NeoN.hpp"

using NeoN::scalar;
using NeoN::SymmTensor;
using NeoN::Tensor;
using NeoN::Vec3;

TEST_CASE("SymmTensor - Constructors")
{
    SECTION("Default constructor zeros all components")
    {
        SymmTensor s;
        for (size_t i = 0; i < 3; ++i)
        {
            for (size_t j = 0; j < 3; ++j)
            {
                REQUIRE(s(i, j) == 0.0);
            }
        }
    }

    SECTION("Diagonal constructor produces scalar * I")
    {
        SymmTensor s(3.0);
        REQUIRE(s(0, 0) == 3.0);
        REQUIRE(s(1, 1) == 3.0);
        REQUIRE(s(2, 2) == 3.0);
        REQUIRE(s(0, 1) == 0.0);
        REQUIRE(s(0, 2) == 0.0);
        REQUIRE(s(1, 2) == 0.0);
    }

    SECTION("6-component constructor - upper-triangle storage")
    {
        SymmTensor s(1.0, 2.0, 3.0, 4.0, 5.0, 6.0);
        // diagonal
        REQUIRE(s(0, 0) == 1.0); // xx
        REQUIRE(s(1, 1) == 4.0); // yy
        REQUIRE(s(2, 2) == 6.0); // zz
        // off-diagonal and symmetry
        REQUIRE(s(0, 1) == 2.0); // xy
        REQUIRE(s(1, 0) == 2.0); // yx == xy
        REQUIRE(s(0, 2) == 3.0); // xz
        REQUIRE(s(2, 0) == 3.0); // zx == xz
        REQUIRE(s(1, 2) == 5.0); // yz
        REQUIRE(s(2, 1) == 5.0); // zy == yz
    }
}

TEST_CASE("SymmTensor - Size")
{
    SymmTensor s;
    REQUIRE(s.size() == 6);
}

TEST_CASE("SymmTensor - Equality")
{
    SymmTensor a(1.0, 2.0, 3.0, 4.0, 5.0, 6.0);
    SymmTensor b(1.0, 2.0, 3.0, 4.0, 5.0, 6.0);
    SymmTensor c(6.0, 5.0, 4.0, 3.0, 2.0, 1.0);

    REQUIRE(a == b);
    REQUIRE(!(a == c));
}

TEST_CASE("SymmTensor - Arithmetic")
{
    SymmTensor a(1.0, 2.0, 3.0, 4.0, 5.0, 6.0);
    SymmTensor b(6.0, 5.0, 4.0, 3.0, 2.0, 1.0);
    SymmTensor sum(7.0, 7.0, 7.0, 7.0, 7.0, 7.0);
    SymmTensor zero(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);

    SECTION("Addition")
    {
        REQUIRE(a + b == sum);
        SymmTensor c = a;
        c += b;
        REQUIRE(c == sum);
    }

    SECTION("Subtraction")
    {
        REQUIRE(a - a == zero);
        SymmTensor c = a;
        c -= a;
        REQUIRE(c == zero);
    }

    SECTION("Scalar multiplication")
    {
        SymmTensor doubled(2.0, 4.0, 6.0, 8.0, 10.0, 12.0);
        REQUIRE(a * 2.0 == doubled);
        REQUIRE(2.0 * a == doubled);
        SymmTensor c = a;
        c *= 2.0;
        REQUIRE(c == doubled);
    }
}

TEST_CASE("SymmTensor - Trace")
{
    SymmTensor s(1.0, 2.0, 3.0, 4.0, 5.0, 6.0);
    REQUIRE(s.trace() == 1.0 + 4.0 + 6.0); // xx + yy + zz

    SymmTensor identity(1.0);
    REQUIRE(identity.trace() == 3.0);
}

TEST_CASE("SymmTensor - Deviatoric parts")
{
    SECTION("dev() = S - (1/3)*tr(S)*I")
    {
        SymmTensor s(3.0, 1.0, 2.0, 6.0, 4.0, 9.0);
        scalar tr = s.trace(); // 3 + 6 + 9 = 18
        scalar tr3 = tr / 3.0; // 6
        SymmTensor d = s.dev();

        REQUIRE_THAT(d(0, 0), Catch::Matchers::WithinRel(3.0 - tr3));
        REQUIRE_THAT(d(1, 1), Catch::Matchers::WithinRel(6.0 - tr3));
        REQUIRE_THAT(d(2, 2), Catch::Matchers::WithinRel(9.0 - tr3));
        REQUIRE(d(0, 1) == s(0, 1)); // off-diagonal unchanged
        REQUIRE(d(0, 2) == s(0, 2));
        REQUIRE(d(1, 2) == s(1, 2));

        // deviatoric is traceless
        REQUIRE_THAT(d.trace(), Catch::Matchers::WithinAbs(0.0, 1e-12));
    }

    SECTION("dev2() = S - (2/3)*tr(S)*I")
    {
        SymmTensor s(3.0, 1.0, 2.0, 6.0, 4.0, 9.0);
        scalar tr = s.trace();        // 18
        scalar tr23 = 2.0 / 3.0 * tr; // 12
        SymmTensor d2 = s.dev2();

        REQUIRE_THAT(d2(0, 0), Catch::Matchers::WithinRel(3.0 - tr23));
        REQUIRE_THAT(d2(1, 1), Catch::Matchers::WithinRel(6.0 - tr23));
        REQUIRE_THAT(d2(2, 2), Catch::Matchers::WithinRel(9.0 - tr23));
        REQUIRE(d2(0, 1) == s(0, 1));
        REQUIRE(d2(0, 2) == s(0, 2));
        REQUIRE(d2(1, 2) == s(1, 2));
    }
}

TEST_CASE("SymmTensor - Matrix-vector product S·v")
{
    SymmTensor identity(1.0);
    Vec3 v(1.0, 2.0, 3.0);
    Vec3 result = identity & v;
    REQUIRE(result[0] == 1.0);
    REQUIRE(result[1] == 2.0);
    REQUIRE(result[2] == 3.0);

    SymmTensor s(2.0, 0.0, 0.0, 3.0, 0.0, 4.0);
    Vec3 scaled = s & v;
    REQUIRE(scaled[0] == 2.0);
    REQUIRE(scaled[1] == 6.0);
    REQUIRE(scaled[2] == 12.0);

    SymmTensor shear(0.0, 1.0, 0.0, 0.0, 0.0, 0.0);
    Vec3 sheared = shear & v;
    REQUIRE(sheared[0] == v[1]);
    REQUIRE(sheared[1] == v[0]);
    REQUIRE(sheared[2] == 0.0);
}

TEST_CASE("SymmTensor - Norms")
{
    SECTION("Identity tensor")
    {
        SymmTensor identity(1.0);
        REQUIRE_THAT(NeoN::mag(identity), Catch::Matchers::WithinRel(std::sqrt(3.0)));
        REQUIRE_THAT(NeoN::magSqr(identity), Catch::Matchers::WithinRel(3.0));
    }

    SECTION("Pure shear - off-diagonal counts twice")
    {
        SymmTensor shear(0.0, 1.0, 0.0, 0.0, 0.0, 0.0); // only xy = yx = 1
        // Frobenius norm squared = xy^2 + yx^2 = 2
        REQUIRE_THAT(NeoN::magSqr(shear), Catch::Matchers::WithinRel(2.0));
        REQUIRE_THAT(NeoN::mag(shear), Catch::Matchers::WithinRel(std::sqrt(2.0)));
    }

    SECTION("Known tensor")
    {
        SymmTensor s(1.0, 2.0, 3.0, 4.0, 5.0, 6.0);
        // magSqr = 1^2 + 4^2 + 6^2 + 2*(2^2 + 3^2 + 5^2) = 1+16+36 + 2*(4+9+25) = 53 + 76 = 129
        scalar expected = 1.0 + 16.0 + 36.0 + 2.0 * (4.0 + 9.0 + 25.0);
        REQUIRE_THAT(NeoN::magSqr(s), Catch::Matchers::WithinRel(expected));
    }
}

TEST_CASE("SymmTensor - symm and twoSymm")
{
    SECTION("symm(T) extracts symmetric part of a full tensor")
    {
        Tensor t(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0);
        SymmTensor s = NeoN::symm(t);
        REQUIRE(s(0, 0) == 1.0);
        REQUIRE(s(1, 1) == 5.0);
        REQUIRE(s(2, 2) == 9.0);
        REQUIRE_THAT(s(0, 1), Catch::Matchers::WithinRel(0.5 * (2.0 + 4.0)));
        REQUIRE_THAT(s(0, 2), Catch::Matchers::WithinRel(0.5 * (3.0 + 7.0)));
        REQUIRE_THAT(s(1, 2), Catch::Matchers::WithinRel(0.5 * (6.0 + 8.0)));
        REQUIRE(s(1, 0) == s(0, 1));
        REQUIRE(s(2, 0) == s(0, 2));
        REQUIRE(s(2, 1) == s(1, 2));
    }

    SECTION("symm of symmetric tensor is identity operation")
    {
        Tensor t(1.0, 2.0, 3.0, 2.0, 5.0, 6.0, 3.0, 6.0, 9.0);
        SymmTensor s = NeoN::symm(t);
        REQUIRE(s(0, 0) == 1.0);
        REQUIRE(s(0, 1) == 2.0);
        REQUIRE(s(0, 2) == 3.0);
        REQUIRE(s(1, 1) == 5.0);
        REQUIRE(s(1, 2) == 6.0);
        REQUIRE(s(2, 2) == 9.0);
    }

    SECTION("twoSymm(T) == T + T^T")
    {
        Tensor t(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0);
        Tensor ts = NeoN::twoSymm(t);
        REQUIRE(ts(0, 0) == 2.0 * t(0, 0));
        REQUIRE(ts(1, 1) == 2.0 * t(1, 1));
        REQUIRE(ts(2, 2) == 2.0 * t(2, 2));
        REQUIRE_THAT(ts(0, 1), Catch::Matchers::WithinRel(t(0, 1) + t(1, 0)));
        REQUIRE_THAT(ts(1, 0), Catch::Matchers::WithinRel(t(1, 0) + t(0, 1)));
        REQUIRE(ts(0, 1) == ts(1, 0));
        REQUIRE(ts(0, 2) == ts(2, 0));
        REQUIRE(ts(1, 2) == ts(2, 1));
    }
}

TEST_CASE("SymmTensor - Traits")
{
    SECTION("zero<SymmTensor>")
    {
        SymmTensor z = NeoN::zero<SymmTensor>();
        for (size_t i = 0; i < 3; ++i)
        {
            for (size_t j = 0; j < 3; ++j)
            {
                REQUIRE(z(i, j) == 0.0);
            }
        }
    }

    SECTION("one<SymmTensor> is identity")
    {
        SymmTensor id = NeoN::one<SymmTensor>();
        REQUIRE(id(0, 0) == 1.0);
        REQUIRE(id(1, 1) == 1.0);
        REQUIRE(id(2, 2) == 1.0);
        REQUIRE(id(0, 1) == 0.0);
        REQUIRE(id(0, 2) == 0.0);
        REQUIRE(id(1, 2) == 0.0);
    }
}
