// SPDX-FileCopyrightText: 2024 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#define CATCH_CONFIG_RUNNER
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "NeoN/NeoN.hpp"

using NeoN::scalar;
using NeoN::Tensor;
using NeoN::Vec3;

TEST_CASE("Tensor - Constructors")
{
    SECTION("Default constructor zeros all components")
    {
        Tensor t;
        for (size_t i = 0; i < 3; ++i)
        {
            for (size_t j = 0; j < 3; ++j)
            {
                REQUIRE(t(i, j) == 0.0);
            }
        }
    }

    SECTION("Diagonal constructor produces scalar * I")
    {
        Tensor t(3.0);
        REQUIRE(t(0, 0) == 3.0);
        REQUIRE(t(1, 1) == 3.0);
        REQUIRE(t(2, 2) == 3.0);
        REQUIRE(t(0, 1) == 0.0);
        REQUIRE(t(0, 2) == 0.0);
        REQUIRE(t(1, 0) == 0.0);
        REQUIRE(t(1, 2) == 0.0);
        REQUIRE(t(2, 0) == 0.0);
        REQUIRE(t(2, 1) == 0.0);
    }

    SECTION("Full component constructor - row-major layout")
    {
        Tensor t(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0);
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
}

TEST_CASE("Tensor - Size")
{
    Tensor t;
    REQUIRE(t.size() == 9);
}

TEST_CASE("Tensor - Equality")
{
    Tensor a(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0);
    Tensor b(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0);
    Tensor c(9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0);

    REQUIRE(a == b);
    REQUIRE(!(a == c));
}

TEST_CASE("Tensor - Arithmetic")
{
    Tensor a(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0);
    Tensor b(9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0);
    Tensor sum(10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0);
    Tensor zero(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);

    SECTION("Addition")
    {
        REQUIRE(a + b == sum);
        Tensor c = a;
        c += b;
        REQUIRE(c == sum);
    }

    SECTION("Subtraction")
    {
        REQUIRE(a - a == zero);
        Tensor c = a;
        c -= a;
        REQUIRE(c == zero);
    }

    SECTION("Scalar multiplication")
    {
        Tensor doubled(2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0);
        REQUIRE(a * 2.0 == doubled);
        REQUIRE(2.0 * a == doubled);
        Tensor c = a;
        c *= 2.0;
        REQUIRE(c == doubled);
    }
}

TEST_CASE("Tensor - Trace")
{
    Tensor t(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0);
    REQUIRE(t.trace() == 1.0 + 5.0 + 9.0);

    Tensor identity(1.0);
    REQUIRE(identity.trace() == 3.0);
}

TEST_CASE("Tensor - Transpose")
{
    Tensor a(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0);
    Tensor at = a.T();

    for (size_t i = 0; i < 3; ++i)
    {
        for (size_t j = 0; j < 3; ++j)
        {
            REQUIRE(at(i, j) == a(j, i));
        }
    }

    Tensor symmetric(1.0, 2.0, 3.0, 2.0, 5.0, 6.0, 3.0, 6.0, 9.0);
    REQUIRE(symmetric.T() == symmetric);
}

TEST_CASE("Tensor - Matrix-vector products")
{
    Tensor t(1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0);
    Vec3 v(1.0, 2.0, 3.0);

    SECTION("T·v")
    {
        Vec3 result = t & v;
        REQUIRE(result[0] == 1.0);
        REQUIRE(result[1] == 4.0);
        REQUIRE(result[2] == 9.0);
    }

    SECTION("v·T")
    {
        Vec3 result = v & t;
        REQUIRE(result[0] == 1.0);
        REQUIRE(result[1] == 4.0);
        REQUIRE(result[2] == 9.0);
    }

    SECTION("General asymmetric tensor")
    {
        Tensor g(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0);
        Vec3 u(1.0, 0.0, 0.0);
        Vec3 row0 = g & u;
        REQUIRE(row0[0] == 1.0);
        REQUIRE(row0[1] == 4.0);
        REQUIRE(row0[2] == 7.0);

        Vec3 col0 = u & g;
        REQUIRE(col0[0] == 1.0);
        REQUIRE(col0[1] == 2.0);
        REQUIRE(col0[2] == 3.0);
    }
}

TEST_CASE("Tensor - Norms")
{
    Tensor identity(1.0);
    REQUIRE_THAT(NeoN::mag(identity), Catch::Matchers::WithinRel(std::sqrt(3.0)));
    REQUIRE_THAT(NeoN::magSqr(identity), Catch::Matchers::WithinRel(3.0));

    Tensor t(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0);
    scalar expectedSqr = 1 + 4 + 9 + 16 + 25 + 36 + 49 + 64 + 81;
    REQUIRE_THAT(NeoN::magSqr(t), Catch::Matchers::WithinRel(expectedSqr));
    REQUIRE_THAT(NeoN::mag(t), Catch::Matchers::WithinRel(std::sqrt(expectedSqr)));
}

TEST_CASE("Tensor - Traits")
{
    SECTION("zero<Tensor>")
    {
        Tensor z = NeoN::zero<Tensor>();
        for (size_t i = 0; i < 3; ++i)
        {
            for (size_t j = 0; j < 3; ++j)
            {
                REQUIRE(z(i, j) == 0.0);
            }
        }
    }

    SECTION("one<Tensor> is identity")
    {
        Tensor id = NeoN::one<Tensor>();
        REQUIRE(id(0, 0) == 1.0);
        REQUIRE(id(1, 1) == 1.0);
        REQUIRE(id(2, 2) == 1.0);
        REQUIRE(id(0, 1) == 0.0);
        REQUIRE(id(0, 2) == 0.0);
        REQUIRE(id(1, 0) == 0.0);
        REQUIRE(id(1, 2) == 0.0);
        REQUIRE(id(2, 0) == 0.0);
        REQUIRE(id(2, 1) == 0.0);
    }
}
