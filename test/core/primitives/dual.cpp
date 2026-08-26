// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#define CATCH_CONFIG_RUNNER
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "NeoN/NeoN.hpp"
#include "NeoN/ad/designVariables.hpp"
#include "NeoN/core/primitives/dual.hpp"

NeoN_DUAL_REGISTER_TRAITS(NeoN::scalar, 2)

using Dual2 = NeoN::Dual<NeoN::scalar, 2>;

namespace
{
void requireClose(NeoN::scalar a, NeoN::scalar b, NeoN::scalar tol = 1e-12)
{
    REQUIRE_THAT(a, Catch::Matchers::WithinRel(b, tol));
}
}

TEST_CASE("Dual")
{
    SECTION("Traits")
    {
        REQUIRE(NeoN::one<Dual2>().value() == 1.0);
        REQUIRE(NeoN::zero<Dual2>().value() == 0.0);
        REQUIRE(NeoN::one<Dual2>().deriv(0) == 0.0);
    }

    SECTION("Seeding")
    {
        Dual2 x(3.0, 0);
        REQUIRE(x.value() == 3.0);
        REQUIRE(x.deriv(0) == 1.0);
        REQUIRE(x.deriv(1) == 0.0);

        // A value constructed without a slot is passive.
        Dual2 c(3.0);
        REQUIRE(c.deriv(0) == 0.0);
    }

    SECTION("Product and quotient rules")
    {
        Dual2 x(2.0, 0);
        Dual2 y(5.0, 1);

        auto p = x * y;
        requireClose(p.value(), 10.0);
        requireClose(p.deriv(0), 5.0); // dy/dx = y
        requireClose(p.deriv(1), 2.0); // dp/dy = x

        auto q = x / y;
        requireClose(q.value(), 0.4);
        requireClose(q.deriv(0), 1.0 / 5.0);
        requireClose(q.deriv(1), -2.0 / 25.0);
    }

    SECTION("Chain rule through elementary functions")
    {
        Dual2 x(4.0, 0);

        auto s = NeoN::sqrt(x);
        requireClose(s.value(), 2.0);
        requireClose(s.deriv(0), 0.25); // 1/(2 sqrt(x))

        auto l = NeoN::log(x);
        requireClose(l.deriv(0), 0.25); // 1/x

        auto e = NeoN::exp(x);
        requireClose(e.deriv(0), e.value());
    }

    SECTION("Shared leaf accumulates rather than overwrites")
    {
        // The diamond case: one design variable reaching the functional by two
        // paths. Getting this wrong yields a partial gradient that looks
        // entirely plausible, which is why it is tested explicitly.
        Dual2 nu(2.0, 0);
        auto J = nu * nu + NeoN::scalar(3.0) * nu; // dJ/dnu = 2 nu + 3 = 7
        requireClose(J.value(), 10.0);
        requireClose(J.deriv(0), 7.0);
    }

    SECTION("Inverse trait")
    {
        Dual2 x(4.0, 0);
        auto i = NeoN::inv(x);
        requireClose(i.value(), 0.25);
        requireClose(i.deriv(0), -1.0 / 16.0);
    }

    SECTION("Trivially copyable - required for device use")
    {
        STATIC_REQUIRE(std::is_trivially_copyable_v<Dual2>);
        STATIC_REQUIRE(sizeof(Dual2) == 3 * sizeof(NeoN::scalar));
    }
}

TEST_CASE("DesignVariables")
{
    SECTION("Declaration seeds distinct slots")
    {
        NeoN::ad::DesignVariables<NeoN::scalar, 2> dv;
        auto a = dv.declare("nu", 1e-2);
        auto b = dv.declare("Cs", 0.17);

        REQUIRE(dv.size() == 2);
        REQUIRE(dv.name(0) == "nu");
        REQUIRE(a.deriv(0) == 1.0);
        REQUIRE(a.deriv(1) == 0.0);
        REQUIRE(b.deriv(1) == 1.0);
    }

    SECTION("Capacity is enforced")
    {
        NeoN::ad::DesignVariables<NeoN::scalar, 1> dv;
        dv.declare("nu", 1e-2);
        REQUIRE_THROWS(dv.declare("Cs", 0.17));
    }

    SECTION("Scaling normalises a badly conditioned gradient")
    {
        NeoN::ad::DesignVariables<NeoN::scalar, 2> dv;
        auto nu = dv.declare("nu", 1e-5);
        auto Cs = dv.declare("Cs", 0.2);

        auto J = nu + Cs;
        auto g = dv.scaledGradient(J);

        // Raw gradient is (1, 1); scaled by the characteristic magnitudes it
        // reflects the relative influence of a fractional change in each.
        requireClose(g[0], 1e-5);
        requireClose(g[1], 0.2);
    }

    SECTION("Passive variables are detected")
    {
        NeoN::ad::DesignVariables<NeoN::scalar, 2> dv;
        auto nu = dv.declare("nu", 1e-2);
        dv.declare("unused", 1.0);

        auto J = nu * nu;
        auto dead = dv.passiveVariables(J);
        REQUIRE(dead.size() == 1);
        REQUIRE(dead[0] == "unused");
    }
}

int main(int argc, char* argv[])
{
    Kokkos::initialize(argc, argv);
    int result = Catch::Session().run(argc, argv);
    Kokkos::finalize();
    return result;
}
