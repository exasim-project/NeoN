// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main
#include "catch2_common.hpp"

#include "NeoN/NeoN.hpp"

using Vector = NeoN::Vector<NeoN::scalar>;
using Coeff = NeoN::dsl::Coeff;
namespace dsl = NeoN::dsl;


TEST_CASE("Coeff")
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    SECTION("Coefficient evaluation on " + execName)
    {
        Vector fA(exec, 3, 2.0);
        Vector res(exec, 1);

        Coeff a {};
        Coeff b {2.0};
        Coeff c = 2 * a * b;
        REQUIRE(c[0] == 4.0);

        Coeff d {3.0, fA};
        dsl::detail::toVector(d, res);
        auto resDexp = std::vector<NeoN::scalar> {6.0, 6.0, 6.0};
        REQUIRE_THAT(res, Equals(resDexp));

        Coeff e = d * b;
        dsl::detail::toVector(e, res);
        auto resEexp = std::vector<NeoN::scalar> {12.0, 12.0, 12.0};
        REQUIRE_THAT(res, Equals(resEexp));
        Coeff f = b * d;
        dsl::detail::toVector(f, res);
        auto resFexp = std::vector<NeoN::scalar> {12.0, 12.0, 12.0};
        REQUIRE_THAT(res, Equals(resFexp));
    }

    SECTION("evaluation in parallelFor" + execName)
    {
        NeoN::localIdx size = 3;

        Vector fieldA(exec, size, 0.0);
        Vector fieldB(exec, size, 1.0);

        SECTION("view")
        {
            Coeff c = fieldB; // is a view with uniform value 1.0
            NeoN::parallelFor(
                fieldA, NEON_LAMBDA(const NeoN::localIdx i) { return c[i] + 2.0; }
            );
            auto resAexp = std::vector<NeoN::scalar> {3.0, 3.0, 3.0};
            REQUIRE_THAT(fieldA, Equals(resAexp));
        }

        SECTION("scalar")
        {
            Coeff c = Coeff(2.0);
            NeoN::parallelFor(
                fieldA, NEON_LAMBDA(const NeoN::localIdx i) { return c[i] + 2.0; }
            );
            auto resAexp = std::vector<NeoN::scalar> {4.0, 4.0, 4.0};
            REQUIRE_THAT(fieldA, Equals(resAexp));
        }

        SECTION("view and scalar")
        {
            Coeff c {-5.0, fieldB};
            NeoN::parallelFor(
                fieldA, NEON_LAMBDA(const NeoN::localIdx i) { return c[i] + 2.0; }
            );
            auto resAexp = std::vector<NeoN::scalar> {-3.0, -3.0, -3.0};
            REQUIRE_THAT(fieldA, Equals(resAexp));
        }
    }
}
