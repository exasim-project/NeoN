// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023-2024 NeoN authors
#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main
#include "catch2_common.hpp"

#include "NeoN/NeoN.hpp"

using Field = NeoN::Field<NeoN::scalar>;
using Coeff = NeoN::dsl::Coeff;
namespace dsl = NeoN::dsl;


TEST_CASE("Coeff")
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    SECTION("Coefficient evaluation on " + execName)
    {
        Field fA(exec, 3, 2.0);
        Field res(exec, 1);

        Coeff a {};
        Coeff b {2.0};
        Coeff c = 2 * a * b;
        REQUIRE(c[0] == 4.0);

        Coeff d {3.0, fA};
        dsl::detail::toField(d, res);
        auto hostResD = res.copyToHost();
        REQUIRE(hostResD.data()[0] == 6.0);
        REQUIRE(hostResD.data()[1] == 6.0);
        REQUIRE(hostResD.data()[2] == 6.0);

        Coeff e = d * b;
        dsl::detail::toField(e, res);
        auto hostResE = res.copyToHost();
        REQUIRE(hostResE.data()[0] == 12.0);
        REQUIRE(hostResE.data()[1] == 12.0);
        REQUIRE(hostResE.data()[2] == 12.0);

        Coeff f = b * d;
        dsl::detail::toField(f, res);
        auto hostResF = res.copyToHost();
        REQUIRE(hostResF.data()[0] == 12.0);
        REQUIRE(hostResF.data()[1] == 12.0);
        REQUIRE(hostResF.data()[2] == 12.0);
    }

    SECTION("evaluation in parallelFor" + execName)
    {
        size_t size = 3;

        Field fieldA(exec, size, 0.0);
        Field fieldB(exec, size, 1.0);

        SECTION("view")
        {
            Coeff coeff = fieldB; // is a view with uniform value 1.0
            {
                NeoN::parallelFor(
                    fieldA, KOKKOS_LAMBDA(const size_t i) { return coeff[i] + 2.0; }
                );
            };
            auto hostFieldA = fieldA.copyToHost();
            REQUIRE(coeff.hasSpan() == true);
            REQUIRE(hostFieldA.view()[0] == 3.0);
            REQUIRE(hostFieldA.view()[1] == 3.0);
            REQUIRE(hostFieldA.view()[2] == 3.0);
        }

        SECTION("scalar")
        {
            Coeff coeff = Coeff(2.0);
            {
                NeoN::parallelFor(
                    fieldA, KOKKOS_LAMBDA(const size_t i) { return coeff[i] + 2.0; }
                );
            };
            auto hostFieldA = fieldA.copyToHost();
            REQUIRE(coeff.hasSpan() == false);
            REQUIRE(hostFieldA.view()[0] == 4.0);
            REQUIRE(hostFieldA.view()[1] == 4.0);
            REQUIRE(hostFieldA.view()[2] == 4.0);
        }

        SECTION("view and scalar")
        {
            Coeff coeff {-5.0, fieldB};
            {
                NeoN::parallelFor(
                    fieldA, KOKKOS_LAMBDA(const size_t i) { return coeff[i] + 2.0; }
                );
            };
            auto hostFieldA = fieldA.copyToHost();
            REQUIRE(coeff.hasSpan() == true);
            REQUIRE(hostFieldA.view()[0] == -3.0);
            REQUIRE(hostFieldA.view()[1] == -3.0);
            REQUIRE(hostFieldA.view()[2] == -3.0);
        }
    }
}
