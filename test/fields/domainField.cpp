// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023-2024 NeoN authors
#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main
#include "catch2_common.hpp"

#include "NeoN/NeoN.hpp"

TEST_CASE("Boundaries")
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    SECTION("domainField_" + execName)
    {

        NeoN::DomainField<double> a(exec, 1000, {0, 10, 20, 30});

        NeoN::fill(a.internalField(), 2.0);
        REQUIRE(equal(a.internalField(), 2.0));
    }

    SECTION("boundaryFields_" + execName)
    {

        NeoN::BoundaryFields<double> bCs(exec, {0, 10, 20, 30});

        NeoN::fill(bCs.value(), 2.0);
        REQUIRE(equal(bCs.value(), 2.0));

        NeoN::fill(bCs.refValue(), 2.0);
        REQUIRE(equal(bCs.refValue(), 2.0));

        NeoN::fill(bCs.refGrad(), 2.0);
        REQUIRE(equal(bCs.refGrad(), 2.0));

        NeoN::fill(bCs.valueFraction(), 2.0);
        REQUIRE(equal(bCs.valueFraction(), 2.0));
    }
}
