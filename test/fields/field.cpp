// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main
#include "catch2_common.hpp"

#include "NeoN/NeoN.hpp"

TEST_CASE("Boundaries")
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    SECTION("domainVector_" + execName)
    {

        NeoN::Field<double> a(exec, 1000, {0, 10, 20, 30});

        NeoN::fill(a.internalVector(), 2.0);
        REQUIRE(equal(a.internalVector(), 2.0));
    }

    SECTION("boundaryVectors_" + execName)
    {

        NeoN::BoundaryData<double> bCs(exec, {0, 10, 20, 30});

        NeoN::fill(bCs.value(), 2.0);
        REQUIRE(equal(bCs.value(), 2.0));

        NeoN::fill(bCs.refValue(), 2.0);
        REQUIRE(equal(bCs.refValue(), 2.0));

        NeoN::fill(bCs.refGrad(), 2.0);
        REQUIRE(equal(bCs.refGrad(), 2.0));

        NeoN::fill(bCs.valueFraction(), 2.0);
        REQUIRE(equal(bCs.valueFraction(), 2.0));
    }

    SECTION("moves do not copy " + execName)
    {
        // A move must hand over the buffers (same pointers, source emptied) instead of
        // deep-copying them; BoundaryData used to have no move constructor, so Field's move
        // constructor and move assignment silently copied every boundary vector.
        NeoN::Field<double> a(exec, 1000, {0, 10, 20, 30});
        NeoN::fill(a.internalVector(), 2.0);
        NeoN::fill(a.boundaryData().value(), 3.0);
        const auto* internal = a.internalVector().data();
        const auto* value = a.boundaryData().value().data();
        const auto* offset = a.boundaryData().offset().data();

        NeoN::Field<double> b(std::move(a));
        REQUIRE(b.internalVector().data() == internal);
        REQUIRE(b.boundaryData().value().data() == value);
        REQUIRE(b.boundaryData().offset().data() == offset);
        REQUIRE(a.boundaryData().value().data() == nullptr);
        REQUIRE(equal(b.internalVector(), 2.0));
        REQUIRE(equal(b.boundaryData().value(), 3.0));
        REQUIRE(b.boundaryData().nBoundaryFaces() == 30);

        NeoN::Field<double> c(exec, 10, {0, 5});
        c = std::move(b);
        REQUIRE(c.boundaryData().value().data() == value);
        REQUIRE(c.boundaryData().offset().data() == offset);
        REQUIRE(equal(c.boundaryData().value(), 3.0));
    }
}
