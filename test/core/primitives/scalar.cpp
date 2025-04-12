// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023-2024 NeoN authors

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>

#include "NeoN/NeoN.hpp"

TEST_CASE("Primitives")
{
    SECTION("Scalar", "[Traits]")
    {
        auto one = NeoN::one<NeoN::scalar>();

        REQUIRE(one == 1.0);

        auto zero = NeoN::zero<NeoN::scalar>();

        REQUIRE(zero == 0.0);
    }

    SECTION("LocalIdx", "[Traits]")
    {
        auto one = NeoN::one<NeoN::localIdx>();

        REQUIRE(one == 1);

        auto zero = NeoN::zero<NeoN::localIdx>();

        REQUIRE(zero == 0);
    }
}
