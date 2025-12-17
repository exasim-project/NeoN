// SPDX-FileCopyrightText: 2023 - 2025 NeoN authors
//
// SPDX-License-Identifier: MIT

#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_adapters.hpp>

#include "NeoN/core/mpi/fullDuplexCommBuffer.hpp"
#include "NeoN/core/mpi/environment.hpp"
#include <cstring>

using namespace NeoN::mpi;

/* This test ensures that mpi::Environment can be instantiated
  even if no mpi is available */
TEST_CASE("noMPIEnvironment")
{
    Environment mpiEnviron;

    SECTION("mpiEnviron has -1 ranks")
    {
        REQUIRE(mpiEnviron.sizeRank() == -1);
        REQUIRE(mpiEnviron.rank() == -1);
    }
}
