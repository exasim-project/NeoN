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

/* This test uses mpi initialized in the catch_mpi_main  */
TEST_CASE("noMPIEnvironment")
{
    Environment mpiEnviron;

    SECTION("mpiEnviron has 3 ranks")
    {
        REQUIRE(mpiEnviron.sizeRank() == 3);
        REQUIRE(mpiEnviron.rank() != -1);
    }
}
