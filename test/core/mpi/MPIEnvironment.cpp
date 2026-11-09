// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
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
TEST_CASE("MPIEnvironment")
{
    Environment mpiEnviron;

    SECTION("mpiEnviron has 3 ranks")
    {
        REQUIRE(mpiEnviron.isInitialized() == true);
        REQUIRE(mpiEnviron.sizeRank() == 3);
        REQUIRE(mpiEnviron.rank() != -1);
    }
}

TEST_CASE("MPIEnvironment default communicator")
{
    REQUIRE(defaultCommunicator() == MPI_COMM_WORLD);
    REQUIRE(hasDefaultCommunicator() == false);

    // rank 0 alone, ranks 1 and 2 together
    int worldRank;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm group;
    MPI_Comm_split(MPI_COMM_WORLD, worldRank == 0 ? 0 : 1, worldRank, &group);

    setDefaultCommunicator(group);
    REQUIRE(hasDefaultCommunicator() == true);
    Environment mpiEnviron;
    REQUIRE(mpiEnviron.comm() == group);
    REQUIRE(mpiEnviron.sizeRank() == (worldRank == 0 ? 1 : 2));
    REQUIRE(mpiEnviron.rank() == (worldRank == 0 ? 0 : static_cast<size_t>(worldRank - 1)));
    REQUIRE(Environment(MPI_COMM_WORLD).sizeRank() == 3);

    setDefaultCommunicator(MPI_COMM_NULL);
    REQUIRE(defaultCommunicator() == MPI_COMM_WORLD);
    REQUIRE(hasDefaultCommunicator() == false);
    MPI_Comm_free(&group);
}
