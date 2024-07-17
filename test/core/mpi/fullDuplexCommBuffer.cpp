// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_adapters.hpp>

#include "NeoFOAM/core/mpi/fullDuplexCommBuffer.hpp"
#include "NeoFOAM/core/mpi/environment.hpp"
#include <cstring>

using namespace NeoFOAM;
using namespace NeoFOAM::mpi;

TEST_CASE("fullDuplexBuffer")
{

    MPIEnvironment mpiEnviron;
    std::vector<NeoFOAM::size_t> rankSendCommSize(mpiEnviron.usizeRank(), 1);
    std::vector<NeoFOAM::size_t> rankReceiveCommSize(mpiEnviron.usizeRank(), 1);
    FullDuplexCommBuffer buffer(mpiEnviron, rankSendCommSize, rankReceiveCommSize);

    SECTION("Parameterized Constructor")
    {
        FullDuplexCommBuffer buffer2;
        REQUIRE_FALSE(buffer2.isCommInit());
    }

    SECTION("Init and Finalise")
    {
        buffer.initComm<int>("Init Comm");
        REQUIRE(buffer.isCommInit());
        REQUIRE(true == buffer.isComplete());
        buffer.finaliseComm();
        REQUIRE(!buffer.isCommInit());
        REQUIRE(true == buffer.isComplete());
    }

    SECTION("Send and Receive")
    {
        buffer.initComm<int>("Send and Receive");
        for (NeoFOAM::size_t rank = 0; rank < mpiEnviron.sizeRank(); ++rank)
        {
            auto data = buffer.getSend<int>(rank);
            data[0] = static_cast<int>(rank);
        }

        buffer.startComm();
        buffer.waitComplete();

        for (NeoFOAM::size_t rank = 0; rank < mpiEnviron.sizeRank(); ++rank)
        {
            auto data = buffer.getReceive<int>(rank);
            REQUIRE(data[0] == static_cast<int>(mpiEnviron.rank()));
        }

        buffer.finaliseComm();
    }
}
