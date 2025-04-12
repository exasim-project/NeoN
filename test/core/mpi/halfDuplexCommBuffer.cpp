// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoN authors

#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_adapters.hpp>

#include "NeoN/core/mpi/halfDuplexCommBuffer.hpp"
#include "NeoN/core/mpi/environment.hpp"

using namespace NeoN;
using namespace NeoN::mpi;

TEST_CASE("halfDuplexBuffer")
{

    MPIEnvironment mpiEnviron;
    std::vector<std::size_t> rankCommSize(mpiEnviron.sizeRank(), 1);
    HalfDuplexCommBuffer buffer(mpiEnviron, rankCommSize);

    SECTION("Default Constructor")
    {
        HalfDuplexCommBuffer buffer2;
        REQUIRE_FALSE(buffer2.isCommInit());
    }

    SECTION("Parameterized Constructor") { REQUIRE_FALSE(buffer.isCommInit()); }

    SECTION("Init and Finalise")
    {
        buffer.initComm<int>("Init Comm");
        REQUIRE(buffer.isCommInit());
        REQUIRE(true == buffer.isComplete());
        REQUIRE(buffer.getCommName() == "Init Comm");
        buffer.finaliseComm();
        REQUIRE(buffer.getCommName() == "unassigned");
        REQUIRE(true == buffer.isComplete());
        REQUIRE(!buffer.isCommInit());
    }

    SECTION("Set Comm Rank Size")
    {
        for (size_t rank = 0; rank < mpiEnviron.sizeRank(); ++rank)
            rankCommSize[rank] = rank;
        buffer.setCommRankSize<double>(rankCommSize);
        buffer.initComm<double>("Set Comm Rank Size");
        for (size_t rank = 0; rank < mpiEnviron.sizeRank(); ++rank)
        {
            auto data = buffer.get<double>(rank);
            REQUIRE(data.size() == rank);
        }
        buffer.finaliseComm();
    }

    SECTION("Send and Receive")
    {

        HalfDuplexCommBuffer send(mpiEnviron, rankCommSize);
        HalfDuplexCommBuffer receive(mpiEnviron, rankCommSize);

        send.initComm<int>("Send and Receive");
        receive.initComm<int>("Send and Receive");
        for (size_t rank = 0; rank < mpiEnviron.sizeRank(); ++rank)
        {
            auto data = send.get<int>(rank);
            data[0] = static_cast<int>(rank);
        }

        send.send();
        receive.receive();

        send.waitComplete();
        receive.waitComplete();

        for (size_t rank = 0; rank < mpiEnviron.sizeRank(); ++rank)
        {
            auto data = receive.get<int>(rank);
            REQUIRE(data[0] == static_cast<int>(mpiEnviron.rank()));
        }

        send.finaliseComm();
        receive.finaliseComm();
    }
}
