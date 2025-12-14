// SPDX-FileCopyrightText: 2023 - 2025 NeoN authors
//
// SPDX-License-Identifier: MIT

#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_adapters.hpp>

#include "NeoN/core/mpi/fullDuplexCommBuffer.hpp"
#include "NeoN/core/mpi/environment.hpp"
#include <cstring>

using namespace NeoN;
using namespace NeoN::mpi;

TEST_CASE("fullDuplexBuffer")
{

    Environment mpiEnviron;
    std::vector<std::size_t> rankSendCommSize(mpiEnviron.sizeRank(), 1);
    std::vector<std::size_t> rankReceiveCommSize(mpiEnviron.sizeRank(), 1);
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
        // a finalized buffer cannot be complete
        // REQUIRE(true == buffer.isComplete());
    }

    SECTION("Send and Receive")
    {
        buffer.initComm<int>("Send and Receive");
        for (size_t rank = 0; rank < mpiEnviron.sizeRank(); ++rank)
        {
            auto data = buffer.getSend<int>(rank);
            data[0] = static_cast<int>(rank);
        }

        buffer.startComm();
        buffer.waitComplete();

        for (size_t rank = 0; rank < mpiEnviron.sizeRank(); ++rank)
        {
            auto data = buffer.getReceive<int>(rank);
            REQUIRE(data[0] == static_cast<int>(mpiEnviron.rank()));
        }

        buffer.finaliseComm();
    }
}
