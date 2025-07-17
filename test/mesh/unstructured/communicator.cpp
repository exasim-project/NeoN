// SPDX-FileCopyrightText: 2023 - 2025 NeoN authors
//
// SPDX-License-Identifier: MIT

// #include <source_location>

#include "catch2_common.hpp"

#include "NeoN/NeoN.hpp"


using namespace NeoN;

TEST_CASE("Communicator Vector Synchronization")
{
    mpi::MPIEnvironment mpiEnviron;
    Communicator comm;

    // first block send (size rank)
    // second block remains the same
    // third block receive (size rank)
    Vector<int> field(SerialExecutor(), 3 * mpiEnviron.sizeRank());

    for (size_t rank = 0; rank < mpiEnviron.sizeRank(); rank++)
    {
        // we send the rank numbers
        field(rank) = static_cast<int>(rank);

        // just make sure its not a communicated value.
        field(rank + mpiEnviron.sizeRank()) = static_cast<int>(mpiEnviron.sizeRank() + rank);

        // set to 0.0 to check if the value is communicated
        field(rank + 2 * mpiEnviron.sizeRank()) = 0;
    }

    // Set up buffer to local map, we will ignore global_idx
    CommMap rankSendMap(mpiEnviron.sizeRank());
    CommMap rankReceiveMap(mpiEnviron.sizeRank());
    for (size_t rank = 0; rank < mpiEnviron.sizeRank(); rank++)
    {
        rankSendMap[rank].emplace_back(NodeCommMap {.local_idx = static_cast<label>(rank)});
        NodeCommMap newNode({.local_idx = static_cast<label>(2 * mpiEnviron.sizeRank() + rank)});
        rankReceiveMap[rank].push_back(newNode); // got tired of fighting with clang-format.
    }

    // Communicate
    comm = Communicator(mpiEnviron, rankSendMap, rankReceiveMap);
    std::string loc = "foo";
    // std::source_location::current().file_name() + std::source_location::current().line();
    comm.startComm(field, loc);
    comm.isComplete(loc); // just call it to make sure it doesn't crash
    comm.finaliseComm(field, loc);

    // Check the values
    for (size_t rank = 0; rank < mpiEnviron.sizeRank(); rank++)
    {
        REQUIRE(field(rank) == static_cast<int>(rank));
        REQUIRE(
            field(rank + mpiEnviron.sizeRank()) == static_cast<int>(mpiEnviron.sizeRank() + rank)
        );
        REQUIRE(field(rank + 2 * mpiEnviron.sizeRank()) == static_cast<int>(mpiEnviron.rank()));
    }
}
