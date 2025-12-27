// SPDX-FileCopyrightText: 2023 - 2025 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "catch2_common.hpp"

#include "NeoN/core/mpi/environment.hpp"
#include "NeoN/distributed/matrix.hpp"
#include <cstring>

using namespace NeoN::mpi;

/* This test uses mpi initialized in the catch_mpi_main  */
TEST_CASE("Distributed Matrix")
{
    Environment mpiEnviron;

    auto [execName, exec] = GENERATE(allAvailableExecutor());

    NeoN::Vector<NeoN::scalar> locVals(exec, {1.0, 5.0, 6.0, 8.0});
    NeoN::Vector<NeoN::localIdx> locColIdx(exec, {0, 1, 2, 1});
    NeoN::Vector<NeoN::localIdx> locRowOffs(exec, {0, 1, 3, 4});

    NeoN::Vector<NeoN::scalar> nonLocVals(exec, {1.0, 5.0, 6.0, 8.0});
    NeoN::Vector<NeoN::localIdx> nonLocColIdx(exec, {0, 1, 2, 1});
    NeoN::Vector<NeoN::localIdx> nonLocRowOffs(exec, {0, 1, 3, 4});

    SECTION("Can instantiate from values")
    {
        NeoN::la::Matrix distMatrix(
            std::move(locVals),
            std::move(locColIdx),
            std::move(locRowOffs),
            std::move(nonLocVals),
            std::move(nonLocColIdx),
            std::move(nonLocRowOffs),
            mpiEnviron
        );
    }
}
