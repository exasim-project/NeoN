// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "catch2_common.hpp"

#include "NeoN/core/mpi/environment.hpp"
#include "NeoN/distributed/matrix.hpp"
#include <cstring>

using namespace NeoN::mpi;

/* This test uses mpi initialized in the catch_mpi_main  */
TEST_CASE("Distributed Matrix w/o non local part")
{
    NeoN::mpi::Environment mpiEnviron;

    auto [execName, exec] = GENERATE(allAvailableExecutor());

    NeoN::Vector<NeoN::scalar> locVals(exec, {1.0, 5.0, 6.0, 8.0});
    NeoN::Vector<NeoN::localIdx> locColIdx(exec, {0, 1, 2, 1});
    NeoN::Vector<NeoN::localIdx> locRowOffs(exec, {0, 1, 3, 4});

    NeoN::Vector<NeoN::scalar> nonLocVals(exec, {});
    NeoN::Vector<NeoN::localIdx> nonLocColIdx(exec, {});
    NeoN::Vector<NeoN::localIdx> nonLocRowOffs(exec, {});

    auto localMtx = std::make_shared<NeoN::la::CSRMatrix<NeoN::scalar, NeoN::localIdx>>(
        locVals, locColIdx, locRowOffs
    );

    auto nonLocalMtx = std::make_shared<NeoN::la::CSRMatrix<NeoN::scalar, NeoN::localIdx>>(
        nonLocVals, nonLocColIdx, nonLocRowOffs
    );

    SECTION("Can instantiate from local and non-local matrix")
    {
        auto distMatrix = NeoN::la::DistributedMatrix<NeoN::scalar, NeoN::localIdx>(
            localMtx, nonLocalMtx, mpiEnviron
        );
    }
}
