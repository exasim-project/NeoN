// SPDX-FileCopyrightText: 2024 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "common.hpp"

#ifdef NF_WITH_MPI_SUPPORT

TEST_CASE("DistributedLaplacianOperator::2D_strong", "[bench]")
{
    auto [minGlobal, maxGlobal] = createStrongScalingSizeFromEnv();
    auto [execName, exec] = createExecutorFromEnv();

    NeoN::mpi::Environment mpiEnviron;
    const auto totalRanks = static_cast<NeoN::localIdx>(mpiEnviron.sizeRank());
    const auto rank = static_cast<NeoN::localIdx>(mpiEnviron.rank());

    NeoN::localIdx ranksXPart = 1;
    while (ranksXPart * ranksXPart < totalRanks)
        ++ranksXPart;
    if (ranksXPart * ranksXPart != totalRanks)
    {
        WARN(
            "DistributedLaplacianOperator::2D_strong requires a square number of ranks (1, 4, 9, "
            "...); skipping for "
            + std::to_string(totalRanks) + " ranks."
        );
        return;
    }

    for (NeoN::localIdx nGlobal = minGlobal; nGlobal <= maxGlobal; nGlobal *= 2)
    {
        if (nGlobal % ranksXPart != 0) continue;
        const auto nCellsPerRank = nGlobal / ranksXPart;
        NeoN::UnstructuredMesh mesh =
            NeoN::create2DUniformMeshPart(exec, nCellsPerRank, totalRanks, rank);
        const std::string sectionName = std::to_string(totalRanks) + " ranks - "
                                      + std::to_string(nGlobal) + "x" + std::to_string(nGlobal)
                                      + " global";
        runDistributedSingleOperatorBenchmark<NeoN::Vec3>(execName, exec, mesh, sectionName);
        runDistributedPoissonBenchmark(execName, exec, mesh, sectionName);
        runDistributedMomentumBenchmark(execName, exec, mesh, sectionName);
    }
}

TEST_CASE("DistributedLaplacianOperator::2DX_strong", "[bench]")
{
    auto [minGlobal, maxGlobal] = createStrongScalingSizeFromEnv();
    auto [execName, exec] = createExecutorFromEnv();

    NeoN::mpi::Environment mpiEnviron;
    const auto totalRanks = static_cast<NeoN::localIdx>(mpiEnviron.sizeRank());
    const auto rank = static_cast<NeoN::localIdx>(mpiEnviron.rank());

    for (NeoN::localIdx nGlobal = minGlobal; nGlobal <= maxGlobal; nGlobal *= 2)
    {
        if (nGlobal % totalRanks != 0) continue;
        const auto nCellsPerRank = nGlobal / totalRanks;
        NeoN::UnstructuredMesh mesh =
            NeoN::create2DUniformMeshPartX(exec, nCellsPerRank, totalRanks, rank);
        const std::string sectionName = std::to_string(totalRanks) + " ranks - "
                                      + std::to_string(nGlobal) + "x" + std::to_string(nGlobal)
                                      + " global";
        runDistributedSingleOperatorBenchmark<NeoN::Vec3>(execName, exec, mesh, sectionName);
        runDistributedPoissonBenchmark(execName, exec, mesh, sectionName);
        runDistributedMomentumBenchmark(execName, exec, mesh, sectionName);
    }
}

#endif // NF_WITH_MPI_SUPPORT
