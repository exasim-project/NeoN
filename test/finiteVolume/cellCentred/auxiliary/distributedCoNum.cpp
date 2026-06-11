// SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "catch2_common.hpp"

#include "NeoN/NeoN.hpp"

namespace fvcc = NeoN::finiteVolume::cellCentred;

TEST_CASE("Distributed Courant Number")
{
    NeoN::mpi::Environment mpiEnviron;

    auto [execName, exec] = GENERATE(allAvailableExecutor());

    SECTION("distributed computeCoNum matches serial result on partitioned 1D mesh: " + execName)
    {
        const NeoN::localIdx nCells = 12; // 3 ranks × 4 cells each

        // Serial reference: non-distributed global mesh, MPI allReduce block is skipped
        auto meshGlobal = NeoN::create1DUniformMesh(exec, nCells);

        std::vector<fvcc::SurfaceBoundary<NeoN::scalar>> bcsGlobal;
        for (auto patchi : I<NeoN::localIdx> {0, 1})
        {
            NeoN::Dictionary dict;
            dict.insert("type", std::string("fixedValue"));
            dict.insert("fixedValue", 1.0);
            bcsGlobal.push_back(fvcc::SurfaceBoundary<NeoN::scalar>(meshGlobal, dict, patchi));
        }
        fvcc::SurfaceField<NeoN::scalar> sfGlobal(exec, "sf", meshGlobal, bcsGlobal);
        NeoN::fill(sfGlobal.internalVector(), 1.0);
        sfGlobal.correctBoundaryConditions();

        const auto [maxCoNumRef, meanCoNumRef] = fvcc::computeCoNum(sfGlobal, 0.01);

        // Distributed: each rank owns nCells/sizeRank cells
        auto meshPart = NeoN::create1DUniformMeshPart(exec, nCells / mpiEnviron.sizeRank());
        auto sfPart = NeoN::oneDPartitionField(sfGlobal, meshPart, mpiEnviron);

        const auto [maxCoNum, meanCoNum] = fvcc::computeCoNum(sfPart, 0.01);

        REQUIRE(maxCoNum == Catch::Approx(maxCoNumRef));
        REQUIRE(meanCoNum == Catch::Approx(meanCoNumRef));
    }
}
