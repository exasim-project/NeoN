// SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "catch2_common.hpp"

#include "NeoN/NeoN.hpp"

namespace fvcc = NeoN::finiteVolume::cellCentred;

// reconstruct() accumulates Sf(x)Sf/|Sf| and (Sf/|Sf|)*ssf over every face touching a cell.
// Processor faces occupy the trailing boundary slots and are easy to miss: dropping them makes
// the result of a cell next to a partition interface differ from the serial answer. The stored
// proc-face flux also keeps the global owner->neighbour sense while the boundary normal points
// out of the LOCAL cell, so the sign must be corrected from boundaryMesh().weights() (same
// convention as BoundedDiv). This test compares each rank's cells against the serial reference.
TEST_CASE("Distributed reconstruct")
{
    NeoN::mpi::Environment mpiEnviron;

    auto [execName, exec] = GENERATE(allAvailableExecutor());

    SECTION("distributed reconstruct matches serial result on partitioned 1D mesh: " + execName)
    {
        const NeoN::localIdx nCells = 12; // 3 ranks x 4 cells each

        auto meshGlobal = NeoN::create1DUniformMesh(exec, nCells);

        std::vector<fvcc::SurfaceBoundary<NeoN::scalar>> bcsGlobal;
        for (NeoN::localIdx patchi = 0; patchi < meshGlobal.nBoundaries(); ++patchi)
        {
            NeoN::Dictionary dict;
            dict.insert("type", std::string("fixedValue"));
            dict.insert("fixedValue", 1.0);
            bcsGlobal.push_back(fvcc::SurfaceBoundary<NeoN::scalar>(meshGlobal, dict, patchi));
        }
        fvcc::SurfaceField<NeoN::scalar> sfGlobal(exec, "sf", meshGlobal, bcsGlobal);
        // A face-varying flux is essential here: with a uniform flux every 1D cell reconstructs
        // to the same value whether or not the interface face is counted, so a uniform field
        // cannot tell a correct proc-face contribution from a missing (or sign-flipped) one.
        auto sfGlobalV = sfGlobal.internalVector().view();
        NeoN::parallelFor(
            exec,
            {0, meshGlobal.nInternalFaces()},
            NEON_LAMBDA(const NeoN::localIdx f) {
                sfGlobalV[f] = 1.0 + static_cast<NeoN::scalar>(f);
            }
        );
        sfGlobal.correctBoundaryConditions();

        auto refField = fvcc::reconstruct(sfGlobal);
        auto refHost = refField.internalVector().copyToHost();
        auto refView = refHost.view();

        auto meshPart = NeoN::create1DUniformMeshPart(exec, nCells / mpiEnviron.sizeRank());
        auto sfPart = NeoN::oneDPartitionField(sfGlobal, meshPart, mpiEnviron);

        auto partField = fvcc::reconstruct(sfPart);
        auto partHost = partField.internalVector().copyToHost();
        auto partView = partHost.view();

        const NeoN::localIdx nLocal = nCells / mpiEnviron.sizeRank();
        REQUIRE(partHost.size() == nLocal);

        const auto offset = static_cast<NeoN::localIdx>(mpiEnviron.rank()) * nLocal;
        for (NeoN::localIdx i = 0; i < nLocal; ++i)
        {
            REQUIRE(partView[i][0] == Catch::Approx(refView[offset + i][0]));
            REQUIRE(partView[i][1] == Catch::Approx(refView[offset + i][1]));
            REQUIRE(partView[i][2] == Catch::Approx(refView[offset + i][2]));
        }
    }
}
