// SPDX-FileCopyrightText: 2024 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main

#include "catch2_common.hpp"

#include "NeoN/NeoN.hpp"

namespace fvcc = NeoN::finiteVolume::cellCentred;

namespace NeoN
{

TEST_CASE("FaceToMatrixAddress")
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    auto nCells = 10;
    auto nFaces = 9;

    // TODO use 2D/3D versions of create1DUniform mesh
    auto mesh = create1DUniformMesh(exec, nCells);
    auto [sp, mi] = NeoN::la::createSparsityPatternFaceToMatrixAddress<
        NeoN::la::CsrSparsityPattern<NeoN::localIdx>>(mesh);

    SECTION("Can construct sparsity pattern " + execName)
    {
        // some basic sanity checks
        REQUIRE(mi->diagOffset().size() == nCells);
        REQUIRE(mi->ownerOffset().size() == nFaces);
        REQUIRE(mi->neighbourOffset().size() == nFaces);
    }

    SECTION("has correct diagOffs" + execName)
    {
        auto exp = std::vector<NeoN::localIdx> {0, 1, 1, 1, 1, 1, 1, 1, 1, 1};
        REQUIRE_THAT(mi->diagOffset(), Equals(exp, EqualInt()));
    }
}

TEST_CASE("EllFaceToMatrixAddress")
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    // Same 1D 4-cell / 3-internal-face mesh used throughout the ELL sparsity/matrix tests.
    auto nCells = 4;
    auto nFaces = 3;
    auto mesh = create1DUniformMesh(exec, nCells);
    auto [sp, mi] = NeoN::la::createSparsityPatternFaceToMatrixAddress<
        NeoN::la::EllSparsityPattern<NeoN::localIdx>>(mesh);

    const auto INV = NeoN::la::EllSparsityView<NeoN::localIdx>::invalidIndex();

    SECTION("Can construct native ELL sparsity pattern " + execName)
    {
        REQUIRE(sp->rows() == nCells);
        REQUIRE(sp->numStoredElementsPerRow() == 3);
        REQUIRE(sp->stride() == nCells);
        REQUIRE(sp->storageSize() == nCells * 3);
        REQUIRE(sp->nnz() == nCells + 2 * nFaces);
    }

    SECTION("Padded, column-major colIdx matches the expected layout " + execName)
    {
        auto colIdxExp = std::vector<NeoN::localIdx> {
            0,
            0,
            1,
            2, // slot 0
            1,
            1,
            2,
            3, // slot 1
            INV,
            2,
            3,
            INV // slot 2
        };
        REQUIRE_THAT(sp->colIdxs(), Equals(colIdxExp, EqualInt()));
    }

    SECTION("has correct diagOffs " + execName)
    {
        auto exp = std::vector<NeoN::localIdx> {0, 1, 1, 1};
        REQUIRE_THAT(mi->diagOffset(), Equals(exp, EqualInt()));
    }

    SECTION("has correct ownerOffset and neighbourOffset " + execName)
    {
        // position-within-row data, format-independent -- same values the CSR builder
        // produces for this mesh.
        REQUIRE_THAT(mi->ownerOffset(), Equals(std::vector<NeoN::localIdx> {1, 2, 2}, EqualInt()));
        REQUIRE_THAT(
            mi->neighbourOffset(), Equals(std::vector<NeoN::localIdx> {0, 0, 0}, EqualInt())
        );
    }

    SECTION("EllFaceToMatrixView resolves the same flat offsets as the padded colIdx " + execName)
    {
        // sparsity-view-typed overload -- format picked by overload resolution.
        auto ellView = mi->view(sp->view());

        Vector<NeoN::localIdx> checkDiag(exec, nCells);
        Vector<NeoN::localIdx> checkUpper(exec, nFaces);
        Vector<NeoN::localIdx> checkLower(exec, nFaces);
        auto checkDiagView = checkDiag.view();
        auto checkUpperView = checkUpper.view();
        auto checkLowerView = checkLower.view();
        parallelFor(
            exec,
            {0, 1},
            NEON_LAMBDA(const NeoN::localIdx) {
                for (NeoN::localIdx celli = 0; celli < nCells; ++celli)
                {
                    checkDiagView[celli] = ellView.diagIdx(celli);
                }
                // faces: 0 owns(0,1), 1 owns(1,2), 2 owns(2,3) -- own = facei, nei = facei+1
                for (NeoN::localIdx facei = 0; facei < nFaces; ++facei)
                {
                    checkUpperView[facei] = ellView.upperIdx(facei, facei);
                    checkLowerView[facei] = ellView.lowerIdx(facei + 1, facei);
                }
            }
        );

        // matches the padded colIdx layout: diagonal at 0,5,6,7; upper at 4,9,10; lower at 1,2,3
        REQUIRE_THAT(checkDiag, Equals(std::vector<NeoN::localIdx> {0, 5, 6, 7}));
        REQUIRE_THAT(checkUpper, Equals(std::vector<NeoN::localIdx> {4, 9, 10}));
        REQUIRE_THAT(checkLower, Equals(std::vector<NeoN::localIdx> {1, 2, 3}));
    }
}

}
