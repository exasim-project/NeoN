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

}
