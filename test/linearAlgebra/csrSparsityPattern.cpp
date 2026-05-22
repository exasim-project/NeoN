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

TEST_CASE("SparsityPattern")
{
    using CsrSparsityType = NeoN::la::CsrSparsityPattern<NeoN::localIdx>;
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    auto nCells = 4;

    auto mesh = create1DUniformMesh(exec, nCells);
    auto [sp, mi] = NeoN::la::createSparsityPatternFaceToMatrixAddress<CsrSparsityType>(mesh);

    // clang-format off
    // Mesh:
    // Cell Ids [ 0, 1, 2, 3]
    //
    // Matrix:
    // Row/ColIdx 0   1   2  3
    //   0        x   x
    //   1        x   x   x
    //   2            x   x  x
    //   3                x  x
    // clang-format on
    SECTION("Can produce internal rowOffs and colIdx " + execName)
    {
        auto rowPtrExp = std::vector<localIdx> {0, 2, 5, 8, 10};
        auto colIdxExp = std::vector<localIdx> {0, 1, 0, 1, 2, 1, 2, 3, 2, 3};

        REQUIRE_THAT(sp->rowOffs(), Equals(rowPtrExp, EqualInt()));
        REQUIRE_THAT(sp->colIdxs(), Equals(colIdxExp, EqualInt()));
    }

    auto bsp = NeoN::la::createBoundarySparsityPattern<CsrSparsityType>(mesh, *mi);
    SECTION("Can produce boundary rowOffs and colIdx " + execName)
    {
        REQUIRE_THAT(bsp->rowOffs(), Equals(std::vector<localIdx> {0, 1, 1, 1, 2}, EqualInt()));
        // REQUIRE_THAT(bsp->colIdxs(), Equals(std::vector<localIdx> {0, 3}, EqualInt()));
    }
}

}
