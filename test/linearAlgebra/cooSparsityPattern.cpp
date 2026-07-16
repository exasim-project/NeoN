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
    using CooSparsityType = NeoN::la::CooSparsityPattern<NeoN::localIdx>;
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    auto nCells = 4;

    auto mesh = create1DUniformMesh(exec, nCells);
    auto [sp, mi] = NeoN::la::createSparsityPatternFaceToMatrixAddress<CooSparsityType>(mesh);

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
        auto rowIdxExp = std::vector<localIdx> {0, 0, 1, 1, 1, 2, 2, 2, 3, 3};
        auto colIdxExp = std::vector<localIdx> {0, 1, 0, 1, 2, 1, 2, 3, 2, 3};

        REQUIRE_THAT(sp->rowIdxs(), Equals(rowIdxExp, EqualInt()));
        REQUIRE_THAT(sp->colIdxs(), Equals(colIdxExp, EqualInt()));
    }

    auto bsp = NeoN::la::createBoundarySparsityPattern<CooSparsityType>(mesh, *mi);
    SECTION("Can produce boundary rowOffs and colIdx " + execName)
    {
        REQUIRE_THAT(bsp->rowIdxs(), Equals(std::vector<localIdx> {0, 3}, EqualInt()));
    }

    // Regression: view() used to expose rowIdxs_ where findEntry()/entry() need rowOffs_.
    // Storage offsets 0..9, every row has a diagonal.
    SECTION("view().findEntry() resolves offsets via rowOffs_, not rowIdxs_ " + execName)
    {
        Vector<localIdx> checkOffset(exec, 2);
        auto checkOffsetView = checkOffset.view();
        auto cooView = sp->view();
        parallelFor(
            exec,
            {0, 1},
            NEON_LAMBDA(const localIdx) {
                checkOffsetView[0] = cooView.findEntry(1, 1); // present -> storage offset 3
                checkOffsetView[1] = cooView.findEntry(0, 3); // absent -> invalidIndex()
            }
        );
        auto checkOffsetHost = checkOffset.copyToHost();
        auto checkOffsetHostView = checkOffsetHost.view();
        REQUIRE(checkOffsetHostView[0] == 3);
        REQUIRE(checkOffsetHostView[1] == decltype(cooView)::invalidIndex());
    }

    SECTION("COOMatrix::diag() extracts the correct values " + execName)
    {
        NeoN::Vector<scalar> values(exec, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0});
        NeoN::la::COOMatrix<scalar, localIdx> cooMatrix(values, sp);
        REQUIRE_THAT(cooMatrix.diag(), Equals(I({1.0, 4.0, 7.0, 10.0})));
    }
}

}
