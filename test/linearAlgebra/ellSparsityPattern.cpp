// SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main

#include <limits>

#include "catch2_common.hpp"

#include "NeoN/NeoN.hpp"

namespace NeoN
{

TEST_CASE("EllSparsityPattern")
{
    using EllSparsityType = NeoN::la::EllSparsityPattern<NeoN::localIdx>;
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    // clang-format off
    // Matrix:
    // Row/ColIdx 0   1   2  3
    //   0        x   x
    //   1        x   x   x
    //   2            x   x  x
    //   3                x  x
    //
    // widest row has 3 entries -> numStoredElementsPerRow = 3
    // nRows = 4 -> stride = 4 (no extra padding rows)
    //
    // column-major, padded layout (INV marks unused slots):
    //   slot 0: [0, 0, 1, 2]
    //   slot 1: [1, 1, 2, 3]
    //   slot 2: [INV, 2, 3, INV]
    // clang-format on
    const auto INV = std::numeric_limits<localIdx>::max();
    const localIdx nRows = 4;
    const localIdx numStoredElementsPerRow = 3;
    const localIdx stride = nRows;

    auto colIdxExp = std::vector<localIdx> {
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

    Vector<localIdx> colIdx(exec, colIdxExp);
    auto sp = std::make_shared<EllSparsityType>(
        std::move(colIdx), la::Dimensions {nRows, nRows}, numStoredElementsPerRow, stride
    );

    SECTION("Can store padded colIdxs and dimensions " + execName)
    {
        REQUIRE_THAT(sp->colIdxs(), Equals(colIdxExp, EqualInt()));
        REQUIRE(sp->rows() == nRows);
        REQUIRE(sp->numStoredElementsPerRow() == numStoredElementsPerRow);
        REQUIRE(sp->stride() == stride);
        REQUIRE(sp->nnz() == stride * numStoredElementsPerRow);
        REQUIRE(sp->dimension().rows == nRows);
        REQUIRE(sp->dimension().cols == nRows);
    }

    SECTION("Can copy to executor " + execName)
    {
        auto spOnHost = sp->copyToExecutor(SerialExecutor());
        REQUIRE_THAT(spOnHost.colIdxs(), Equals(colIdxExp, EqualInt()));
        REQUIRE(spOnHost.numStoredElementsPerRow() == numStoredElementsPerRow);
        REQUIRE(spOnHost.stride() == stride);
    }

    SECTION("Can resolve entry offsets on " + execName)
    {
        Vector<localIdx> checkOffset(exec, 4);
        auto checkOffsetView = checkOffset.view();
        auto ellView = sp->view();
        parallelFor(
            exec,
            {0, 1},
            NEON_LAMBDA(const localIdx) {
                checkOffsetView[0] = ellView.entry(0, 0); // row 0, slot 0 -> idx 0 + 4*0 = 0
                checkOffsetView[1] = ellView.entry(1, 2); // row 1, slot 2 -> idx 1 + 4*2 = 9
                checkOffsetView[2] = ellView.entry(2, 3); // row 2, slot 2 -> idx 2 + 4*2 = 10
                checkOffsetView[3] = ellView.entry(3, 3); // row 3, slot 1 -> idx 3 + 4*1 = 7
            }
        );
        auto checkOffsetHost = checkOffset.copyToHost();
        auto checkOffsetHostView = checkOffsetHost.view();
        REQUIRE(checkOffsetHostView[0] == 0);
        REQUIRE(checkOffsetHostView[1] == 9);
        REQUIRE(checkOffsetHostView[2] == 10);
        REQUIRE(checkOffsetHostView[3] == 7);
    }
}

}
