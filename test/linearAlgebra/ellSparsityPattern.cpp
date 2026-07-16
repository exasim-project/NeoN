// SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main

#include <limits>
#include <type_traits>

#include "catch2_common.hpp"

#include "NeoN/NeoN.hpp"

namespace NeoN
{

// Freezes ELL's exclusion from the FaceToMatrixAddress constructor -- CSR row-local
// offsets don't apply to ELL's column-major storage.
static_assert(
    !std::is_constructible_v<
        la::ELLMatrix<scalar, localIdx>,
        Vector<scalar>,
        std::shared_ptr<const la::EllSparsityPattern<localIdx>>,
        std::shared_ptr<const la::FaceToMatrixAddress>>,
    "ELLMatrix must not be constructible with a FaceToMatrixAddress"
);

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
    const localIdx logicalNnz = 10; // true nonzeros: row lengths 2+3+3+2, no padding

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
        std::move(colIdx),
        la::Dimensions {nRows, nRows},
        numStoredElementsPerRow,
        stride,
        logicalNnz
    );

    SECTION("Can store padded colIdxs and dimensions " + execName)
    {
        REQUIRE_THAT(sp->colIdxs(), Equals(colIdxExp, EqualInt()));
        REQUIRE(sp->rows() == nRows);
        REQUIRE(sp->numStoredElementsPerRow() == numStoredElementsPerRow);
        REQUIRE(sp->stride() == stride);
        REQUIRE(sp->storageSize() == stride * numStoredElementsPerRow);
        REQUIRE(sp->nnz() == logicalNnz);
        REQUIRE(sp->nnz() != sp->storageSize());
        REQUIRE(sp->dimension().rows == nRows);
        REQUIRE(sp->dimension().cols == nRows);
    }

    SECTION("Can copy to executor " + execName)
    {
        auto spOnHost = sp->copyToExecutor(SerialExecutor());
        REQUIRE_THAT(spOnHost.colIdxs(), Equals(colIdxExp, EqualInt()));
        REQUIRE(spOnHost.numStoredElementsPerRow() == numStoredElementsPerRow);
        REQUIRE(spOnHost.stride() == stride);
        REQUIRE(spOnHost.nnz() == logicalNnz);
        REQUIRE(spOnHost.storageSize() == stride * numStoredElementsPerRow);
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

    SECTION("findEntry returns invalidIndex() for missing entries " + execName)
    {
        Vector<localIdx> checkFound(exec, 2);
        auto checkFoundView = checkFound.view();
        auto ellView = sp->view();
        parallelFor(
            exec,
            {0, 1},
            NEON_LAMBDA(const localIdx) {
                // row 0 has no entry in column 3 (row 0's neighbours are {0,1})
                checkFoundView[0] = ellView.findEntry(0, 3);
                // row 3 has no entry in column 0 (row 3's neighbours are {2,3})
                checkFoundView[1] = ellView.findEntry(3, 0);
            }
        );
        auto checkFoundHost = checkFound.copyToHost();
        auto checkFoundHostView = checkFoundHost.view();
        REQUIRE(checkFoundHostView[0] == decltype(ellView)::invalidIndex());
        REQUIRE(checkFoundHostView[1] == decltype(ellView)::invalidIndex());
    }
}

TEST_CASE("ELLMatrix")
{
    using EllSparsityType = NeoN::la::EllSparsityPattern<NeoN::localIdx>;
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    // Same 1D 4-cell / 3-internal-face mesh as the EllSparsityPattern test above.
    const localIdx nRows = 4;
    const localIdx numStoredElementsPerRow = 3;
    const localIdx stride = nRows;
    const localIdx logicalNnz = 10;
    const auto INV = std::numeric_limits<localIdx>::max();

    Vector<localIdx> colIdx(exec, std::vector<localIdx> {0, 0, 1, 2, 1, 1, 2, 3, INV, 2, 3, INV});
    auto sp = std::make_shared<const EllSparsityType>(
        std::move(colIdx),
        la::Dimensions {nRows, nRows},
        numStoredElementsPerRow,
        stride,
        logicalNnz
    );

    // storageSize() == 12 (padded); diagonal entries sit at flat offsets 0, 5, 6, 7,
    // so diag() reads 1, 6, 7, 8.
    Vector<scalar> values(
        exec, std::vector<scalar> {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0}
    );

    SECTION("Constructs when values.size() == storageSize(), not nnz() " + execName)
    {
        NeoN::la::ELLMatrix<scalar, localIdx> ellMatrix(values, sp);
        REQUIRE(ellMatrix.values().size() == sp->storageSize());
        REQUIRE(ellMatrix.values().size() != sp->nnz());
        REQUIRE(ellMatrix.nNonZeros() == logicalNnz);
    }

    SECTION("diag() extracts the correct values, generic across sparsity types " + execName)
    {
        NeoN::la::ELLMatrix<scalar, localIdx> ellMatrix(values, sp);
        REQUIRE_THAT(ellMatrix.diag(), Equals(I({1.0, 6.0, 7.0, 8.0})));
    }

    SECTION("view().entry() reads and writes through the generic MatrixView " + execName)
    {
        NeoN::la::ELLMatrix<scalar, localIdx> ellMatrix(values, sp);
        const NeoN::la::ELLMatrix<scalar, localIdx> ellMatrixConst(values, sp);

        Vector<scalar> checkRead(exec, 4);
        auto checkReadView = checkRead.view();
        auto constView = ellMatrixConst.view();
        parallelFor(
            exec,
            {0, 1},
            NEON_LAMBDA(const localIdx) {
                checkReadView[0] = constView.entry(0, 0); // flat offset 0 -> 1.0
                checkReadView[1] = constView.entry(1, 1); // flat offset 5 -> 6.0
                checkReadView[2] = constView.entry(2, 2); // flat offset 6 -> 7.0
                checkReadView[3] = constView.entry(3, 3); // flat offset 7 -> 8.0
            }
        );
        REQUIRE_THAT(checkRead, Equals(I({1.0, 6.0, 7.0, 8.0})));

        auto mutableView = ellMatrix.view();
        parallelFor(
            exec, {0, 1}, NEON_LAMBDA(const localIdx) { mutableView.entry(0, 0) = -1.0; }
        );
        auto writtenHost = ellMatrix.values().copyToHost();
        REQUIRE(writtenHost.view()[0] == -1.0);
    }

    SECTION("copyToExecutor() preserves values and sparsity metadata " + execName)
    {
        NeoN::la::ELLMatrix<scalar, localIdx> ellMatrix(values, sp);
        auto hostMatrix = ellMatrix.copyToExecutor(SerialExecutor());

        REQUIRE(hostMatrix.values().size() == sp->storageSize());
        REQUIRE(hostMatrix.nNonZeros() == logicalNnz);
        REQUIRE(hostMatrix.sparsity()->stride() == stride);
        REQUIRE(hostMatrix.sparsity()->numStoredElementsPerRow() == numStoredElementsPerRow);
        REQUIRE_THAT(hostMatrix.diag(), Equals(I({1.0, 6.0, 7.0, 8.0})));

        Vector<scalar> checkValue(SerialExecutor(), 1);
        auto checkValueView = checkValue.view();
        auto hostView = hostMatrix.view();
        parallelFor(
            SerialExecutor(),
            {0, 1},
            NEON_LAMBDA(const localIdx) { checkValueView[0] = hostView.entry(1, 2); }
        );
        // (row 1, col 2) sits at flat offset 1 + stride*2 = 9 -> values[9] == 10.0
        REQUIRE(checkValueView[0] == 10.0);
    }

    // Same sparse example as the CSR "Can extract diagonal" test in matrix.cpp:
    //   [ 1 .  . ]
    //   [ . 5  6 ]
    //   [ . 8 .  ]
    // row 2 has no diagonal -- diag() should leave it at zero.
    SECTION("diag() leaves a missing diagonal at zero " + execName)
    {
        const localIdx smallNRows = 3;
        const localIdx smallWidth = 2;
        const localIdx smallStride = smallNRows;
        Vector<localIdx> smallColIdx(exec, std::vector<localIdx> {0, 1, 1, INV, 2, INV});
        auto smallSp = std::make_shared<const EllSparsityType>(
            std::move(smallColIdx),
            la::Dimensions {smallNRows, smallNRows},
            smallWidth,
            smallStride,
            4
        );
        Vector<scalar> smallValues(exec, std::vector<scalar> {1.0, 5.0, 8.0, 0.0, 6.0, 0.0});
        NeoN::la::ELLMatrix<scalar, localIdx> smallMatrix(smallValues, smallSp);
        REQUIRE_THAT(smallMatrix.diag(), Equals(I({1.0, 5.0, 0.0})));
    }
}

}
