// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#define CATCH_CONFIG_RUNNER
#include "catch2_common.hpp"

#include "NeoN/NeoN.hpp"

using NeoN::scalar;
using NeoN::localIdx;
using NeoN::Vector;
using NeoN::la::CSRMatrix;
using NeoN::la::COOMatrix;
using NeoN::la::LinearSystem;

// ---------------------------------------------------------------------------
// Shared setup: 3-cell tridiagonal CSR with cell 1 pinned to 7.
//
//   row 0: [10, -1]        row 1 (pinned): [-2, 10, -2]   row 2: [-1, 10]
//   rhs = [5, 5, 5],  mask = [0, 1, 0],  pinValues = [0, 7, 0]
//
// Expected after FixedValueConstraints:
//   mat values: [10, 0,  0, 10, 0,  0, 10]
//   rhs: [5-(-1)*7, 10*7, 5-(-1)*7] = [12, 70, 12]
// ---------------------------------------------------------------------------

TEST_CASE("FixedValueConstraints")
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    SECTION("CSR row/column pinning - " + execName)
    {
        Vector<scalar> matVals(exec, {10.0, -1.0, -2.0, 10.0, -2.0, -1.0, 10.0});
        Vector<localIdx> colIdxs(exec, {0, 1, 0, 1, 2, 1, 2});
        Vector<localIdx> rowOffs(exec, {0, 2, 5, 7});
        CSRMatrix<scalar, localIdx> csr(matVals, colIdxs, rowOffs, {3, 3});

        Vector<scalar> rhs(exec, {5.0, 5.0, 5.0});

        // empty boundary (size-0 COO + rhs)
        auto emptySp = std::make_shared<const NeoN::la::CooSparsityPattern<localIdx>>(
            Vector<localIdx>(exec, 0), Vector<localIdx>(exec, 0), NeoN::la::Dimensions {0, 0}
        );
        COOMatrix<scalar, localIdx> emptyB(Vector<scalar>(exec, 0, 0.0), emptySp);
        Vector<scalar> bRhs(exec, 3, 0.0);

        LinearSystem<scalar> ls(csr, rhs, emptyB, bRhs);

        Vector<scalar> mask(exec, {0.0, 1.0, 0.0});
        Vector<scalar> pinVals(exec, {0.0, 7.0, 0.0});
        NeoN::dsl::FixedValueConstraints<scalar> pin(mask.view(), pinVals.view(), 3);
        pin(ls);

        auto mH = ls.matrix().values().copyToHost();
        auto rH = ls.rhs().copyToHost();

        // row 0: diagonal kept, col-1 cut (absorbs -(-1)*7 into rhs)
        CHECK(mH.view()[0] == Catch::Approx(10.0));
        CHECK(mH.view()[1] == Catch::Approx(0.0));
        // row 1 (pinned): off-diagonals zeroed, diagonal kept
        CHECK(mH.view()[2] == Catch::Approx(0.0));
        CHECK(mH.view()[3] == Catch::Approx(10.0));
        CHECK(mH.view()[4] == Catch::Approx(0.0));
        // row 2: col-1 cut, diagonal kept
        CHECK(mH.view()[5] == Catch::Approx(0.0));
        CHECK(mH.view()[6] == Catch::Approx(10.0));

        CHECK(rH.view()[0] == Catch::Approx(12.0));
        CHECK(rH.view()[1] == Catch::Approx(70.0));
        CHECK(rH.view()[2] == Catch::Approx(12.0));
    }

    SECTION("offDiagonalMatrix zeroed for pinned rows - " + execName)
    {
        Vector<scalar> matVals(exec, {10.0, -1.0, -2.0, 10.0, -2.0, -1.0, 10.0});
        Vector<localIdx> colIdxs(exec, {0, 1, 0, 1, 2, 1, 2});
        Vector<localIdx> rowOffs(exec, {0, 2, 5, 7});
        CSRMatrix<scalar, localIdx> csr(matVals, colIdxs, rowOffs, {3, 3});
        Vector<scalar> rhs(exec, {5.0, 5.0, 5.0});

        // offDiag: (row=1, col=3, val=5.0) pinned → must be zeroed
        //          (row=0, col=4, val=3.0) not pinned → unchanged
        auto offSp = std::make_shared<const NeoN::la::CooSparsityPattern<localIdx>>(
            Vector<localIdx>(exec, {3, 4}), // colIdxs
            Vector<localIdx>(exec, {1, 0}), // rowIdxs
            NeoN::la::Dimensions {3, 5}
        );
        COOMatrix<scalar, localIdx> offDiag(Vector<scalar>(exec, {5.0, 3.0}), offSp);

        auto emptySp = std::make_shared<const NeoN::la::CooSparsityPattern<localIdx>>(
            Vector<localIdx>(exec, 0), Vector<localIdx>(exec, 0), NeoN::la::Dimensions {0, 0}
        );
        COOMatrix<scalar, localIdx> emptyB(Vector<scalar>(exec, 0, 0.0), emptySp);
        Vector<scalar> bRhs(exec, 3, 0.0);

        LinearSystem<scalar> ls(csr, rhs, offDiag, emptyB, bRhs);

        Vector<scalar> mask(exec, {0.0, 1.0, 0.0});
        Vector<scalar> pinVals(exec, {0.0, 7.0, 0.0});
        NeoN::dsl::FixedValueConstraints<scalar> pin(mask.view(), pinVals.view(), 3);
        pin(ls);

        auto offH = ls.offDiagonalMatrix().values().copyToHost();

        CHECK(offH.view()[0] == Catch::Approx(0.0)); // row=1 (pinned) → zeroed
        CHECK(offH.view()[1] == Catch::Approx(3.0)); // row=0 (not pinned) → unchanged
    }
}
