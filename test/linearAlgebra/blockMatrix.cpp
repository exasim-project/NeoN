// SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#define CATCH_CONFIG_RUNNER
#include "catch2_common.hpp"

#include "NeoN/NeoN.hpp"
#include "NeoN/linearAlgebra/blockMatrix.hpp"

TEST_CASE("BlockMatrix")
{
    using namespace NeoN;
    using namespace NeoN::la;

    auto [execName, exec] = GENERATE(allAvailableExecutor());

    // Helper: 1-cell sparsity (single diagonal entry, nnz=1)
    auto sp1 = std::make_shared<SparsityPattern<localIdx>>(
        Vector<localIdx>(exec, std::vector<localIdx> {0}),
        Vector<localIdx>(exec, std::vector<localIdx> {0, 1})
    );

    SECTION("Construction from nBlocks and sparsity " + execName)
    {
        BlockMatrix bm(exec, 2, sp1);

        REQUIRE(bm.nBlocks() == 2);
        REQUIRE(bm.nCells() == 1);
        REQUIRE(bm.nnz() == 1);
        REQUIRE(bm.totalSize() == 2);
        REQUIRE(bm.values().size() == 4); // nnz * nBlocks^2 = 1 * 4
    }

    // Monolithic CSR layout: for 1 cell, 1 nnz, 2 blocks:
    // monoRowOffs = {0, 2, 4}, values = [coupling(0,0), coupling(0,1), coupling(1,0),
    // coupling(1,1)]
    SECTION("Construct with values " + execName)
    {
        Vector<scalar> vals(exec, std::vector<scalar> {4.0, 1.0, 1.0, 3.0});
        BlockMatrix bm(exec, 2, sp1, vals);

        auto hostVals = bm.values().copyToHost();
        REQUIRE(hostVals.view()[0] == 4.0);
        REQUIRE(hostVals.view()[1] == 1.0);
        REQUIRE(hostVals.view()[2] == 1.0);
        REQUIRE(hostVals.view()[3] == 3.0);
    }

    SECTION("BlockView operator()(i,j) read access " + execName)
    {
        Vector<scalar> vals(exec, std::vector<scalar> {4.0, 1.0, 1.0, 3.0});
        BlockMatrix bm(exec, 2, sp1, vals);

        auto bmView = bm.view();
        Vector<scalar> result(exec, 4, 0.0);
        auto rv = result.view();

        parallelFor(
            exec,
            {0, 1},
            NEON_LAMBDA(const localIdx) {
                auto coupling = bmView(0); // coupling matrix at CSR position 0
                rv[0] = coupling(0, 0);    // 4.0
                rv[1] = coupling(1, 0);    // 1.0
                rv[2] = coupling(0, 1);    // 1.0
                rv[3] = coupling(1, 1);    // 3.0
            },
            "BlockView_read"
        );

        auto hostResult = result.copyToHost();
        REQUIRE(hostResult.view()[0] == 4.0);
        REQUIRE(hostResult.view()[1] == 1.0);
        REQUIRE(hostResult.view()[2] == 1.0);
        REQUIRE(hostResult.view()[3] == 3.0);
    }

    SECTION("BlockView operator()(i,j) write access " + execName)
    {
        BlockMatrix bm(exec, 2, sp1);

        auto bmView = bm.view();

        parallelFor(
            exec,
            {0, 1},
            NEON_LAMBDA(const localIdx) {
                auto coupling = bmView(0);
                coupling(0, 0) = 4.0;
                coupling(1, 0) = 1.0;
                coupling(0, 1) = 1.0;
                coupling(1, 1) = 3.0;
            },
            "BlockView_write"
        );

        auto hostVals = bm.values().copyToHost();
        REQUIRE(hostVals.view()[0] == 4.0);
        REQUIRE(hostVals.view()[1] == 1.0);
        REQUIRE(hostVals.view()[2] == 1.0);
        REQUIRE(hostVals.view()[3] == 3.0);
    }

    SECTION("BlockView flat offset access " + execName)
    {
        Vector<scalar> vals(exec, std::vector<scalar> {4.0, 1.0, 1.0, 3.0});
        BlockMatrix bm(exec, 2, sp1, vals);

        auto bmView = bm.view();
        Vector<scalar> result(exec, 4, 0.0);
        auto rv = result.view();

        parallelFor(
            exec,
            {0, 1},
            NEON_LAMBDA(const localIdx) {
                auto coupling = bmView(0);
                rv[0] = coupling[0]; // 4.0
                rv[1] = coupling[1]; // 1.0
                rv[2] = coupling[2]; // 1.0
                rv[3] = coupling[3]; // 3.0
            },
            "BlockView_flat"
        );

        auto hostResult = result.copyToHost();
        REQUIRE(hostResult.view()[0] == 4.0);
        REQUIRE(hostResult.view()[1] == 1.0);
        REQUIRE(hostResult.view()[2] == 1.0);
        REQUIRE(hostResult.view()[3] == 3.0);
    }

    SECTION("BlockRowView single row " + execName)
    {
        // Coupling at pos 0: (0,0)=4, (1,0)=1, (0,1)=1, (1,1)=3
        Vector<scalar> vals(exec, std::vector<scalar> {4.0, 1.0, 1.0, 3.0});
        BlockMatrix bm(exec, 2, sp1, vals);

        auto bmView = bm.view();
        Vector<scalar> result(exec, 4, 0.0);
        auto rv = result.view();

        parallelFor(
            exec,
            {0, 1},
            NEON_LAMBDA(const localIdx) {
                // Row 0 of coupling at pos 0: [(0,0)=4, (0,1)=1]
                auto r0 = bmView.rowView(0, 0, 1);
                rv[0] = r0(0, 0); // 4.0
                rv[1] = r0(0, 1); // 1.0

                // Row 1 of coupling at pos 0: [(1,0)=1, (1,1)=3]
                auto r1 = bmView.rowView(0, 1, 2);
                rv[2] = r1(0, 0); // 1.0
                rv[3] = r1(0, 1); // 3.0
            },
            "BlockRowView_single"
        );

        auto hostResult = result.copyToHost();
        REQUIRE(hostResult.view()[0] == 4.0);
        REQUIRE(hostResult.view()[1] == 1.0);
        REQUIRE(hostResult.view()[2] == 1.0);
        REQUIRE(hostResult.view()[3] == 3.0);
    }

    SECTION("BlockRowView all rows " + execName)
    {
        Vector<scalar> vals(exec, std::vector<scalar> {4.0, 1.0, 1.0, 3.0});
        BlockMatrix bm(exec, 2, sp1, vals);

        auto bmView = bm.view();
        Vector<scalar> result(exec, 4, 0.0);
        auto rv = result.view();

        parallelFor(
            exec,
            {0, 1},
            NEON_LAMBDA(const localIdx) {
                // All rows of coupling at pos 0 (full 2x2 matrix)
                auto full = bmView.rowView(0, 0, 2);
                rv[0] = full(0, 0); // 4.0
                rv[1] = full(1, 0); // 1.0
                rv[2] = full(0, 1); // 1.0
                rv[3] = full(1, 1); // 3.0
            },
            "BlockRowView_all"
        );

        auto hostResult = result.copyToHost();
        REQUIRE(hostResult.view()[0] == 4.0);
        REQUIRE(hostResult.view()[1] == 1.0);
        REQUIRE(hostResult.view()[2] == 1.0);
        REQUIRE(hostResult.view()[3] == 3.0);
    }

    SECTION("BlockRowView write access " + execName)
    {
        BlockMatrix bm(exec, 2, sp1);

        auto bmView = bm.view();

        parallelFor(
            exec,
            {0, 1},
            NEON_LAMBDA(const localIdx) {
                auto r0 = bmView.rowView(0, 0, 1);
                r0(0, 0) = 4.0;
                r0(0, 1) = 1.0;

                auto r1 = bmView.rowView(0, 1, 2);
                r1(0, 0) = 1.0;
                r1(0, 1) = 3.0;
            },
            "BlockRowView_write"
        );

        auto hostVals = bm.values().copyToHost();
        REQUIRE(hostVals.view()[0] == 4.0);
        REQUIRE(hostVals.view()[1] == 1.0);
        REQUIRE(hostVals.view()[2] == 1.0);
        REQUIRE(hostVals.view()[3] == 3.0);
    }
}

TEST_CASE("BlockMatrix - multi-cell")
{
    using namespace NeoN;
    using namespace NeoN::la;

    auto [execName, exec] = GENERATE(allAvailableExecutor());

    SECTION("3-cell tridiagonal 2x2 block " + execName)
    {
        // 3-cell tridiagonal sparsity: nnz = 7
        // colIdxs = {0,1, 0,1,2, 1,2}, rowOffs = {0,2,5,7}
        auto sp3 = std::make_shared<SparsityPattern<localIdx>>(
            Vector<localIdx>(exec, std::vector<localIdx> {0, 1, 0, 1, 2, 1, 2}),
            Vector<localIdx>(exec, std::vector<localIdx> {0, 2, 5, 7})
        );

        // Interleaved layout: at each CSR position, a column-major nBlocks x nBlocks coupling
        // Diagonal couplings: (0,0)=2, (1,0)=0.5, (0,1)=0.5, (1,1)=3
        // Off-diagonal couplings: (0,0)=-1, (1,0)=0, (0,1)=0, (1,1)=-1
        std::vector<scalar> valsVec = {
            2,  0.5, 0.5, 3,  // pos 0: cell 0->0 diagonal
            -1, 0,   0,   -1, // pos 1: cell 0->1 off-diagonal
            -1, 0,   0,   -1, // pos 2: cell 1->0 off-diagonal
            2,  0.5, 0.5, 3,  // pos 3: cell 1->1 diagonal
            -1, 0,   0,   -1, // pos 4: cell 1->2 off-diagonal
            -1, 0,   0,   -1, // pos 5: cell 2->1 off-diagonal
            2,  0.5, 0.5, 3   // pos 6: cell 2->2 diagonal
        };
        Vector<scalar> vals(exec, valsVec);
        BlockMatrix bm(exec, 2, sp3, vals);

        REQUIRE(bm.nBlocks() == 2);
        REQUIRE(bm.nCells() == 3);
        REQUIRE(bm.nnz() == 7);
        REQUIRE(bm.values().size() == 28);

        auto bmView = bm.view();
        Vector<scalar> result(exec, 8, 0.0);
        auto rv = result.view();

        parallelFor(
            exec,
            {0, 1},
            NEON_LAMBDA(const localIdx) {
                // Coupling at CSR position 0 (cell 0->0 diagonal)
                auto c0 = bmView(0);
                rv[0] = c0(0, 0); // 2.0
                rv[1] = c0(1, 0); // 0.5
                rv[2] = c0(0, 1); // 0.5
                rv[3] = c0(1, 1); // 3.0

                // Coupling at CSR position 1 (cell 0->1 off-diagonal)
                auto c1 = bmView(1);
                rv[4] = c1(0, 0); // -1.0
                rv[5] = c1(1, 0); // 0.0
                rv[6] = c1(0, 1); // 0.0
                rv[7] = c1(1, 1); // -1.0
            },
            "MultiCell_read"
        );

        auto hostResult = result.copyToHost();
        REQUIRE(hostResult.view()[0] == 2.0);
        REQUIRE(hostResult.view()[1] == 0.5);
        REQUIRE(hostResult.view()[2] == 0.5);
        REQUIRE(hostResult.view()[3] == 3.0);
        REQUIRE(hostResult.view()[4] == -1.0);
        REQUIRE(hostResult.view()[5] == 0.0);
        REQUIRE(hostResult.view()[6] == 0.0);
        REQUIRE(hostResult.view()[7] == -1.0);
    }

    SECTION("3-cell rowView access " + execName)
    {
        auto sp3 = std::make_shared<SparsityPattern<localIdx>>(
            Vector<localIdx>(exec, std::vector<localIdx> {0, 1, 0, 1, 2, 1, 2}),
            Vector<localIdx>(exec, std::vector<localIdx> {0, 2, 5, 7})
        );

        // Interleaved layout: at each CSR position, a column-major nBlocks x nBlocks coupling
        std::vector<scalar> valsVec = {
            2,  0.5, 0.5, 3,  // pos 0: cell 0->0 diagonal
            -1, 0,   0,   -1, // pos 1: cell 0->1 off-diagonal
            -1, 0,   0,   -1, // pos 2: cell 1->0 off-diagonal
            2,  0.5, 0.5, 3,  // pos 3: cell 1->1 diagonal
            -1, 0,   0,   -1, // pos 4: cell 1->2 off-diagonal
            -1, 0,   0,   -1, // pos 5: cell 2->1 off-diagonal
            2,  0.5, 0.5, 3   // pos 6: cell 2->2 diagonal
        };
        Vector<scalar> vals(exec, valsVec);
        BlockMatrix bm(exec, 2, sp3, vals);

        auto bmView = bm.view();
        Vector<scalar> result(exec, 8, 0.0);
        auto rv = result.view();

        parallelFor(
            exec,
            {0, 1},
            NEON_LAMBDA(const localIdx) {
                // Diagonal coupling at pos 3 (cell 1->1): [2, 0.5, 0.5, 3]
                // Row 0 of this coupling: (0,0)=2, (0,1)=0.5
                auto r0 = bmView.rowView(3, 0, 1);
                rv[0] = r0(0, 0); // 2.0
                rv[1] = r0(0, 1); // 0.5

                // Row 1 of this coupling: (1,0)=0.5, (1,1)=3
                auto r1 = bmView.rowView(3, 1, 2);
                rv[2] = r1(0, 0); // 0.5
                rv[3] = r1(0, 1); // 3.0

                // Off-diagonal coupling at pos 1 (cell 0->1): [-1, 0, 0, -1]
                // Both rows (full 2x2)
                auto full = bmView.rowView(1, 0, 2);
                rv[4] = full(0, 0); // -1.0
                rv[5] = full(1, 0); // 0.0
                rv[6] = full(0, 1); // 0.0
                rv[7] = full(1, 1); // -1.0
            },
            "MultiCell_rowView"
        );

        auto hostResult = result.copyToHost();
        REQUIRE(hostResult.view()[0] == 2.0);
        REQUIRE(hostResult.view()[1] == 0.5);
        REQUIRE(hostResult.view()[2] == 0.5);
        REQUIRE(hostResult.view()[3] == 3.0);
        REQUIRE(hostResult.view()[4] == -1.0);
        REQUIRE(hostResult.view()[5] == 0.0);
        REQUIRE(hostResult.view()[6] == 0.0);
        REQUIRE(hostResult.view()[7] == -1.0);
    }
}
