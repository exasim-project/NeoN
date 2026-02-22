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

    // Helper: 1-cell sparsity (single diagonal entry)
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
        REQUIRE(bm.values().size() == 4); // 2^2 * 1

        auto hostVals = bm.values().copyToHost();
        REQUIRE(hostVals.view()[0] == 0.0);
        REQUIRE(hostVals.view()[1] == 0.0);
        REQUIRE(hostVals.view()[2] == 0.0);
        REQUIRE(hostVals.view()[3] == 0.0);
    }

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

    SECTION("BlockView mdspan-like read access " + execName)
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
                rv[0] = bmView(0, 0)(0, 0); // 4.0
                rv[1] = bmView(0, 1)(0, 0); // 1.0
                rv[2] = bmView(1, 0)(0, 0); // 1.0
                rv[3] = bmView(1, 1)(0, 0); // 3.0
            },
            "BlockView_mdspanRead"
        );

        auto hostResult = result.copyToHost();
        REQUIRE(hostResult.view()[0] == 4.0);
        REQUIRE(hostResult.view()[1] == 1.0);
        REQUIRE(hostResult.view()[2] == 1.0);
        REQUIRE(hostResult.view()[3] == 3.0);
    }

    SECTION("BlockView mdspan-like write access " + execName)
    {
        BlockMatrix bm(exec, 2, sp1);

        auto bmView = bm.view();

        parallelFor(
            exec,
            {0, 1},
            NEON_LAMBDA(const localIdx celli) {
                bmView(0, 0)(celli, celli) = 4.0;
                bmView(0, 1)(celli, celli) = 1.0;
                bmView(1, 0)(celli, celli) = 1.0;
                bmView(1, 1)(celli, celli) = 3.0;
            },
            "BlockView_mdspanWrite"
        );

        auto hostVals = bm.values().copyToHost();
        REQUIRE(hostVals.view()[0] == 4.0);
        REQUIRE(hostVals.view()[1] == 1.0);
        REQUIRE(hostVals.view()[2] == 1.0);
        REQUIRE(hostVals.view()[3] == 3.0);
    }

    SECTION("BlockView direct offset access " + execName)
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
                rv[0] = bmView(0, 0)[0]; // 4.0
                rv[1] = bmView(0, 1)[0]; // 1.0
                rv[2] = bmView(1, 0)[0]; // 1.0
                rv[3] = bmView(1, 1)[0]; // 3.0
            },
            "BlockView_offsetAccess"
        );

        auto hostResult = result.copyToHost();
        REQUIRE(hostResult.view()[0] == 4.0);
        REQUIRE(hostResult.view()[1] == 1.0);
        REQUIRE(hostResult.view()[2] == 1.0);
        REQUIRE(hostResult.view()[3] == 3.0);
    }

    SECTION("BlockRowView row extraction " + execName)
    {
        BlockMatrix bm(exec, 2, sp1);

        auto bmView = bm.view();
        Vector<scalar> result(exec, 4, 0.0);
        auto rv = result.view();

        parallelFor(
            exec,
            {0, 1},
            NEON_LAMBDA(const localIdx celli) {
                auto row0 = bmView.row(0);
                row0(0)(celli, celli) = 4.0;
                row0(1)(celli, celli) = 1.0;

                auto row1 = bmView.row(1);
                row1(0)(celli, celli) = 1.0;
                row1(1)(celli, celli) = 3.0;

                rv[0] = row0(0)(0, 0);
                rv[1] = row0(1)(0, 0);
                rv[2] = row1(0)(0, 0);
                rv[3] = row1(1)(0, 0);
            },
            "BlockRowView_assembly"
        );

        auto hostResult = result.copyToHost();
        REQUIRE(hostResult.view()[0] == 4.0);
        REQUIRE(hostResult.view()[1] == 1.0);
        REQUIRE(hostResult.view()[2] == 1.0);
        REQUIRE(hostResult.view()[3] == 3.0);
    }

    SECTION("BlockMatrixView global entry access " + execName)
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
                rv[0] = bmView.entry(0, 0); // 4.0
                rv[1] = bmView.entry(0, 1); // 1.0
                rv[2] = bmView.entry(1, 0); // 1.0
                rv[3] = bmView.entry(1, 1); // 3.0
            },
            "BlockMatrixView_globalEntry"
        );

        auto hostResult = result.copyToHost();
        REQUIRE(hostResult.view()[0] == 4.0);
        REQUIRE(hostResult.view()[1] == 1.0);
        REQUIRE(hostResult.view()[2] == 1.0);
        REQUIRE(hostResult.view()[3] == 3.0);
    }
}

TEST_CASE("BlockMatrix - monolithic")
{
    using namespace NeoN;
    using namespace NeoN::la;

    auto [execName, exec] = GENERATE(allAvailableExecutor());

    SECTION("Monolithic flattening 2x2 of 1-cell blocks " + execName)
    {
        auto sp = std::make_shared<SparsityPattern<localIdx>>(
            Vector<localIdx>(exec, std::vector<localIdx> {0}),
            Vector<localIdx>(exec, std::vector<localIdx> {0, 1})
        );
        Vector<scalar> vals(exec, std::vector<scalar> {4.0, 1.0, 1.0, 3.0});
        BlockMatrix bm(exec, 2, sp, vals);

        auto mono = bm.monolithic();

        REQUIRE(mono.nRows() == 2);
        REQUIRE(mono.nNonZeros() == 4);

        auto hostMono = mono.copyToHost();
        auto hostVals = hostMono.values().view();
        auto hostCols = hostMono.colIdxs().view();
        auto hostRows = hostMono.rowOffs().view();

        REQUIRE(hostRows[0] == 0);
        REQUIRE(hostRows[1] == 2);
        REQUIRE(hostRows[2] == 4);

        REQUIRE(hostCols[0] == 0);
        REQUIRE(hostCols[1] == 1);
        REQUIRE(hostCols[2] == 0);
        REQUIRE(hostCols[3] == 1);

        REQUIRE(hostVals[0] == 4.0);
        REQUIRE(hostVals[1] == 1.0);
        REQUIRE(hostVals[2] == 1.0);
        REQUIRE(hostVals[3] == 3.0);
    }

    SECTION("Monolithic with zero off-diagonal " + execName)
    {
        auto sp = std::make_shared<SparsityPattern<localIdx>>(
            Vector<localIdx>(exec, std::vector<localIdx> {0}),
            Vector<localIdx>(exec, std::vector<localIdx> {0, 1})
        );
        Vector<scalar> vals(exec, std::vector<scalar> {4.0, 0.0, 0.0, 3.0});
        BlockMatrix bm(exec, 2, sp, vals);

        auto mono = bm.monolithic();

        REQUIRE(mono.nRows() == 2);
        REQUIRE(mono.nNonZeros() == 4);

        auto hostMono = mono.copyToHost();
        auto hostVals = hostMono.values().view();

        REQUIRE(hostVals[0] == 4.0);
        REQUIRE(hostVals[1] == 0.0);
        REQUIRE(hostVals[2] == 0.0);
        REQUIRE(hostVals[3] == 3.0);
    }

    SECTION("Monolithic flattening 2x2 of 3-cell blocks " + execName)
    {
        auto sp3 = std::make_shared<SparsityPattern<localIdx>>(
            Vector<localIdx>(exec, std::vector<localIdx> {0, 1, 0, 1, 2, 1, 2}),
            Vector<localIdx>(exec, std::vector<localIdx> {0, 2, 5, 7})
        );
        std::vector<scalar> valsVec = {
            2,   -1, -1, 2,   -1, -1, 2,   // block(0,0)
            0.5, 0,  0,  0.5, 0,  0,  0.5, // block(0,1)
            0.5, 0,  0,  0.5, 0,  0,  0.5, // block(1,0)
            3,   -1, -1, 3,   -1, -1, 3    // block(1,1)
        };
        Vector<scalar> vals(exec, valsVec);
        BlockMatrix bm(exec, 2, sp3, vals);

        auto mono = bm.monolithic();

        REQUIRE(mono.nRows() == 6);
        REQUIRE(mono.nNonZeros() == 28);

        auto hostMono = mono.copyToHost();
        auto hv = hostMono.values().view();
        auto hc = hostMono.colIdxs().view();
        auto hr = hostMono.rowOffs().view();

        // Row 0 (I=0, cell 0): J=0 cols {0,1} vals {2,-1}; J=1 cols {3,4} vals {0.5,0}
        REQUIRE(hr[0] == 0);
        REQUIRE(hr[1] == 4);
        REQUIRE(hc[0] == 0);
        REQUIRE(hc[1] == 1);
        REQUIRE(hc[2] == 3);
        REQUIRE(hc[3] == 4);
        REQUIRE(hv[0] == 2.0);
        REQUIRE(hv[1] == -1.0);
        REQUIRE(hv[2] == 0.5);
        REQUIRE(hv[3] == 0.0);

        // Row 1 (I=0, cell 1): J=0 cols {0,1,2} vals {-1,2,-1}; J=1 cols {3,4,5} vals {0,0.5,0}
        REQUIRE(hr[2] == 10);
        REQUIRE(hc[4] == 0);
        REQUIRE(hc[5] == 1);
        REQUIRE(hc[6] == 2);
        REQUIRE(hc[7] == 3);
        REQUIRE(hc[8] == 4);
        REQUIRE(hc[9] == 5);
        REQUIRE(hv[4] == -1.0);
        REQUIRE(hv[5] == 2.0);
        REQUIRE(hv[6] == -1.0);
        REQUIRE(hv[7] == 0.0);
        REQUIRE(hv[8] == 0.5);
        REQUIRE(hv[9] == 0.0);
    }
}
