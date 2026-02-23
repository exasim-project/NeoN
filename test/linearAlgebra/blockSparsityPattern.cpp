// SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#define CATCH_CONFIG_RUNNER
#include "catch2_common.hpp"

#include "NeoN/NeoN.hpp"
#include "NeoN/linearAlgebra/blockSparsityPattern.hpp"

TEST_CASE("BlockSparsityPattern")
{
    using namespace NeoN;
    using namespace NeoN::la;

    auto [execName, exec] = GENERATE(allAvailableExecutor());

    // 1-cell diagonal sparsity (nnz=1)
    SparsityPattern<localIdx> sp1(
        Vector<localIdx>(exec, std::vector<localIdx> {0}),
        Vector<localIdx>(exec, std::vector<localIdx> {0, 1})
    );

    SECTION("metadata from 1-cell 2-block " + execName)
    {
        BlockSparsityPattern bsp(2, sp1);

        REQUIRE(bsp.nBlocks() == 2);
        REQUIRE(bsp.nCells() == 1);
        REQUIRE(bsp.baseNnz() == 1);
        REQUIRE(bsp.rows() == 2); // nBlocks * nCells
        REQUIRE(bsp.nnz() == 4);  // nBlocks^2 * baseNnz
    }

    SECTION("expanded layout from 1-cell 2-block " + execName)
    {
        BlockSparsityPattern bsp(2, sp1);
        auto host = bsp.copyToHost();

        auto cols = host.colIdxs().view();
        auto rows = host.rowOffs().view();

        // 2 monolithic rows, 4 non-zeros
        // Row 0 (I=0, cell=0): J=0->col0, J=1->col1
        // Row 1 (I=1, cell=0): J=0->col0, J=1->col1
        REQUIRE(rows[0] == 0);
        REQUIRE(rows[1] == 2);
        REQUIRE(rows[2] == 4);

        REQUIRE(cols[0] == 0);
        REQUIRE(cols[1] == 1);
        REQUIRE(cols[2] == 0);
        REQUIRE(cols[3] == 1);
    }

    SECTION("3-cell tridiagonal 2-block expansion " + execName)
    {
        // 3-cell tridiagonal: colIdxs={0,1, 0,1,2, 1,2}, rowOffs={0,2,5,7}
        SparsityPattern<localIdx> sp3(
            Vector<localIdx>(exec, std::vector<localIdx> {0, 1, 0, 1, 2, 1, 2}),
            Vector<localIdx>(exec, std::vector<localIdx> {0, 2, 5, 7})
        );

        BlockSparsityPattern bsp(2, sp3);

        REQUIRE(bsp.nBlocks() == 2);
        REQUIRE(bsp.nCells() == 3);
        REQUIRE(bsp.baseNnz() == 7);
        REQUIRE(bsp.rows() == 6); // 2 * 3
        REQUIRE(bsp.nnz() == 28); // 4 * 7

        auto host = bsp.copyToHost();
        auto cols = host.colIdxs().view();
        auto rows = host.rowOffs().view();

        // Row 0 (I=0, cell=0): J=0->{col0,col1}, J=1->{col3,col4} => 4 entries
        REQUIRE(rows[0] == 0);
        REQUIRE(rows[1] == 4);
        REQUIRE(cols[0] == 0);
        REQUIRE(cols[1] == 1);
        REQUIRE(cols[2] == 3);
        REQUIRE(cols[3] == 4);

        // Row 1 (I=0, cell=1): J=0->{col0,col1,col2}, J=1->{col3,col4,col5} => 6 entries
        REQUIRE(rows[2] == 10);
        REQUIRE(cols[4] == 0);
        REQUIRE(cols[5] == 1);
        REQUIRE(cols[6] == 2);
        REQUIRE(cols[7] == 3);
        REQUIRE(cols[8] == 4);
        REQUIRE(cols[9] == 5);
    }

    SECTION("copy constructor preserves metadata " + execName)
    {
        BlockSparsityPattern bsp(2, sp1);
        BlockSparsityPattern copy(bsp);

        REQUIRE(copy.nBlocks() == 2);
        REQUIRE(copy.nCells() == 1);
        REQUIRE(copy.baseNnz() == 1);
        REQUIRE(copy.rows() == 2);
        REQUIRE(copy.nnz() == 4);
    }

    SECTION("copyToHost preserves metadata " + execName)
    {
        BlockSparsityPattern bsp(2, sp1);
        auto host = bsp.copyToHost();

        REQUIRE(host.nBlocks() == 2);
        REQUIRE(host.nCells() == 1);
        REQUIRE(host.baseNnz() == 1);
        REQUIRE(host.rows() == 2);
        REQUIRE(host.nnz() == 4);
    }

    SECTION("copyToExecutor preserves metadata " + execName)
    {
        BlockSparsityPattern bsp(2, sp1);
        auto copied = bsp.copyToExecutor(exec);

        REQUIRE(copied.nBlocks() == 2);
        REQUIRE(copied.nCells() == 1);
        REQUIRE(copied.baseNnz() == 1);
        REQUIRE(copied.rows() == 2);
        REQUIRE(copied.nnz() == 4);
    }

    SECTION("toCSR round-trip with BlockCSRMatrix " + execName)
    {
        BlockSparsityPattern bsp(2, sp1);
        auto spPtr = std::make_shared<const BlockSparsityPattern>(bsp);

        // 4 values for 2x2 monolithic matrix
        Vector<scalar> vals(exec, std::vector<scalar> {4.0, 1.0, 1.0, 3.0});
        BlockCSRMatrix bm(vals, spPtr);

        auto csr = toCSR(bm);
        REQUIRE(csr.nRows() == 2);
        REQUIRE(csr.nNonZeros() == 4);

        auto hostCSR = csr.copyToHost();
        REQUIRE(hostCSR.values().view()[0] == 4.0);
        REQUIRE(hostCSR.values().view()[1] == 1.0);
        REQUIRE(hostCSR.values().view()[2] == 1.0);
        REQUIRE(hostCSR.values().view()[3] == 3.0);
    }
}
