// SPDX-FileCopyrightText: 2024 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main
#include <string>

#include "catch2_common.hpp"

#include "NeoN/NeoN.hpp"

using NeoN::scalar;
using NeoN::localIdx;
using NeoN::Vector;
using NeoN::la::LinearSystem;
using NeoN::la::CSRMatrix;
using NeoN::la::COOMatrix;

TEMPLATE_TEST_CASE("LinearSystem", "[template]", NeoN::scalar)
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    Vector<scalar> values(exec, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0});
    Vector<localIdx> colIdx(exec, {0, 1, 2, 0, 1, 2, 0, 1, 2});
    Vector<localIdx> rowOffs(exec, {0, 3, 6, 9});
    CSRMatrix<scalar, localIdx> csrMatrix(values, colIdx, rowOffs);

    Vector<scalar> bValues(exec, {0.0, 0.0, 0.0});
    Vector<localIdx> bColIdx(exec, {0, 1, 2});
    Vector<localIdx> bRowOffs(exec, {0, 1, 2});
    COOMatrix<scalar, localIdx> bCooMatrix(bValues, bColIdx, bRowOffs);

    SECTION("construct " + execName)
    {
        Vector<scalar> rhs(exec, 3, 0.0);
        Vector<scalar> bRhs(exec, 3, 0.0);
        LinearSystem<scalar, NeoN::la::CSRMatrix<scalar, NeoN::localIdx>> linearSystem(
            csrMatrix, rhs, bCooMatrix, bRhs
        );

        REQUIRE(linearSystem.matrix().values().size() == 9);
        REQUIRE(linearSystem.matrix().colIdxs().size() == 9);
        REQUIRE(linearSystem.matrix().rowOffs().size() == 4);
        REQUIRE(linearSystem.matrix().nRows() == 3);
        REQUIRE(linearSystem.rhs().size() == 3);
    }

    SECTION("construct zero initialized from sparsity " + execName)
    {
        auto nCells = 10;
        auto nFaces = 9;
        auto nnz = nCells + 2 * nFaces;
        auto mesh = create1DUniformMesh(exec, nCells);

        auto linearSystem = NeoN::la::createEmptyLinearSystem<scalar>(mesh);

        REQUIRE(linearSystem.matrix().values().size() == nnz);
        REQUIRE(linearSystem.matrix().colIdxs().size() == nnz);
        REQUIRE(linearSystem.matrix().rowOffs().size() == nCells + 1);
        REQUIRE(linearSystem.matrix().nRows() == nCells);
        REQUIRE(linearSystem.rhs().size() == nCells);
    }

    SECTION("construct zero initialized from sparsity with CSR matrix " + execName)
    {
        auto nCells = 10;
        auto nFaces = 9;
        auto nnz = nCells + 2 * nFaces;
        auto mesh = create1DUniformMesh(exec, nCells);

        using CSRMatrix = NeoN::la::CSRMatrix<scalar, localIdx>;

        auto linearSystem = NeoN::la::createEmptyLinearSystem<scalar, CSRMatrix, CSRMatrix>(mesh);

        REQUIRE(linearSystem.matrix().values().size() == nnz);
        REQUIRE(linearSystem.matrix().colIdxs().size() == nnz);
        REQUIRE(linearSystem.matrix().rowOffs().size() == nCells + 1);
        REQUIRE(linearSystem.matrix().nRows() == nCells);
        REQUIRE(linearSystem.rhs().size() == nCells);
    }

    SECTION("construct zero initialized from sparsity with COO matrix " + execName)
    {
        auto nCells = 10;
        auto nFaces = 9;
        auto nnz = nCells + 2 * nFaces;
        auto mesh = create1DUniformMesh(exec, nCells);

        using COOMatrix = NeoN::la::COOMatrix<scalar, localIdx>;

        auto linearSystem = NeoN::la::createEmptyLinearSystem<scalar, COOMatrix, COOMatrix>(mesh);

        REQUIRE(linearSystem.matrix().values().size() == nnz);
        REQUIRE(linearSystem.matrix().colIdxs().size() == nnz);
        REQUIRE(linearSystem.matrix().nRows() == nCells);
        REQUIRE(linearSystem.rhs().size() == nCells);
    }

    SECTION("view read/write " + execName)
    {
        Vector<scalar> rhs(exec, {10.0, 20.0, 30.0});
        Vector<scalar> bRhs(exec, {0.0, 0.0, 0.0});
        LinearSystem<scalar, CSRMatrix<scalar, localIdx>> ls(csrMatrix, rhs, bCooMatrix, bRhs);

        auto lsView = ls.view();
        auto hostLS = ls.copyToHost();
        auto hostLSView = hostLS.view();

        // some simple sanity checks
        REQUIRE(hostLSView.matrix.values.size() == 9);
        REQUIRE(hostLSView.matrix.sparsity.colIdxs.size() == 9);
        REQUIRE(hostLSView.matrix.sparsity.rowOffs.size() == 4);
        REQUIRE(hostLSView.rhs.size() == 3);

        // check system values
        for (NeoN::localIdx i = 0; i < hostLSView.matrix.values.size(); ++i)
        {
            REQUIRE(hostLSView.matrix.values[i] == static_cast<scalar>(i + 1));
            REQUIRE(hostLSView.matrix.sparsity.colIdxs[i] == (i % 3));
        }
        for (NeoN::localIdx i = 0; i < hostLSView.matrix.sparsity.rowOffs.size(); ++i)
        {
            REQUIRE(hostLSView.matrix.sparsity.rowOffs[i] == i * 3);
        }
        for (NeoN::localIdx i = 0; i < hostLSView.rhs.size(); ++i)
        {
            REQUIRE(hostLSView.rhs[i] == static_cast<scalar>((i + 1) * 10));
        }

        // Modify values.
        parallelFor(
            exec,
            {0, lsView.matrix.values.size()},
            NEON_LAMBDA(const localIdx i) { lsView.matrix.values[i] = -lsView.matrix.values[i]; }
        );

        // Modify values.
        parallelFor(
            exec,
            {0, lsView.rhs.size()},
            NEON_LAMBDA(const localIdx i) { lsView.rhs[i] = -lsView.rhs[i]; }
        );

        // Check modification.
        auto hostLS2 = ls.copyToHost();
        auto hostLS2View = hostLS2.view();
        for (NeoN::localIdx i = 0; i < hostLS2View.matrix.values.size(); ++i)
        {
            REQUIRE(hostLS2View.matrix.values[i] == -static_cast<scalar>(i + 1));
        }
        for (NeoN::localIdx i = 0; i < hostLSView.rhs.size(); ++i)
        {
            REQUIRE(hostLS2View.rhs[i] == -static_cast<scalar>((i + 1) * 10));
        }
    }
}
