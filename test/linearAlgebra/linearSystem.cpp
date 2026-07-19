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
    CSRMatrix<scalar, localIdx> csrMatrix(values, colIdx, rowOffs, {3, 3});

    Vector<scalar> bValues(exec, {0.0, 0.0, 0.0});
    Vector<localIdx> bColIdx(exec, {0, 1, 2});
    Vector<localIdx> bRowOffs(exec, {0, 1, 2});
    COOMatrix<scalar, localIdx> bCooMatrix(bValues, bColIdx, bRowOffs, {3, 1});

    SECTION("construct " + execName)
    {
        Vector<scalar> rhs(exec, 3, 0.0);
        Vector<scalar> bRhs(exec, 3, 0.0);
        LinearSystem<scalar> linearSystem(csrMatrix, rhs, bCooMatrix, bCooMatrix, bRhs);

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

        auto linearSystem =
            NeoN::la::createEmptyLinearSystem<scalar, scalar, CSRMatrix, CSRMatrix>(mesh);

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

        auto linearSystem =
            NeoN::la::createEmptyLinearSystem<scalar, scalar, COOMatrix, COOMatrix>(mesh);

        REQUIRE(linearSystem.matrix().values().size() == nnz);
        REQUIRE(linearSystem.matrix().colIdxs().size() == nnz);
        REQUIRE(linearSystem.matrix().nRows() == nCells);
        REQUIRE(linearSystem.rhs().size() == nCells);
    }

    SECTION("Construct with MeshCellIterator " + execName)
    {
        auto nCells = 10;
        auto nFaces = 9;
        auto nnz = nCells + 2 * nFaces;
        auto mesh = create1DUniformMesh(exec, nCells);

        auto cellIterator = std::make_shared<NeoN::la::CellBasedIterator>();
        auto linearSystem = NeoN::la::createEmptyLinearSystem<scalar>(mesh, cellIterator);

        REQUIRE(linearSystem.matrix().values().size() == nnz);
        REQUIRE(linearSystem.matrix().colIdxs().size() == nnz);
        REQUIRE(linearSystem.matrix().rowOffs().size() == nCells + 1);
        REQUIRE(linearSystem.matrix().nRows() == nCells);
        REQUIRE(linearSystem.rhs().size() == nCells);
        REQUIRE(linearSystem.getMeshIterator()->name() == "CellBased");
    }

    // Regression: each boundary face must land on its own owner cell, not another one --
    // CooSparsityPattern::view() briefly exposed row-range data here instead.
    SECTION("removeBoundaryContributions applies each correction to its own owner cell " + execName)
    {
        auto nCells = 4;
        auto mesh = create1DUniformMesh(exec, nCells);
        auto ls = NeoN::la::createEmptyLinearSystem<scalar>(mesh);

        REQUIRE(ls.boundaryMatrix().values().size() == 2);
        Vector<scalar> boundaryValues(exec, {10.0, 20.0});
        ls.boundaryMatrix().values() = boundaryValues;

        auto bRowIdxsHost = ls.boundaryMatrix().sparsity()->rowIdxs().copyToHost();
        auto owner0 = bRowIdxsHost.view()[0];
        auto owner1 = bRowIdxsHost.view()[1];
        REQUIRE(owner0 != owner1); // sanity: the two boundary faces belong to different cells

        auto lsNoBnd = NeoN::la::removeBoundaryContributions(ls);
        auto diagHost = lsNoBnd.matrix().diag().copyToHost();
        auto diagView = diagHost.view();

        for (localIdx i = 0; i < nCells; ++i)
        {
            scalar expected = 0.0;
            if (i == owner0) expected += 10.0;
            if (i == owner1) expected += 20.0;
            REQUIRE(diagView[i] == expected);
        }
    }

    // Assembles a 1D Laplacian stencil (diag = face count per cell, off-diag = -1 per face)
    // through createEmptyLinearSystem + faceToMatrixView(), checking every logical (row,col).
    SECTION("createEmptyLinearSystem<ELLMatrix> assembles via faceToMatrixView " + execName)
    {
        using ELLMatrix = NeoN::la::ELLMatrix<scalar, localIdx>;

        auto nCells = 4;
        auto nFaces = 3;
        auto mesh = create1DUniformMesh(exec, nCells);
        auto ls = NeoN::la::createEmptyLinearSystem<scalar, scalar, ELLMatrix>(mesh);

        REQUIRE(ls.matrix().values().size() == ls.matrix().sparsity()->storageSize());
        REQUIRE(ls.matrix().nNonZeros() == nCells + 2 * nFaces);

        auto ma = ls.matrix().faceToMatrixView();
        auto matrixV = ls.matrix().values().view();
        parallelFor(
            exec,
            {0, nFaces},
            NEON_LAMBDA(const localIdx facei) {
                const localIdx own = facei;
                const localIdx nei = facei + 1;
                matrixV[ma.upperIdx(own, facei)] = -1.0;
                matrixV[ma.lowerIdx(nei, facei)] = -1.0;
                Kokkos::atomic_add(&matrixV[ma.diagIdx(own)], 1.0);
                Kokkos::atomic_add(&matrixV[ma.diagIdx(nei)], 1.0);
            },
            "assembleLaplacianStyleELL"
        );

        REQUIRE_THAT(ls.matrix().diag(), Equals(I({1.0, 2.0, 2.0, 1.0})));

        auto matView = ls.matrix().view();
        Vector<scalar> checkOffDiag(exec, 6);
        auto checkOffDiagV = checkOffDiag.view();
        parallelFor(
            exec,
            {0, 1},
            NEON_LAMBDA(const localIdx) {
                checkOffDiagV[0] = matView.entry(0, 1);
                checkOffDiagV[1] = matView.entry(1, 0);
                checkOffDiagV[2] = matView.entry(1, 2);
                checkOffDiagV[3] = matView.entry(2, 1);
                checkOffDiagV[4] = matView.entry(2, 3);
                checkOffDiagV[5] = matView.entry(3, 2);
            }
        );
        REQUIRE_THAT(checkOffDiag, Equals(I({-1.0, -1.0, -1.0, -1.0, -1.0, -1.0})));

        // padded colIdx layout {0,0,1,2, 1,1,2,3, INV,2,3,INV}: slot 2 of rows 0 and 3 (flat
        // offsets 8 and 11) is padding -- assembly never touched it, so it stays zero-filled.
        auto valuesHost = ls.matrix().values().copyToHost();
        auto valuesHostV = valuesHost.view();
        REQUIRE(valuesHostV[8] == 0.0);
        REQUIRE(valuesHostV[11] == 0.0);
    }

    SECTION("ELL system copyToExecutor preserves faceToMatrixView " + execName)
    {
        using ELLMatrix = NeoN::la::ELLMatrix<scalar, localIdx>;

        auto nCells = 4;
        auto mesh = create1DUniformMesh(exec, nCells);
        auto ls = NeoN::la::createEmptyLinearSystem<scalar, scalar, ELLMatrix>(mesh);

        auto hostMatrix = ls.matrix().copyToExecutor(NeoN::SerialExecutor());
        REQUIRE(hostMatrix.faceToMatrixAddress() != nullptr);

        auto ma = hostMatrix.faceToMatrixView();
        auto matrixV = hostMatrix.values().view();
        parallelFor(
            NeoN::SerialExecutor(),
            {0, 1},
            NEON_LAMBDA(const localIdx) { matrixV[ma.diagIdx(0)] = 42.0; }
        );
        REQUIRE(hostMatrix.values().view()[ma.diagIdx(0)] == 42.0);
    }

    // ELL counterpart to "removeBoundaryContributions applies each correction to its own
    // owner cell" above -- same check, ELL system matrix instead of CSR.
    SECTION("removeBoundaryContributions works with an ELL system matrix " + execName)
    {
        using ELLMatrix = NeoN::la::ELLMatrix<scalar, localIdx>;

        auto nCells = 4;
        auto mesh = create1DUniformMesh(exec, nCells);
        auto ls = NeoN::la::createEmptyLinearSystem<scalar, scalar, ELLMatrix>(mesh);

        REQUIRE(ls.boundaryMatrix().values().size() == 2);
        Vector<scalar> boundaryValues(exec, {10.0, 20.0});
        ls.boundaryMatrix().values() = boundaryValues;

        auto bRowIdxsHost = ls.boundaryMatrix().sparsity()->rowIdxs().copyToHost();
        auto owner0 = bRowIdxsHost.view()[0];
        auto owner1 = bRowIdxsHost.view()[1];
        REQUIRE(owner0 != owner1);

        auto lsNoBnd = NeoN::la::removeBoundaryContributions(ls);
        auto diagHost = lsNoBnd.matrix().diag().copyToHost();
        auto diagView = diagHost.view();

        for (localIdx i = 0; i < nCells; ++i)
        {
            scalar expected = 0.0;
            if (i == owner0) expected += 10.0;
            if (i == owner1) expected += 20.0;
            REQUIRE(diagView[i] == expected);
        }
    }


    SECTION("view read/write " + execName)
    {
        Vector<scalar> rhs(exec, {10.0, 20.0, 30.0});
        Vector<scalar> bRhs(exec, {0.0, 0.0, 0.0});
        LinearSystem<scalar> linearSystem(csrMatrix, rhs, bCooMatrix, bCooMatrix, bRhs);

        auto lsView = linearSystem.view();
        auto hostLS = linearSystem.copyToHost();
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
        auto hostLS2 = linearSystem.copyToHost();
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
