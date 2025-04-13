// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoN authors

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
using NeoN::la::spmv;

TEST_CASE("LinearSystem")
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    Vector<scalar> values(exec, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0});
    Vector<localIdx> colIdx(exec, {0, 1, 2, 0, 1, 2, 0, 1, 2});
    Vector<localIdx> rowPtrs(exec, {0, 3, 6, 9});
    CSRMatrix<scalar, localIdx> csrMatrix(values, colIdx, rowPtrs);

    SECTION("construct " + execName)
    {

        Vector<scalar> rhs(exec, 3, 0.0);
        LinearSystem<scalar, localIdx> linearSystem(csrMatrix, rhs);

        REQUIRE(linearSystem.matrix().values().size() == 9);
        REQUIRE(linearSystem.matrix().colIdxs().size() == 9);
        REQUIRE(linearSystem.matrix().rowPtrs().size() == 4);
        REQUIRE(linearSystem.matrix().nRows() == 3);
        REQUIRE(linearSystem.rhs().size() == 3);
    }

    SECTION("construct zero initialized from sparsity " + execName)
    {
        auto nCells = 10;
        auto nFaces = 9;
        auto nnz = nCells + 2 * nFaces;
        auto mesh = create1DUniformMesh(exec, nCells);

        // TODO improve structure here
        auto sp = NeoN::finiteVolume::cellCentred::SparsityPattern {mesh};
        auto linearSystem = NeoN::la::createEmptyLinearSystem<
            scalar,
            localIdx,
            NeoN::finiteVolume::cellCentred::SparsityPattern>(sp);

        REQUIRE(linearSystem.matrix().values().size() == nnz);
        REQUIRE(linearSystem.matrix().colIdxs().size() == nnz);
        REQUIRE(linearSystem.matrix().rowPtrs().size() == nCells + 1);
        REQUIRE(linearSystem.matrix().nRows() == nCells);
        REQUIRE(linearSystem.rhs().size() == nCells);
    }

    SECTION("view read/write " + execName)
    {
        Vector<scalar> rhs(exec, {10.0, 20.0, 30.0});
        LinearSystem<scalar, localIdx> ls(csrMatrix, rhs);

        auto lsView = ls.view();
        auto hostLS = ls.copyToHost();
        auto hostLSView = hostLS.view();

        // some simple sanity checks
        REQUIRE(hostLSView.matrix.values.size() == 9);
        REQUIRE(hostLSView.matrix.colIdxs.size() == 9);
        REQUIRE(hostLSView.matrix.rowOffs.size() == 4);
        REQUIRE(hostLSView.rhs.size() == 3);

        // check system values
        for (size_t i = 0; i < hostLSView.matrix.values.size(); ++i)
        {
            REQUIRE(hostLSView.matrix.values[i] == static_cast<scalar>(i + 1));
            REQUIRE(hostLSView.matrix.colIdxs[i] == (i % 3));
        }
        for (size_t i = 0; i < hostLSView.matrix.rowOffs.size(); ++i)
        {
            REQUIRE(hostLSView.matrix.rowOffs[i] == static_cast<localIdx>(i * 3));
        }
        for (size_t i = 0; i < hostLSView.rhs.size(); ++i)
        {
            REQUIRE(hostLSView.rhs[i] == static_cast<scalar>((i + 1) * 10));
        }

        // Modify values.
        parallelFor(
            exec,
            {0, lsView.matrix.values.size()},
            KOKKOS_LAMBDA(const size_t i) { lsView.matrix.values[i] = -lsView.matrix.values[i]; }
        );

        // Modify values.
        parallelFor(
            exec,
            {0, lsView.rhs.size()},
            KOKKOS_LAMBDA(const size_t i) { lsView.rhs[i] = -lsView.rhs[i]; }
        );

        // Check modification.
        hostLS = ls.copyToHost();
        hostLSView = hostLS.view();
        for (size_t i = 0; i < hostLSView.matrix.values.size(); ++i)
        {
            REQUIRE(hostLSView.matrix.values[i] == -static_cast<scalar>(i + 1));
        }
        for (size_t i = 0; i < hostLSView.rhs.size(); ++i)
        {
            REQUIRE(hostLSView.rhs[i] == -static_cast<scalar>((i + 1) * 10));
        }
    }


    SECTION("SpmV" + execName)
    {
        Vector<scalar> rhs(exec, 3, 0.0);
        LinearSystem<scalar, localIdx> linearSystem(csrMatrix, rhs);
        Vector<scalar> x(exec, {1.0, 2.0, 3.0});

        Vector<scalar> y = spmv(linearSystem, x);
        auto yHost = y.copyToHost();
        auto yHostView = yHost.view();

        REQUIRE(yHostView[0] == 1.0 * 1.0 + 2.0 * 2.0 + 3.0 * 3.0);
        REQUIRE(yHostView[1] == 4.0 * 1.0 + 5.0 * 2.0 + 6.0 * 3.0);
        REQUIRE(yHostView[2] == 7.0 * 1.0 + 8.0 * 2.0 + 9.0 * 3.0);

        // test with non-zero rhs
        Vector<scalar> rhs2(exec, {1.0, 2.0, 3.0});
        LinearSystem<scalar, localIdx> linearSystem2(csrMatrix, rhs2);
        y = spmv(linearSystem2, x);
        yHost = y.copyToHost();
        yHostView = yHost.view();

        REQUIRE(yHostView[0] == 14.0 - 1.0);
        REQUIRE(yHostView[1] == 32.0 - 2.0);
        REQUIRE(yHostView[2] == 50.0 - 3.0);
    }
}
