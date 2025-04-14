// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoN authors

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main

#include "catch2_common.hpp"

#include "NeoN/NeoN.hpp"

#include <Kokkos_Core.hpp>

TEST_CASE("CSRMatrix")
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    // sparse matrix
    NeoN::Vector<NeoN::scalar> valuesSparse(exec, {1.0, 5.0, 6.0, 8.0});
    NeoN::Vector<NeoN::localIdx> colIdxSparse(exec, {0, 1, 2, 1});
    NeoN::Vector<NeoN::localIdx> rowPtrsSparse(exec, {0, 1, 3, 4});
    NeoN::la::CSRMatrix<NeoN::scalar, NeoN::localIdx> sparseMatrix(
        valuesSparse, colIdxSparse, rowPtrsSparse
    );
    const NeoN::la::CSRMatrix<NeoN::scalar, NeoN::localIdx> sparseMatrixConst(
        valuesSparse, colIdxSparse, rowPtrsSparse
    );

    // dense matrix
    NeoN::Vector<NeoN::scalar> valuesDense(exec, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0});
    NeoN::Vector<NeoN::localIdx> colIdxDense(exec, {0, 1, 2, 0, 1, 2, 0, 1, 2});
    NeoN::Vector<NeoN::localIdx> rowPtrsDense(exec, {0, 3, 6, 9});
    NeoN::la::CSRMatrix<NeoN::scalar, NeoN::localIdx> denseMatrix(
        valuesDense, colIdxDense, rowPtrsDense
    );
    const NeoN::la::CSRMatrix<NeoN::scalar, NeoN::localIdx> denseMatrixConst(
        valuesDense, colIdxDense, rowPtrsDense
    );

    // NOTE: The purpose of this test is to detect changes in the order
    // of the structured bindings
    SECTION("View Order " + execName)
    {
        auto denseMatrixHost = denseMatrix.copyToHost();
        auto [values, colIdxs, rowOffs] = denseMatrixHost.view();
        auto valuesDenseHost = valuesDense.copyToHost();
        auto valuesDenseHostView = valuesDenseHost.view();
        auto colIdxDenseHost = colIdxDense.copyToHost();
        auto colIdxDenseHostView = colIdxDenseHost.view();
        auto rowPtrsDenseHost = rowPtrsDense.copyToHost();
        auto rowPtrsDenseHostView = rowPtrsDenseHost.view();

        for (int i = 0; i < valuesDenseHostView.size(); ++i)
        {
            REQUIRE(valuesDenseHostView[i] == values[i]);
            REQUIRE(colIdxDenseHostView[i] == colIdxs[i]);
        }
        for (int i = 0; i < rowPtrsDenseHostView.size(); ++i)
        {
            REQUIRE(rowPtrsDenseHostView[i] == rowOffs[i]);
        }
    }

    SECTION("Read entry on " + execName)
    {
        // Sparse
        NeoN::Vector<NeoN::scalar> checkSparse(exec, 4);
        auto checkSparseView = checkSparse.view();
        auto csrView = sparseMatrixConst.view();
        parallelFor(
            exec,
            {0, 1},
            KOKKOS_LAMBDA(const NeoN::localIdx) {
                checkSparseView[0] = csrView.entry(0, 0);
                checkSparseView[1] = csrView.entry(1, 1);
                checkSparseView[2] = csrView.entry(1, 2);
                checkSparseView[3] = csrView.entry(2, 1);
            }
        );

        auto checkHost = checkSparse.copyToHost();
        REQUIRE(checkHost.view()[0] == 1.0);
        REQUIRE(checkHost.view()[1] == 5.0);
        REQUIRE(checkHost.view()[2] == 6.0);
        REQUIRE(checkHost.view()[3] == 8.0);

        // Dense
        NeoN::Vector<NeoN::scalar> checkDense(exec, 9);
        auto checkDenseView = checkDense.view();
        auto denseView = denseMatrixConst.view();
        parallelFor(
            exec,
            {0, 1},
            KOKKOS_LAMBDA(const NeoN::localIdx) {
                checkDenseView[0] = denseView.entry(0, 0);
                checkDenseView[1] = denseView.entry(0, 1);
                checkDenseView[2] = denseView.entry(0, 2);
                checkDenseView[3] = denseView.entry(1, 0);
                checkDenseView[4] = denseView.entry(1, 1);
                checkDenseView[5] = denseView.entry(1, 2);
                checkDenseView[6] = denseView.entry(2, 0);
                checkDenseView[7] = denseView.entry(2, 1);
                checkDenseView[8] = denseView.entry(2, 2);
            }
        );
        checkHost = checkDense.copyToHost();
        REQUIRE(checkHost.view()[0] == 1.0);
        REQUIRE(checkHost.view()[1] == 2.0);
        REQUIRE(checkHost.view()[2] == 3.0);
        REQUIRE(checkHost.view()[3] == 4.0);
        REQUIRE(checkHost.view()[4] == 5.0);
        REQUIRE(checkHost.view()[5] == 6.0);
        REQUIRE(checkHost.view()[6] == 7.0);
        REQUIRE(checkHost.view()[7] == 8.0);
        REQUIRE(checkHost.view()[8] == 9.0);
    }

    SECTION("Update existing entry on " + execName)
    {
        // Sparse
        auto csrView = sparseMatrix.view();
        parallelFor(
            exec,
            {0, 1},
            KOKKOS_LAMBDA(const NeoN::localIdx) {
                csrView.entry(0, 0) = -1.0;
                csrView.entry(1, 1) = -5.0;
                csrView.entry(1, 2) = -6.0;
                csrView.entry(2, 1) = -8.0;
            }
        );

        auto hostMatrix = sparseMatrix.copyToHost();
        auto checkHost = hostMatrix.values().view();
        REQUIRE(checkHost[0] == -1.0);
        REQUIRE(checkHost[1] == -5.0);
        REQUIRE(checkHost[2] == -6.0);
        REQUIRE(checkHost[3] == -8.0);

        // Dense
        auto denseView = denseMatrix.view();
        parallelFor(
            exec,
            {0, 1},
            KOKKOS_LAMBDA(const NeoN::localIdx) {
                denseView.entry(0, 0) = -1.0;
                denseView.entry(0, 1) = -2.0;
                denseView.entry(0, 2) = -3.0;
                denseView.entry(1, 0) = -4.0;
                denseView.entry(1, 1) = -5.0;
                denseView.entry(1, 2) = -6.0;
                denseView.entry(2, 0) = -7.0;
                denseView.entry(2, 1) = -8.0;
                denseView.entry(2, 2) = -9.0;
            }
        );

        hostMatrix = denseMatrix.copyToHost();
        checkHost = hostMatrix.values().view();
        REQUIRE(checkHost[0] == -1.0);
        REQUIRE(checkHost[1] == -2.0);
        REQUIRE(checkHost[2] == -3.0);
        REQUIRE(checkHost[3] == -4.0);
        REQUIRE(checkHost[4] == -5.0);
        REQUIRE(checkHost[5] == -6.0);
        REQUIRE(checkHost[6] == -7.0);
        REQUIRE(checkHost[7] == -8.0);
        REQUIRE(checkHost[8] == -9.0);
    }

    SECTION("Read directValue on " + execName)
    {
        // Sparse
        NeoN::Vector<NeoN::scalar> checkSparse(exec, 4);
        auto checkSparseView = checkSparse.view();
        auto csrView = sparseMatrixConst.view();
        parallelFor(
            exec,
            {0, 1},
            KOKKOS_LAMBDA(const NeoN::localIdx) {
                checkSparseView[0] = csrView.entry(0);
                checkSparseView[1] = csrView.entry(1);
                checkSparseView[2] = csrView.entry(2);
                checkSparseView[3] = csrView.entry(3);
            }
        );
        auto checkHost = checkSparse.copyToHost();
        REQUIRE(checkHost.view()[0] == 1.0);
        REQUIRE(checkHost.view()[1] == 5.0);
        REQUIRE(checkHost.view()[2] == 6.0);
        REQUIRE(checkHost.view()[3] == 8.0);

        // Dense
        NeoN::Vector<NeoN::scalar> checkDense(exec, 9);
        auto checkDenseView = checkDense.view();
        auto denseView = denseMatrixConst.view();
        parallelFor(
            exec,
            {0, 1},
            KOKKOS_LAMBDA(const NeoN::localIdx) {
                checkDenseView[0] = denseView.entry(0);
                checkDenseView[1] = denseView.entry(1);
                checkDenseView[2] = denseView.entry(2);
                checkDenseView[3] = denseView.entry(3);
                checkDenseView[4] = denseView.entry(4);
                checkDenseView[5] = denseView.entry(5);
                checkDenseView[6] = denseView.entry(6);
                checkDenseView[7] = denseView.entry(7);
                checkDenseView[8] = denseView.entry(8);
            }
        );
        checkHost = checkDense.copyToHost();
        REQUIRE(checkHost.view()[0] == 1.0);
        REQUIRE(checkHost.view()[1] == 2.0);
        REQUIRE(checkHost.view()[2] == 3.0);
        REQUIRE(checkHost.view()[3] == 4.0);
        REQUIRE(checkHost.view()[4] == 5.0);
        REQUIRE(checkHost.view()[5] == 6.0);
        REQUIRE(checkHost.view()[6] == 7.0);
        REQUIRE(checkHost.view()[7] == 8.0);
        REQUIRE(checkHost.view()[8] == 9.0);
    }

    SECTION("Update existing directValue on " + execName)
    {
        // Sparse
        auto csrView = sparseMatrix.view();
        parallelFor(
            exec,
            {0, 1},
            KOKKOS_LAMBDA(const NeoN::localIdx) {
                csrView.entry(0) = -1.0;
                csrView.entry(1) = -5.0;
                csrView.entry(2) = -6.0;
                csrView.entry(3) = -8.0;
            }
        );


        auto hostMatrix = sparseMatrix.copyToHost();
        auto checkHost = hostMatrix.values().view();
        REQUIRE(checkHost[0] == -1.0);
        REQUIRE(checkHost[1] == -5.0);
        REQUIRE(checkHost[2] == -6.0);
        REQUIRE(checkHost[3] == -8.0);

        // Dense
        auto denseView = denseMatrix.view();
        parallelFor(
            exec,
            {0, 1},
            KOKKOS_LAMBDA(const NeoN::localIdx) {
                denseView.entry(0) = -1.0;
                denseView.entry(1) = -2.0;
                denseView.entry(2) = -3.0;
                denseView.entry(3) = -4.0;
                denseView.entry(4) = -5.0;
                denseView.entry(5) = -6.0;
                denseView.entry(6) = -7.0;
                denseView.entry(7) = -8.0;
                denseView.entry(8) = -9.0;
            }
        );

        hostMatrix = denseMatrix.copyToHost();
        checkHost = hostMatrix.values().view();
        REQUIRE(checkHost[0] == -1.0);
        REQUIRE(checkHost[1] == -2.0);
        REQUIRE(checkHost[2] == -3.0);
        REQUIRE(checkHost[3] == -4.0);
        REQUIRE(checkHost[4] == -5.0);
        REQUIRE(checkHost[5] == -6.0);
        REQUIRE(checkHost[6] == -7.0);
        REQUIRE(checkHost[7] == -8.0);
        REQUIRE(checkHost[8] == -9.0);
    }

    SECTION("View " + execName)
    {
        auto hostMatrix = sparseMatrix.copyToHost();
        auto [value, column, row] = hostMatrix.view();
        auto hostvaluesSparse = valuesSparse.copyToHost();
        auto hostcolIdxSparse = colIdxSparse.copyToHost();
        auto hostrowPtrsSparse = rowPtrsSparse.copyToHost();

        REQUIRE(hostvaluesSparse.size() == value.size());
        REQUIRE(hostcolIdxSparse.size() == column.size());
        REQUIRE(hostrowPtrsSparse.size() == row.size());

        for (size_t i = 0; i < value.size(); ++i)
        {
            REQUIRE(hostvaluesSparse.view()[i] == value[i]);
            REQUIRE(hostcolIdxSparse.view()[i] == column[i]);
        }
        for (size_t i = 0; i < row.size(); ++i)
        {
            REQUIRE(hostrowPtrsSparse.view()[i] == row[i]);
        }
    }
}
