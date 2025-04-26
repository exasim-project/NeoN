// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoN authors
#pragma once

#include "NeoN/core/vector.hpp"
#include "NeoN/linearAlgebra/CSRMatrix.hpp"

#include <string>

namespace NeoN::la
{

/**
 * @struct LinearSystemView
 * @brief A view linear into a linear system's data.
 *
 * @tparam ValueType The value type of the linear system.
 * @tparam IndexType The index type of the linear system.
 */
template<typename ValueType, typename IndexType>
struct LinearSystemView
{
    LinearSystemView() = default;
    ~LinearSystemView() = default;

    LinearSystemView(CSRMatrixView<ValueType, IndexType> matrixView, View<ValueType> rhsView)
        : matrix(matrixView), rhs(rhsView) {};

    CSRMatrixView<ValueType, IndexType> matrix;
    View<ValueType> rhs;
};

/**
 * @class LinearSystem
 * @brief A class representing a linear system of equations.
 *
 * The LinearSystem class provides functionality to store and manipulate a linear system of
 * equations. It supports the storage of the coefficient matrix and the right-hand side vector, as
 * well as the solution vector.
 */
template<typename ValueType, typename IndexType>
class LinearSystem
{
public:

    LinearSystem(const CSRMatrix<ValueType, IndexType>& matrix, const Vector<ValueType>& rhs)
        : matrix_(matrix), rhs_(rhs)
    {
        NF_ASSERT(matrix.exec() == rhs.exec(), "Executors are not the same");
        NF_ASSERT(matrix.nRows() == rhs.size(), "Matrix and RHS size mismatch");
    };

    LinearSystem(const LinearSystem& ls) : matrix_(ls.matrix_), rhs_(ls.rhs_) {};

    LinearSystem(const Executor exec) : matrix_(exec), rhs_(exec, 0) {}

    ~LinearSystem() = default;

    [[nodiscard]] CSRMatrix<ValueType, IndexType>& matrix() { return matrix_; }

    [[nodiscard]] Vector<ValueType>& rhs() { return rhs_; }

    [[nodiscard]] const CSRMatrix<ValueType, IndexType>& matrix() const { return matrix_; }

    [[nodiscard]] const Vector<ValueType>& rhs() const { return rhs_; }

    [[nodiscard]] LinearSystem copyToHost() const
    {
        return LinearSystem(matrix_.copyToHost(), rhs_.copyToHost());
    }

    void reset()
    {
        fill(matrix_.values(), zero<ValueType>());
        fill(rhs_, zero<ValueType>());
    }

    [[nodiscard]] LinearSystemView<ValueType, IndexType> view() && = delete;

    [[nodiscard]] LinearSystemView<ValueType, IndexType> view() const&& = delete;

    [[nodiscard]] LinearSystemView<ValueType, IndexType> view() &
    {
        return LinearSystemView<ValueType, IndexType>(matrix_.view(), rhs_.view());
    }

    [[nodiscard]] LinearSystemView<const ValueType, const IndexType> view() const&
    {
        return LinearSystemView<const ValueType, const IndexType>(matrix_.view(), rhs_.view());
    }

    const Executor& exec() const { return matrix_.exec(); }

private:

    CSRMatrix<ValueType, IndexType> matrix_;
    Vector<ValueType> rhs_;
};


template<typename ValueTypeIn, typename IndexTypeIn, typename ValueTypeOut, typename IndexTypeOut>
LinearSystem<ValueTypeOut, IndexTypeOut>
convertLinearSystem(const LinearSystem<ValueTypeIn, IndexTypeIn>& ls)
{
    auto exec = ls.exec();
    Vector<ValueTypeOut> convertedRhs(exec, ls.rhs().data(), ls.rhs().size());
    return {
        convert<ValueTypeIn, IndexTypeIn, ValueTypeOut, IndexTypeOut>(exec, ls.view.matrix),
        convertedRhs,
        ls.sparsityPattern()
    };
}

/*@brief helper function that creates a zero initialised linear system based on given sparsity
 * pattern
 */
template<typename ValueType, typename IndexType, typename SparsityType>
LinearSystem<ValueType, IndexType> createEmptyLinearSystem(const SparsityType& sparsity)
{
    const auto& exec = sparsity.mesh().exec();

    localIdx rows {sparsity.rows()};
    localIdx nnzs {sparsity.nnz()};

    return {
        CSRMatrix<ValueType, IndexType> {
            Vector<ValueType>(exec, nnzs, zero<ValueType>()), sparsity.colIdxs(), sparsity.rowOffs()
        },
        Vector<ValueType> {exec, rows, zero<ValueType>()}
    };
}


} // namespace NeoN::la
