// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include "NeoN/core/vector/vector.hpp"
#include "NeoN/core/dictionary.hpp"
#include "NeoN/linearAlgebra/Matrix.hpp"
#include "NeoN/linearAlgebra/sparsityPattern.hpp"
#include "NeoN/linearAlgebra/matrixIterator.hpp"

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

    LinearSystemView(
        MatrixView<ValueType, IndexType> matrixView,
        View<ValueType> rhsView,
        MatrixView<ValueType, IndexType> boundaryMatrixView,
        View<ValueType> boundaryRhsView
    )
        : matrix(matrixView), rhs(rhsView), boundaryMatrix(boundaryMatrixView),
          boundaryRhs(boundaryRhsView) {};

    MatrixView<ValueType, IndexType> matrix;
    View<ValueType> rhs;

    MatrixView<ValueType, IndexType> boundaryMatrix;
    View<ValueType> boundaryRhs;
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

    void validate()
    {
        NF_ASSERT(matrix_.exec() == rhs_.exec(), "Executors are not the same");
        NF_ASSERT(matrix_.nRows() == rhs_.size(), "Matrix and RHS size mismatch");
    }

public:

    LinearSystem(
        const Matrix<ValueType, IndexType>& matrix,
        const Vector<ValueType>& rhs,
        const Matrix<ValueType, IndexType>& boundaryMatrix,
        const Vector<ValueType>& boundaryRhs
    )
        : matrix_(matrix), rhs_(rhs), boundaryMatrix_(boundaryMatrix), boundaryRhs_(boundaryRhs)
    {
        validate();
    }

    LinearSystem(const LinearSystem& ls)
        : matrix_(ls.matrix_), rhs_(ls.rhs_), boundaryMatrix_(ls.boundaryMatrix_),
          boundaryRhs_(ls.boundaryRhs_)
    {}

    ~LinearSystem() = default;

    [[nodiscard]] Matrix<ValueType, IndexType>& matrix() { return matrix_; }

    [[nodiscard]] const Matrix<ValueType, IndexType>& matrix() const { return matrix_; }

    [[nodiscard]] Matrix<ValueType, IndexType>& boundaryMatrix() { return boundaryMatrix_; }

    [[nodiscard]] const Matrix<ValueType, IndexType>& boundaryMatrix() const
    {
        return boundaryMatrix_;
    }

    [[nodiscard]] Vector<ValueType>& rhs() { return rhs_; }

    [[nodiscard]] const Vector<ValueType>& rhs() const { return rhs_; }

    [[nodiscard]] Vector<ValueType>& boundaryRhs() { return boundaryRhs_; }

    [[nodiscard]] const Vector<ValueType>& boundaryRhs() const { return boundaryRhs_; }

    [[nodiscard]] LinearSystem copyToHost() const
    {
        return LinearSystem(
            matrix_.copyToHost(),
            rhs_.copyToHost(),
            boundaryMatrix_.copyToHost(),
            boundaryRhs_.copyToHost()
        );
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
        return LinearSystemView<ValueType, IndexType>(
            matrix_.view(), rhs_.view(), boundaryMatrix_.view(), boundaryRhs_.view()
        );
    }

    [[nodiscard]] LinearSystemView<const ValueType, const IndexType> view() const&
    {
        return LinearSystemView<const ValueType, const IndexType>(
            matrix_.view(), rhs_.view(), boundaryMatrix_.view(), boundaryRhs_.view()
        );
    }

    const Executor& exec() const { return matrix_.exec(); }

private:

    // internal values
    Matrix<ValueType, IndexType> matrix_;

    Vector<ValueType> rhs_;

    // boundary values
    Matrix<ValueType, IndexType> boundaryMatrix_;

    Vector<ValueType> boundaryRhs_;

    Dictionary auxiliaryCoefficients_;
};


/*@brief helper function that converts the internal storage type
 * pattern
 */
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
template<typename ValueType, typename IndexType>
LinearSystem<ValueType, IndexType> createEmptyLinearSystem(
    const UnstructuredMesh& mesh,
    std::shared_ptr<const SparsityPattern<IndexType>> sparsity,
    std::shared_ptr<const SparsityPattern<IndexType>> boundarySparsity
)
{
    const auto& exec = mesh.exec();
    localIdx rows {sparsity->rows()};
    localIdx nnzs {sparsity->nnz()};
    localIdx nBoundaryFaces {mesh.boundaryMesh().faceCells().size()};

    return {
        Matrix<ValueType, IndexType> {Vector<ValueType>(exec, nnzs, zero<ValueType>()), sparsity},
        Vector<ValueType> {exec, rows, zero<ValueType>()},
        Matrix<ValueType, IndexType> {
            Vector<ValueType>(exec, nBoundaryFaces, zero<ValueType>()), sparsity
        },
        Vector<ValueType> {exec, nBoundaryFaces, zero<ValueType>()},
    };
}


} // namespace NeoN::la
