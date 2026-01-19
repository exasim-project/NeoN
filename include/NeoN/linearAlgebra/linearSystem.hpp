// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include "NeoN/core/vector/vector.hpp"
#include "NeoN/core/dictionary.hpp"
#include "NeoN/linearAlgebra/matrix.hpp"
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
template<typename ValueType, typename MatrixViewType>
struct LinearSystemView
{
    LinearSystemView() = default;
    ~LinearSystemView() = default;

    LinearSystemView(
        MatrixViewType matrixView,
        View<ValueType> rhsView,
        MatrixViewType boundaryMatrixView,
        View<ValueType> boundaryRhsView
    )
        : matrix(matrixView), rhs(rhsView), boundaryMatrix(boundaryMatrixView),
          boundaryRhs(boundaryRhsView) {};

    MatrixViewType matrix;
    View<ValueType> rhs;

    MatrixViewType boundaryMatrix;
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
template<typename ValueType, typename MatrixType>
class LinearSystem
{

    void validate()
    {
        NF_ASSERT(matrix_.exec() == rhs_.exec(), "Executors are not the same");
        NF_ASSERT(matrix_.nRows() == rhs_.size(), "Matrix and RHS size mismatch");
    }

public:

    LinearSystem(
        const MatrixType& matrix,
        const Vector<ValueType>& rhs,
        const MatrixType& boundaryMatrix,
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

    [[nodiscard]] MatrixType& matrix() { return matrix_; }

    [[nodiscard]] const MatrixType& matrix() const { return matrix_; }

    [[nodiscard]] MatrixType& boundaryMatrix() { return boundaryMatrix_; }

    [[nodiscard]] const MatrixType& boundaryMatrix() const { return boundaryMatrix_; }

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

    [[nodiscard]] LinearSystemView<
        ValueType,
        MatrixView<ValueType, SparsityView<typename MatrixType::MatrixSparsityType>>>
    view() && = delete;

    [[nodiscard]] LinearSystemView<
        ValueType,
        MatrixView<ValueType, SparsityView<typename MatrixType::MatrixSparsityType>>>
    view() const&& = delete;

    [[nodiscard]] LinearSystemView<
        ValueType,
        MatrixView<ValueType, SparsityView<typename MatrixType::MatrixSparsityType>>>
    view() &
    {
        return LinearSystemView<
            ValueType,
            MatrixView<ValueType, SparsityView<typename MatrixType::MatrixSparsityType>>>(
            matrix_.view(), rhs_.view(), boundaryMatrix_.view(), boundaryRhs_.view()
        );
    }

    [[nodiscard]] LinearSystemView<
        const ValueType,
        const MatrixView<ValueType, SparsityView<const typename MatrixType::MatrixSparsityType>>>
    view() const&
    {
        return LinearSystemView<
            const ValueType,
            const MatrixView<
                ValueType,
                SparsityView<const typename MatrixType::MatrixSparsityType>>>(
            matrix_.view(), rhs_.view(), boundaryMatrix_.view(), boundaryRhs_.view()
        );
    }

    const Executor& exec() const { return matrix_.exec(); }

private:

    // internal values
    MatrixType matrix_;

    Vector<ValueType> rhs_;

    // boundary values
    MatrixType boundaryMatrix_;

    Vector<ValueType> boundaryRhs_;

    Dictionary auxiliaryCoefficients_;
};


// /*@brief helper function that converts the internal storage type
//  * pattern
//  */
// template<typename ValueTypeIn, typename IndexTypeIn, typename ValueTypeOut, typename
// IndexTypeOut> LinearSystem<ValueTypeOut, IndexTypeOut> convertLinearSystem(const
// LinearSystem<ValueTypeIn, IndexTypeIn>& ls)
// {
//     auto exec = ls.exec();
//     Vector<ValueTypeOut> convertedRhs(exec, ls.rhs().data(), ls.rhs().size());
//     return {
//         convert<ValueTypeIn, IndexTypeIn, ValueTypeOut, IndexTypeOut>(exec, ls.view.matrix),
//         convertedRhs,
//         ls.sparsityPattern()
//     };
// }

/*@brief helper function that creates a zero initialised linear system based on given sparsity
 * pattern
 */
template<typename ValueType, typename SparsityType>
LinearSystem<ValueType, Matrix<ValueType, SparsityType>> createEmptyLinearSystem(
    const UnstructuredMesh& mesh,
    std::shared_ptr<const SparsityType> sparsity,
    std::shared_ptr<const SparsityType> boundarySparsity
)
{
    const auto& exec = mesh.exec();
    localIdx rows {sparsity->rows()};
    localIdx nnzs {sparsity->nnz()};
    localIdx nBoundaryFaces {mesh.boundaryMesh().faceCells().size()};

    return {
        Matrix<ValueType, SparsityType> {
            Vector<ValueType>(exec, nnzs, zero<ValueType>()), sparsity
        },
        Vector<ValueType> {exec, rows, zero<ValueType>()},
        Matrix<ValueType, SparsityType> {
            Vector<ValueType>(exec, nBoundaryFaces, zero<ValueType>()), sparsity
        },
        Vector<ValueType> {exec, nBoundaryFaces, zero<ValueType>()},
    };
}


} // namespace NeoN::la
