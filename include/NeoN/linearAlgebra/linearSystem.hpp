// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include "NeoN/core/vector/vector.hpp"
#include "NeoN/core/dictionary.hpp"
#include "NeoN/linearAlgebra/matrix.hpp"
#include "NeoN/linearAlgebra/sparsityPattern.hpp"
#include "NeoN/linearAlgebra/faceToMatrixAddress.hpp"

#include <string>

namespace NeoN::la
{

/**
 * @struct LinearSystemView
 * @brief A view linear into a linear system's data.
 *
 * @tparam ValueType The value type of the linear system.
 * @tparam MatrixViewType The type representing the matrix view
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
template<typename ValueType, typename MatrixType = CSRMatrix<ValueType, localIdx>>
class LinearSystem
{

    void validate()
    {
        NF_ASSERT(matrix_.exec() == rhs_.exec(), "Executors are not the same");
        NF_ASSERT(matrix_.nRows() == rhs_.size(), "Matrix and RHS size mismatch");
        NF_ASSERT(
            boundaryMatrix_.nRows() == boundaryRhs_.size(), "BMatrix.nRows() != boundaryRHS.size()"
        );
    }

public:

    using LinearSystemIndexType = typename MatrixType::MatrixSparsityType::SparsityIndexType;

    LinearSystem(
        std::shared_ptr<const FaceToMatrixAddress<LinearSystemIndexType>> faceToMatrixAddress
    )
        : matrix_(
            Vector<ValueType>(
                faceToMatrixAddress->exec(), faceToMatrixAddress->localNonZeros(), zero<ValueType>()
            ),
            faceToMatrixAddress->sparsityPattern()
        ),
          rhs_(faceToMatrixAddress->exec(), faceToMatrixAddress->localRows(), zero<ValueType>()),
          boundaryMatrix_(
              Vector<ValueType>(
                  faceToMatrixAddress->exec(),
                  faceToMatrixAddress->boundaryNonZeros(),
                  zero<ValueType>()
              ),
              faceToMatrixAddress->boundarySparsityPattern()
          ),
          boundaryRhs_(
              faceToMatrixAddress->exec(),
              faceToMatrixAddress->boundaryNonZeros(),
              zero<ValueType>()
          ),
          faceToMatrixAddress_(faceToMatrixAddress)
    {
        validate();
    }

    LinearSystem(
        const MatrixType& matrix,
        const Vector<ValueType>& rhs,
        const MatrixType& boundaryMatrix,
        const Vector<ValueType>& boundaryRhs,
        std::shared_ptr<const FaceToMatrixAddress<LinearSystemIndexType>> mi
    )
        : matrix_(matrix), rhs_(rhs), boundaryMatrix_(boundaryMatrix), boundaryRhs_(boundaryRhs),
          faceToMatrixAddress_(mi)
    {
        validate();
    }

    LinearSystem(const LinearSystem& ls)
        : matrix_(ls.matrix_), rhs_(ls.rhs_), boundaryMatrix_(ls.boundaryMatrix_),
          boundaryRhs_(ls.boundaryRhs_), faceToMatrixAddress_(ls.faceToMatrixAddress_)
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

    [[nodiscard]] LinearSystem<ValueType, MatrixType> copyToHost() const
    {
        if (faceToMatrixAddress_ == nullptr)
        {
            return {
                matrix_.copyToHost(),
                rhs_.copyToHost(),
                boundaryMatrix_.copyToHost(),
                boundaryRhs_.copyToHost(),
                {}
            };
        }
        auto mi = std::make_shared<FaceToMatrixAddress<LinearSystemIndexType>>(
            faceToMatrixAddress_->ownerOffset().copyToHost(),
            faceToMatrixAddress_->neighbourOffset().copyToHost(),
            faceToMatrixAddress_->diagOffset().copyToHost(),
            std::make_shared<SparsityPattern<LinearSystemIndexType>>(
                faceToMatrixAddress_->sparsityPattern()->copyToHost()
            ),
            std::make_shared<SparsityPattern<LinearSystemIndexType>>(
                faceToMatrixAddress_->boundarySparsityPattern()->copyToHost()
            )
        );
        return {
            matrix_.copyToHost(),
            rhs_.copyToHost(),
            boundaryMatrix_.copyToHost(),
            boundaryRhs_.copyToHost(),
            mi
        };
    }

    void reset()
    {
        fill(matrix_.values(), zero<ValueType>());
        fill(rhs_, zero<ValueType>());
        fill(boundaryMatrix_.values(), zero<ValueType>());
        fill(boundaryRhs_, zero<ValueType>());
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
        MatrixView<ValueType, SparsityView<LinearSystemIndexType>>>
    view() &
    {
        return {matrix_.view(), rhs_.view(), boundaryMatrix_.view(), boundaryRhs_.view()};
    }

    std::shared_ptr<const FaceToMatrixAddress<LinearSystemIndexType>> faceToMatrixAddress() const
    {
        return faceToMatrixAddress_;
    }

    [[nodiscard]] LinearSystemView<
        const ValueType,
        const MatrixView<ValueType, SparsityView<const LinearSystemIndexType>>>
    view() const&
    {
        return {matrix_.view(), rhs_.view(), boundaryMatrix_.view(), boundaryRhs_.view()};
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

    std::shared_ptr<const FaceToMatrixAddress<LinearSystemIndexType>> faceToMatrixAddress_;
};

/*@brief helper function that creates a zero initialised linear system based on a given mesh
 */
template<typename ValueType, typename MatrixType = CSRMatrix<ValueType, localIdx>>
LinearSystem<ValueType, MatrixType> createEmptyLinearSystem(const UnstructuredMesh& mesh)
{
    return {createSparsityPatternFaceToMatrixAddress<NeoN::localIdx>(mesh)};
}


} // namespace NeoN::la
