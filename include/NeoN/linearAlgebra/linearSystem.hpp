// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include "NeoN/core/vector/vector.hpp"
#include "NeoN/core/dictionary.hpp"
#include "NeoN/linearAlgebra/matrix.hpp"
#include "NeoN/linearAlgebra/cooSparsityPattern.hpp"
#include "NeoN/linearAlgebra/csrSparsityPattern.hpp"
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
template<
    typename ValueType,
    typename SystemMatrixType = CSRMatrix<ValueType, localIdx>,
    typename BoundaryMatrixType = COOMatrix<ValueType, localIdx>>
class LinearSystem
{

    void validate()
    {
        NF_ASSERT(matrix_.exec() == rhs_.exec(), "Executors are not the same");
        NF_ASSERT(matrix_.nRows() == rhs_.size(), "Matrix and RHS size mismatch");
        // NF_ASSERT(
        //     boundaryMatrix_.nRows() == boundaryRhs_.size(), "BMatrix.nRows() !=
        //     boundaryRHS.size()"
        // );
    }

public:

    using LinearSystemIndexType = typename SystemMatrixType::MatrixSparsityType::SparsityIndexType;

    LinearSystem(
        const SystemMatrixType& matrix,
        const Vector<ValueType>& rhs,
        const BoundaryMatrixType& boundaryMatrix,
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

    [[nodiscard]] SystemMatrixType& matrix() { return matrix_; }

    [[nodiscard]] const SystemMatrixType& matrix() const { return matrix_; }

    [[nodiscard]] BoundaryMatrixType& boundaryMatrix() { return boundaryMatrix_; }

    [[nodiscard]] const BoundaryMatrixType& boundaryMatrix() const { return boundaryMatrix_; }

    [[nodiscard]] Vector<ValueType>& rhs() { return rhs_; }

    [[nodiscard]] const Vector<ValueType>& rhs() const { return rhs_; }

    [[nodiscard]] Vector<ValueType>& boundaryRhs() { return boundaryRhs_; }

    [[nodiscard]] const Vector<ValueType>& boundaryRhs() const { return boundaryRhs_; }

    [[nodiscard]] LinearSystem<ValueType, SystemMatrixType, BoundaryMatrixType> copyToHost() const
    {
        return {
            matrix_.copyToHost(),
            rhs_.copyToHost(),
            boundaryMatrix_.copyToHost(),
            boundaryRhs_.copyToHost()
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
        MatrixView<
            ValueType,
            SparsityView<typename SystemMatrixType::MatrixSparsityType::SparsityIndexType>>>
    view() && = delete;

    [[nodiscard]] LinearSystemView<
        ValueType,
        MatrixView<
            ValueType,
            SparsityView<typename SystemMatrixType::MatrixSparsityType::SparsityIndexType>>>
    view() const&& = delete;

    [[nodiscard]] LinearSystemView<
        ValueType,
        MatrixView<ValueType, SparsityView<LinearSystemIndexType>>>
    view() &
    {
        return {matrix_.view(), rhs_.view(), boundaryMatrix_.view(), boundaryRhs_.view()};
    }

    std::shared_ptr<const FaceToMatrixAddress> faceToMatrixAddress() const
    {
        return matrix_.faceToMatrixAddress();
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
    SystemMatrixType matrix_;

    Vector<ValueType> rhs_;

    // boundary values
    BoundaryMatrixType boundaryMatrix_;

    Vector<ValueType> boundaryRhs_;

    Dictionary auxiliaryCoefficients_;
};

/*@brief helper function that creates a zero initialised linear system based on a given mesh
 */
template<
    typename ValueType,
    typename SystemMatrixType = CSRMatrix<ValueType, localIdx>,
    typename BoundaryMatrixType = COOMatrix<ValueType, localIdx>>
LinearSystem<ValueType, SystemMatrixType, BoundaryMatrixType>
createEmptyLinearSystem(const UnstructuredMesh& mesh)
{
    auto [systemSp, ftma] =
        createSparsityPatternFaceToMatrixAddress<typename SystemMatrixType::MatrixSparsityType>(mesh
        );
    auto bSp =
        createBoundarySparsityPattern<typename BoundaryMatrixType::MatrixSparsityType>(mesh, *ftma);
    return {
        SystemMatrixType(
            Vector<ValueType>(systemSp->exec(), systemSp->nnz(), zero<ValueType>()), systemSp, ftma
        ),
        Vector<ValueType>(systemSp->exec(), systemSp->rows(), zero<ValueType>()),
        BoundaryMatrixType(Vector<ValueType>(bSp->exec(), bSp->nnz(), zero<ValueType>()), bSp),
        Vector<ValueType>(bSp->exec(), bSp->nnz(), zero<ValueType>())
    };
}

/** @brief for testing purposes, this function reverses boundary contributions previously applied to
 * the matrix diagonal and RHS for some operators (e.g., div). **/
template<typename ValueType>
inline la::LinearSystem<ValueType>
removeBoundaryContributions(const la::LinearSystem<ValueType>& lsIn)
{
    auto ls = la::LinearSystem<ValueType>(lsIn);
    auto lsView = ls.view();
    auto& matrix = lsView.matrix;
    auto& rhs = lsView.rhs;
    auto& bMatrix = lsView.boundaryMatrix;
    auto& bRhs = lsView.boundaryRhs;

    const auto ma = ls.faceToMatrixAddress()->view(ls.matrix().sparsity()->rowOffs().view());

    parallelFor(
        ls.exec(),
        {0, bMatrix.values.size()},
        NEON_LAMBDA(const localIdx facei) {
            const auto celli = bMatrix.sparsity.rowOffs[facei]; // cell index stored in rowOffs
            Kokkos::atomic_add(&matrix.values[ma.diagIdx(celli)], bMatrix.values[facei]);
            Kokkos::atomic_add(&rhs[celli], bRhs[facei]);
        },
        "removeBoundaryContributions"
    );

    return ls;
}

} // namespace NeoN::la
