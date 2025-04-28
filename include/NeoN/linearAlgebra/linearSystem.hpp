// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoN authors
#pragma once

#include "NeoN/core/vector.hpp"
#include "NeoN/linearAlgebra/CSRMatrix.hpp"

#include "NeoN/finiteVolume/cellCentred/fields/volumeField.hpp"
#include "NeoN/finiteVolume/cellCentred/linearAlgebra/sparsityPattern.hpp"

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
template<typename ValueType>
struct LinearSystemView
{
    LinearSystemView() = default;
    ~LinearSystemView() = default;

    LinearSystemView(CSRMatrixView<ValueType, const localIdx> matrixView, View<ValueType> rhsView)
        : matrix(matrixView), rhs(rhsView) {};

    CSRMatrixView<ValueType, const localIdx> matrix;
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
template<typename ValueType>
class LinearSystem
{
public:

    LinearSystem(const CSRMatrix<ValueType, localIdx>& matrix, const Vector<ValueType>& rhs)
        : matrix_(matrix), rhs_(rhs)
    {
        NF_ASSERT(matrix.exec() == rhs.exec(), "Executors are not the same");
        NF_ASSERT(matrix.nRows() == rhs.size(), "Matrix and RHS size mismatch");
    };

    LinearSystem(const LinearSystem& ls) : matrix_(ls.matrix_), rhs_(ls.rhs_) {};

    LinearSystem(const Executor exec) : matrix_(exec), rhs_(exec, 0) {}

    ~LinearSystem() = default;

    /* @brief create an empty linear system, ie every matrix coefficient and rhs value are zero
     *
     */
    [[nodiscard]] static LinearSystem<ValueType>
    createEmpty(const finiteVolume::cellCentred::SparsityPattern& sp)
    {
        const auto& exec = sp.mesh().exec();
        return {
            CSRMatrix<ValueType, localIdx> {
                Vector<ValueType>(exec, sp.nnz(), zero<ValueType>()), sp.colIdxs(), sp.rowOffs()
            },
            Vector<ValueType> {exec, sp.rows(), zero<ValueType>()}
        };
    }

    [[nodiscard]] CSRMatrix<ValueType, localIdx>& matrix() { return matrix_; }

    [[nodiscard]] Vector<ValueType>& rhs() { return rhs_; }

    [[nodiscard]] const CSRMatrix<ValueType, localIdx>& matrix() const { return matrix_; }

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

    [[nodiscard]] LinearSystemView<ValueType> view() && = delete;

    [[nodiscard]] LinearSystemView<ValueType> view() const&& = delete;

    [[nodiscard]] LinearSystemView<ValueType> view() &
    {
        return LinearSystemView<ValueType>(matrix_.view(), rhs_.view());
    }

    [[nodiscard]] LinearSystemView<const ValueType> view() const&
    {
        return LinearSystemView<const ValueType>(matrix_.view(), rhs_.view());
    }

    const Executor& exec() const { return matrix_.exec(); }

private:

    CSRMatrix<ValueType, localIdx> matrix_;
    Vector<ValueType> rhs_;
};


// template<typename ValueTypeIn, typename IndexTypeIn, typename ValueTypeOut, typename
// IndexTypeOut> LinearSystem<ValueTypeOut> convertLinearSystem(const LinearSystem<ValueTypeIn,
// IndexTypeIn>& ls)
// {
//     auto exec = ls.exec();
//     Vector<ValueTypeOut> convertedRhs(exec, ls.rhs().data(), ls.rhs().size());
//     return {
//         convert<ValueTypeIn, IndexTypeIn, ValueTypeOut, IndexTypeOut>(exec, ls.view.matrix),
//         convertedRhs,
//         ls.sparsityPattern()
//     };
// }

template<typename ValueType>
finiteVolume::cellCentred::VolumeField<ValueType> operator&(
    const LinearSystem<ValueType> ls, const finiteVolume::cellCentred::VolumeField<ValueType>& x
)
{
    finiteVolume::cellCentred::VolumeField<ValueType> res(x);
    computeResidual(ls.matrix(), ls.rhs(), x.internalVector(), res.internalVector());
    return res;
}

} // namespace NeoN::la
