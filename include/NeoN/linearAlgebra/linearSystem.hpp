// SPDX-FileCopyrightText: 2023 - 2025 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include "NeoN/core/vector/vector.hpp"
#include "NeoN/core/dictionary.hpp"
#include "NeoN/linearAlgebra/CSRMatrix.hpp"
#include "NeoN/linearAlgebra/sparsityPattern.hpp"

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

// TODO move to fvcc
template<typename ValueType, typename IndexType>
struct BoundaryCoefficients
{
    Vector<ValueType> matrixValues;
    Vector<IndexType> matrixIdxs;
    Vector<ValueType> rhsValues;
    Vector<IndexType> rhsIdxs;
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

    LinearSystem(
        const CSRMatrix<ValueType, IndexType>& matrix,
        const Vector<ValueType>& rhs,
        const Dictionary& aux = {}
    )
        : matrix_(matrix), rhs_(rhs), auxiliaryCoefficients_(aux)
    {
        NF_ASSERT(matrix.exec() == rhs.exec(), "Executors are not the same");
        NF_ASSERT(matrix.nRows() == rhs.size(), "Matrix and RHS size mismatch");
    };

    LinearSystem(const LinearSystem& ls)
        : matrix_(ls.matrix_), rhs_(ls.rhs_), auxiliaryCoefficients_(ls.auxiliaryCoefficients_) {};

    LinearSystem(const Executor exec) : matrix_(exec), rhs_(exec, 0), auxiliaryCoefficients_() {}

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

    // TODO move to fvcc
    [[nodiscard]] const Dictionary& auxiliaryCoefficients() const { return auxiliaryCoefficients_; }

    [[nodiscard]] Dictionary& auxiliaryCoefficients() { return auxiliaryCoefficients_; }

private:

    CSRMatrix<ValueType, IndexType> matrix_;
    Vector<ValueType> rhs_;
    Dictionary auxiliaryCoefficients_;
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
 * FIXME sparsity should be const
 */
template<typename ValueType, typename IndexType>
LinearSystem<ValueType, IndexType>
createEmptyLinearSystem(const UnstructuredMesh& mesh, SparsityPattern& sparsity)
{
    const auto& exec = mesh.exec();
    localIdx rows {sparsity.rows()};
    localIdx nnzs {sparsity.nnz()};

    localIdx nBoundaryFaces {mesh.boundaryMesh().faceCells().size()};

    const auto [diagOffset, rowOffs, faceCells] =
        views(sparsity.diagOffset(), sparsity.rowOffs(), mesh.boundaryMesh().faceCells());

    BoundaryCoefficients<ValueType, IndexType> bcCoeffs {
        Vector<ValueType>(exec, nBoundaryFaces),
        Vector<IndexType>(exec, nBoundaryFaces),
        Vector<ValueType>(exec, nBoundaryFaces),
        Vector<IndexType>(exec, nBoundaryFaces)
    };

    auto [mValue, mColIdx, rhsValue, rhsIdx] =
        views(bcCoeffs.matrixValues, bcCoeffs.matrixIdxs, bcCoeffs.rhsValues, bcCoeffs.rhsIdxs);

    parallelFor(
        exec,
        {0, nBoundaryFaces},
        KOKKOS_LAMBDA(const localIdx bfacei) {
            localIdx celli = faceCells[bfacei];

            mValue[bfacei] = zero<ValueType>();
            mColIdx[bfacei] = celli + diagOffset[celli];
            rhsValue[bfacei] = zero<ValueType>();
            rhsIdx[bfacei] = celli;
        },
        "createEmptyLinearSystem"
    );

    Dictionary aux;
    aux.insert("boundaryCoefficients", bcCoeffs);

    return {
        CSRMatrix<ValueType, IndexType> {
            Vector<ValueType>(exec, nnzs, zero<ValueType>()), sparsity
        },
        Vector<ValueType> {exec, rows, zero<ValueType>()},
        aux
    };
}


} // namespace NeoN::la
