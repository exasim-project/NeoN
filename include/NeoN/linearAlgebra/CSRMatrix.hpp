// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include "NeoN/core/vector/vector.hpp"
#include "sparsityPattern.hpp"

#include <type_traits>

namespace NeoN::la
{

/**
 * @struct CSRMatrixView
 * @brief A view struct to allow easy read/write on all executors.
 *
 * @tparam ValueType The value type of the non-zero entries.
 * @tparam IndexType The index type of the rows and columns.
 */
template<typename ValueType, typename IndexType>
struct CSRMatrixView
{
    /**
     * @brief Constructor for CSRMatrixView.
     * @param valueView View of the non-zero values of the matrix.
     * @param colIdxsView View of the column indices for each non-zero value.
     * @param rowOffsView View of the starting index in values/colIdxs for each row.
     */
    CSRMatrixView(
        const View<ValueType> valueView,
        const View<const IndexType> colIdxsView,
        const View<const IndexType> rowOffsView
    )
        : values(valueView), colIdxs(colIdxsView), rowOffs(rowOffsView) {};

    /**
     * @brief Default destructor.
     */
    ~CSRMatrixView() = default;

    /**
     * @brief Retrieve a reference to the matrix element at position (i,j).
     * @param i The row index.
     * @param j The column index.
     * @return Reference to the matrix element if it exists.
     */
    KOKKOS_INLINE_FUNCTION
    ValueType& entry(const IndexType i, const IndexType j) const
    {
        const IndexType rowSize = rowOffs[i + 1] - rowOffs[i];
        for (std::remove_const_t<IndexType> ic = 0; ic < rowSize; ++ic)
        {
            const IndexType localCol = rowOffs[i] + ic;
            if (colIdxs[localCol] == j)
            {
                return values[localCol];
            }
            if (colIdxs[localCol] > j) break;
        }
        Kokkos::abort("Memory not allocated for CSR matrix component.");
        return values[0]; // compiler warning suppression.
    }

    /**
     * @brief Direct access to a value given the offset.
     * @param offset The offset, from 0, to the value.
     * @return Reference to the matrix element if it exists.
     */
    KOKKOS_INLINE_FUNCTION
    ValueType& entry(const IndexType offset) const { return values[offset]; }

    View<ValueType> values;        //!< View to the values of the CSR matrix.
    View<const IndexType> colIdxs; //!< View to the column indices of the CSR matrix.
    View<const IndexType> rowOffs; //!< View to the row offsets for the CSR matrix.
};

/**
 * @class CSRMatrix
 * @brief Sparse matrix class with compact storage by row (CSR) format.
 * @tparam ValueType The value type of the non-zero entries.
 * @tparam IndexType The index type of the rows and columns.
 */
template<typename ValueType, typename IndexType>
class CSRMatrix
{

    void validate()
    {
        NF_ASSERT(
            values_.exec() == sparsityPattern_->colIdxs().exec(), "Executors are not the same"
        );
        NF_ASSERT(
            values_.exec() == sparsityPattern_->rowOffs().exec(), "Executors are not the same"
        );
    }

public:

    /**
     * @brief Constructor for CSRMatrix.
     * @param values The non-zero values of the matrix.
     * @param colIdxs The column indices for each non-zero value.
     * @param rowOffs The starting index in values/colIdxs for each row.
     */
    CSRMatrix(const Vector<ValueType>& values, std::shared_ptr<const SparsityPattern<IndexType>> sp)
        : values_(values), sparsityPattern_(sp)
    {
        validate();
    }

    /**
     * @brief Constructor for CSRMatrix.
     * @param values The non-zero values of the matrix.
     * @param colIdxs The column indices for each non-zero value.
     * @param rowOffs The starting index in values/colIdxs for each row.
     */
    CSRMatrix(
        const Vector<ValueType>& values,
        const Vector<IndexType>& colIdxs,
        const Vector<IndexType>& rowOffs
    )
        : values_(values),
          sparsityPattern_(
              std::make_shared<const SparsityPattern<IndexType>>(Vector(colIdxs), Vector(rowOffs))
          )
    {
        validate();
    }

    /**
     * @brief Default destructor.
     */
    ~CSRMatrix() = default;

    /**
     * @brief Get the executor associated with this matrix.
     * @return Reference to the executor.
     */
    [[nodiscard]] const Executor& exec() const { return values_.exec(); }

    /**
     * @brief Get the number of rows in the matrix.
     * @return Number of rows.
     */
    [[nodiscard]] localIdx nRows() const { return sparsityPattern_->rows(); }

    /**
     * @brief Get the number of non-zero values in the matrix.
     * @return Number of non-zero values.
     */
    [[nodiscard]] localIdx nNonZeros() const { return sparsityPattern_->nnz(); }

    /**
     * @brief Get a reference to values vector.
     * @return Vector containing the matrix values.
     */
    [[nodiscard]] Vector<ValueType>& values() { return values_; }

    /**
     * @brief Get a reference to column indices vector.
     * @return Vector containing the column indices.
     */
    [[nodiscard]] const Vector<IndexType>& colIdxs() const { return sparsityPattern_->colIdxs(); }

    /**
     * @brief Get a reference to row offset vector.
     * @return Vector containing the row pointers.
     */
    [[nodiscard]] const Vector<IndexType>& rowOffs() const { return sparsityPattern_->rowOffs(); }

    /**
     * @brief Get a const reference to values vector.
     * @return Const vector containing the matrix values.
     */
    [[nodiscard]] const Vector<ValueType>& values() const { return values_; }

    /**
     * @brief Get a const reference to column indices vector.
     * @return Const vector containing the column indices.
     */
    [[nodiscard]] const Vector<IndexType>& colIdxs() const { return colIdxs_; }

    /**
     * @brief Get a const reference to row offset vector.
     * @return Const vector containing the row pointers.
     */
    [[nodiscard]] const Vector<IndexType>& rowOffs() const { return rowOffs_; }

    /** @brief extract the diagonal of the matrix*/
    [[nodiscard]] Vector<ValueType> diag() const;

    /**
     * @brief Copy the matrix to another executor.
     * @param dstExec The destination executor.
     * @return A copy of the matrix on the destination executor.
     */
    [[nodiscard]] CSRMatrix<ValueType, IndexType> copyToExecutor(Executor dstExec) const
    {
        if (dstExec == values_.exec())
        {
            return *this;
        }
        return {
            values_.copyToHost(),
            std::make_shared<const la::SparsityPattern<IndexType>>(
                this->sparsityPattern_->copyToHost()
            )
        };
    }

    /**
     * @brief Get a reference to column indices vector.
     * @return Vector containing the column indices.
     */
    [[nodiscard]] std::shared_ptr<const SparsityPattern<IndexType>> sparsity() const
    {
        return sparsityPattern_;
    }


    /**
     * @brief Copy the matrix to the host.
     * @return A copy of the matrix on the host.
     */
    [[nodiscard]] CSRMatrix<ValueType, IndexType> copyToHost() const
    {
        return copyToExecutor(SerialExecutor());
    }

    /**
     * @brief Get a view representation of the matrix's data.
     * @return CSRMatrixView for easy access to matrix elements.
     */
    [[nodiscard]] CSRMatrixView<ValueType, IndexType> view()
    {
        return CSRMatrixView(
            values_.view(), sparsityPattern_->colIdxs().view(), sparsityPattern_->rowOffs().view()
        );
    }

    /**
     * @brief Get a const view representation of the matrix's data.
     * @return Const CSRMatrixView for read-only access to matrix elements.
     */
    [[nodiscard]] CSRMatrixView<const ValueType, const IndexType> view() const
    {
        return CSRMatrixView<const ValueType, const IndexType>(
            View<const ValueType>(values_.view()),
            sparsityPattern_->colIdxs().view(),
            sparsityPattern_->rowOffs().view()
        );
    }

    /** @brief extract the diagonal of the matrix
     *
     */
    [[nodiscard]] Vector<ValueType> diag() const;

    /** @brief computes the inverted diagonal of a matrix and scales it by a, ie. a*D^-1
     *
     */
    [[nodiscard]] Vector<ValueType> scaledInverseDiag(const Vector<scalar>& a) const;

    void scaledInverseDiag(const Vector<scalar>& a, Vector<ValueType>& out) const;

    /** @brief computes out = -(L+U) x
     *
     * @notes explicitly sets out values to zero
     */
    void negLUx(const Vector<ValueType>& a, Vector<ValueType>& out) const;

private:

    Vector<ValueType> values_; //!< The (non-zero) values of the CSR matrix.

    std::shared_ptr<const SparsityPattern<IndexType>> sparsityPattern_;
};

/** @brief extract the upper triangular of the matrix
 * @note this function is meant for testing purposes, it will recompute upper offsets
 */
template<typename ValueType, typename IndexType>
[[nodiscard]] Vector<ValueType> upper(const CSRMatrix<ValueType, IndexType>& mtx);

// FIXME
// template<typename ValueType, typename IndexType>
// void faceIterateNegLUx(
//         const CSRMatrix<ValueType, IndexType>& mat,
//         const MatrixIterator& mi,
//         const Vector<ValueType>& a,
//         Vector<ValueType>& out
//     ) const;

// /* @brief given a csr matrix this function copies the matrix and converts to requested target
// types
//  *
//  */
// template<typename ValueTypeIn, typename IndexTypeIn, typename ValueTypeOut, typename
// IndexTypeOut> la::CSRMatrix<ValueTypeOut, IndexTypeOut> convert(const Executor exec, const
// la::CSRMatrixView<const ValueTypeIn, const IndexTypeIn> in)
// {
//     Vector<IndexTypeOut> colIdxsTmp(exec, in.colIdxs.size());
//     Vector<IndexTypeOut> rowOffsTmp(exec, in.rowOffs.size());
//     Vector<ValueTypeOut> valuesTmp(exec, in.values.data(), in.values.size());

//     parallelFor(
//         colIdxsTmp, NEON_LAMBDA(const localIdx i) { return IndexTypeOut {in.colIdxs[i]}; }
//     );
//     parallelFor(
//         rowOffsTmp, NEON_LAMBDA(const localIdx i) { return IndexTypeOut {in.rowOffs[i]}; }
//     );
//     parallelFor(
//         valuesTmp, NEON_LAMBDA(const localIdx i) { return ValueTypeOut {in.values[i]}; }
//     );

//     return la::CSRMatrix<ValueTypeOut, IndexTypeOut> {valuesTmp, colIdxsTmp, rowOffsTmp};
// }


} // namespace NeoN
