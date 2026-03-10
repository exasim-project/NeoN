// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include "NeoN/core/vector/vector.hpp"
#include "NeoN/linearAlgebra/cooSparsityPattern.hpp"
#include "NeoN/linearAlgebra/csrSparsityPattern.hpp"
#include "NeoN/linearAlgebra/faceToMatrixAddress.hpp"

namespace NeoN::la
{

/**
 * @struct MatrixView
 * @brief A view struct to allow easy read/write on all executors.
 *
 * @tparam ValueType The value type of the non-zero entries.
 * @tparam SparsityViewType The type of the sparsity pattern, eg COO or CSR.
 */
template<typename ValueType, typename SparsityViewType>
struct MatrixView
{
    /**
     * @brief Constructor for MatrixView.
     * @param valueView View of the non-zero values of the matrix.
     * @param colIdxsView View of the column indices for each non-zero value.
     * @param rowOffsView View of the starting index in values/colIdxs for each row.
     */
    MatrixView(const View<ValueType> valueView, SparsityViewType sparsityView)
        : values(valueView), sparsity(sparsityView) {};

    /**
     * @brief Default destructor.
     */
    ~MatrixView() = default;

    /**
     * @brief Retrieve a reference to the matrix element at position (i,j).
     * @param i The row index.
     * @param j The column index.
     * @return Reference to the matrix element if it exists.
     */
    KOKKOS_INLINE_FUNCTION
    ValueType& entry(const localIdx i, const localIdx j) const
    {
        return values[sparsity.entry(i, j)];
    }

    /**
     * @brief Direct access to a value given the offset.
     * @param offset The offset, from 0, to the value.
     * @return Reference to the matrix element if it exists.
     */
    KOKKOS_INLINE_FUNCTION
    ValueType& entry(const localIdx offset) const { return values[offset]; }

    View<ValueType> values; //!< View to the values of the CSR matrix.
    SparsityViewType sparsity;
};

/**
 * @class Matrix
 * @brief Sparse matrix class with compact storage by row (CSR) format.
 * @tparam ValueType The value type of the non-zero entries.
 * @tparam IndexType The index type of the rows and columns.
 */
template<typename ValueType, typename SparsityType>
class Matrix : public NeoN::SupportsCopyTo<Matrix<ValueType, SparsityType>>
{

    void validate()
    {
        NF_ASSERT(values_.exec() == sparsityPattern_->exec(), "Executors are not the same");
        // TODO this is not necessarily true for matrix types with padding like ELL
        NF_ASSERT(values_.size() == sparsityPattern_->nnz(), "Matrix values and columns mismatch");
    }

public:

    using MatrixValueType = ValueType;
    using MatrixSparsityType = SparsityType;

    /**
     * @brief Constructor for Matrix.
     * @param values The non-zero values of the matrix.
     * @param sp The sparsity pattern of the matrix
     */
    Matrix(const Vector<ValueType>& values, std::shared_ptr<const SparsityType> sp)
        : values_(values), sparsityPattern_(sp)
    {
        validate();
    }

    /**
     * @brief Constructor for Matrix.
     * @param values The non-zero values of the matrix.
     * @param colIdxs The column indices for each non-zero value.
     * @param rowOffs The starting index in values/colIdxs for each row.
     */
    Matrix(
        const Vector<ValueType>& values,
        const Vector<typename SparsityType::SparsityIndexType>& colIdxs,
        const Vector<typename SparsityType::SparsityIndexType>& rowOffs,
        Dimensions dimensions
    )
        : values_(values),
          sparsityPattern_(
              std::make_shared<const SparsityType>(Vector(colIdxs), Vector(rowOffs), dimensions)
          )
    {
        validate();
    }

    /**
     * @brief Constructor for Matrix with a FaceToMatrixAddress.
     *
     * The sparsity pattern and the face-to-matrix address are passed as independent objects.
     * Only available for matrices whose index type is localIdx.
     *
     * @param values The non-zero values of the matrix.
     * @param sparsity The sparsity pattern of the matrix.
     * @param faceToMatrixAddress The face-to-matrix address mapping.
     */
    Matrix(
        const Vector<ValueType>& values,
        std::shared_ptr<const SparsityType> sparsity,
        std::shared_ptr<const FaceToMatrixAddress> faceToMatrixAddress
    )
        requires std::is_same_v<typename SparsityType::SparsityIndexType, localIdx>
        : values_(values), sparsityPattern_(sparsity), faceToMatrixAddress_(faceToMatrixAddress)
    {
        validate();
    }

    /**
     * @brief Default destructor.
     */
    ~Matrix() = default;

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

    void reset() { values_ = ValueType {}; }


    /**
     * @brief Get a reference to values vector.
     * @return Vector containing the matrix values.
     */
    [[nodiscard]] Vector<ValueType>& values() { return values_; }

    /**
     * @brief Get a reference to column indices vector.
     * @return Vector containing the column indices.
     */
    [[nodiscard]] const Vector<typename MatrixSparsityType::SparsityIndexType>& colIdxs() const
    {
        return sparsityPattern_->colIdxs();
    }

    /**
     * @brief Get a reference to row offset vector.
     * @return Vector containing the row pointers.
     */
    [[nodiscard]] const Vector<typename MatrixSparsityType::SparsityIndexType>& rowOffs() const
    {
        return sparsityPattern_->rowOffs();
    }

    /**
     * @brief Get a const reference to values vector.
     * @return Const vector containing the matrix values.
     */
    [[nodiscard]] const Vector<ValueType>& values() const { return values_; }

    /**
     * @brief Copy the matrix to another executor.
     * @param dstExec The destination executor.
     * @return A copy of the matrix on the destination executor.
     */
    [[nodiscard]] Matrix<ValueType, SparsityType> copyToExecutor(Executor dstExec) const override;

    /**
     * @brief Get a reference to column indices vector.
     * @return Vector containing the column indices.
     */
    [[nodiscard]] std::shared_ptr<const SparsityType> sparsity() const { return sparsityPattern_; }

    /**
     * @brief Get the FaceToMatrixAddress associated with this matrix (may be null).
     */
    [[nodiscard]] std::shared_ptr<const FaceToMatrixAddress> faceToMatrixAddress() const
    {
        return faceToMatrixAddress_;
    }

    /**
     * @brief Get a view representation of the matrix's data.
     * @return MatrixView for easy access to matrix elements.
     */
    [[nodiscard]] MatrixView<
        ValueType,
        SparsityView<typename MatrixSparsityType::SparsityIndexType>>
    view()
    {
        return MatrixView<ValueType, SparsityView<typename MatrixSparsityType::SparsityIndexType>>(
            values_.view(), sparsityPattern_->view()
        );
    }

    /**
     * @brief Get a const view representation of the matrix's data.
     * @return Const MatrixView for read-only access to matrix elements.
     */
    [[nodiscard]] MatrixView<
        const ValueType,
        SparsityView<typename MatrixSparsityType::SparsityIndexType>>
    view() const
    {
        return MatrixView<
            const ValueType,
            SparsityView<typename MatrixSparsityType::SparsityIndexType>>(
            View<const ValueType>(values_.view()), sparsityPattern_->view()
        );
    }

    /** @brief extract the diagonal of the matrix
     *
     */
    [[nodiscard]] Vector<ValueType> diag() const;


private:

    Vector<ValueType> values_; //!< The (non-zero) values of the CSR matrix.

    std::shared_ptr<const SparsityType> sparsityPattern_;

    std::shared_ptr<const FaceToMatrixAddress> faceToMatrixAddress_;
};


template<typename ValueType, typename IndexType>
using CSRMatrix = Matrix<ValueType, la::CsrSparsityPattern<IndexType>>;

template<typename ValueType, typename IndexType>
using COOMatrix = Matrix<ValueType, la::CooSparsityPattern<IndexType>>;

/** @brief extract the upper triangular of the matrix
 * @note this function is meant for testing purposes, it will recompute upper offsets
 */
template<typename ValueType, typename IndexType>
[[nodiscard]] Vector<ValueType> upper(const CSRMatrix<ValueType, IndexType>&);

/** @brief computes the inverted diagonal of a matrix and scales it by a, ie. a*D^-1
 * @note this function is a specialized function for CSR<Vec3> matrices assuming all diagonal
 * entries are identical
 */
[[nodiscard]] Vector<scalar>
scaledInverseDiag(const CSRMatrix<Vec3, localIdx>&, const Vector<scalar>&);

void scaledInverseDiag(
    const CSRMatrix<Vec3, localIdx>& mtx, const Vector<scalar>& a, Vector<scalar>& out
);

/** @brief computes the inverted diagonal of a matrix and scales it by a, ie. a*D^-1
 * @note this function is a specialized function for CSR<Vec3> matrices assuming all diagonal
 * entries are identical
 */
[[nodiscard]] Vector<scalar>
scaledInverseDiag(const CSRMatrix<Vec3, localIdx>&, const FaceToMatrixAddress& mi, const Vector<scalar>&);

void scaledInverseDiag(
    const CSRMatrix<Vec3, localIdx>& mtx,
    const FaceToMatrixAddress& mi,
    const Vector<scalar>& a,
    Vector<scalar>& out
);

/* @brief given Matrix<Vec3> this function returns a component Matrix<scalar>*/
template<unsigned int I>
[[nodiscard]] auto getComponent(const CSRMatrix<Vec3, localIdx>& in)
{
    auto sparsity = in.sparsity();
    return CSRMatrix<scalar, localIdx>(getComponent<I>(in.values()), sparsity);
}

/** @brief computes out = -(L+U) x
 *
 * @notes explicitly sets out values to zero
 */
void negLUx(
    const CSRMatrix<Vec3, localIdx>& mtx,
    const Vector<Vec3>& a,
    const Vector<Vec3>& b,
    const Vector<scalar>& rAU,
    const Vector<scalar>& vol,
    Vector<Vec3>& out
);

/** @brief computes out = -(L+U) x
 *
 * @notes explicitly sets out values to zero
 */
void scaledInvDiagNegLUx(
    const CSRMatrix<Vec3, localIdx>& mtx,
    const Vector<Vec3>& a,
    const Vector<Vec3>& b,
    const Vector<scalar>& vol,
    Vector<scalar>& rAU,
    Vector<Vec3>& out
);

} // namespace NeoN
