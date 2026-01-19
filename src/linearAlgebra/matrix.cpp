// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "NeoN/core/primitives/vec3.hpp" // for Vec3
#include "NeoN/core/primitives/label.hpp"
#include "NeoN/core/parallelAlgorithms.hpp"
#include "NeoN/linearAlgebra/matrix.hpp"

namespace NeoN::la
{

template<typename ValueType, typename IndexType>
Vector<ValueType> Matrix<ValueType, IndexType>::diag() const
{
    auto diag = Vector<ValueType>(values_.exec(), nRows());
    auto [diagV, rowOffsV, colIdxV, matrixV] =
        views(diag, sparsityPattern_->rowOffs(), sparsityPattern_->colIdxs(), values_);

    parallelFor(
        values_.exec(),
        {0, nRows()},
        NEON_LAMBDA(const localIdx rowi) {
            for (auto i = rowOffsV[rowi]; i < rowOffsV[rowi + 1]; i++)
            {
                if (rowi == colIdxV[i])
                {
                    diagV[rowi] = matrixV[i];
                    break;
                }
            }
        },
        "copyDiag"
    );
    return diag;
}

template<typename ValueType, typename IndexType>
Vector<ValueType> upper(const Matrix<ValueType, IndexType>& mtx)
{
    localIdx nRows = mtx.nRows();
    localIdx nUpper = (mtx.nNonZeros() - mtx.nRows()) / 2;
    auto exec = mtx.exec();

    auto upper = Vector<ValueType>(exec, nUpper);
    auto count = Vector<IndexType>(exec, nRows, 0);
    auto offset = Vector<IndexType>(exec, nRows + 1, 0);

    auto [upperV, rowOffsV, colIdxV, matrixV, countV, offsetV] =
        views(upper, mtx.rowOffs(), mtx.colIdxs(), mtx.values(), count, offset);

    // A three step process to copy only the upper matrix
    // values to a return value:
    // 1. count number of upper values per row eg. [2, 2, 1, 0]
    // 2. sum count to generate offset in upper array eg. [0, 2, 4, 5, 5]
    // 3. copy all upper values into return value based on offset
    parallelFor(
        exec,
        {0, nRows},
        NEON_LAMBDA(const localIdx rowi) {
            for (auto i = rowOffsV[rowi]; i < rowOffsV[rowi + 1]; i++)
            {
                if (colIdxV[i] > rowi)
                {
                    Kokkos::atomic_inc(&countV[rowi]);
                }
            }
        },
        "computeNumUpperValues"
    );

    parallelScan(
        exec,
        {1, offsetV.size()},
        NEON_LAMBDA(const NeoN::localIdx i, NeoN::localIdx& update, const bool final) {
            update += countV[i - 1];
            if (final)
            {
                offsetV[i] = update;
            }
        }
    );

    parallelFor(
        exec,
        {0, nRows},
        NEON_LAMBDA(const localIdx rowi) {
            label j = 0; // index of nth element found in this row
            for (auto i = rowOffsV[rowi]; i < rowOffsV[rowi + 1]; i++)
            {
                if (colIdxV[i] > rowi)
                {
                    upperV[offsetV[rowi] + j] = matrixV[i];
                    j++;
                }
            }
        },
        "copyUpperMatrixValues"
    );
    return upper;
}

Vector<ValueType> Matrix<ValueType, IndexType>::scaledInverseDiag(const Vector<scalar>& a) const
{
    auto diag = Vector<ValueType>(values_.exec(), nRows());
    scaledInverseDiag(a, diag);
    return diag;
}

template<typename ValueType, typename IndexType>
void Matrix<ValueType, IndexType>::scaledInverseDiag(
    const Vector<scalar>& a, Vector<ValueType>& out
) const
{
    NF_ASSERT(nRows() == a.size(), "Dimension mismatch");

    auto [outV, rowOffsV, colIdxV, matrixV, aV] =
        views(out, sparsityPattern_->rowOffs(), sparsityPattern_->colIdxs(), values_, a);

    parallelFor(
        values_.exec(),
        {0, nRows()},
        NEON_LAMBDA(const localIdx rowi) {
            for (auto i = rowOffsV[rowi]; i < rowOffsV[rowi + 1]; i++)
            {
                if (rowi == colIdxV[i])
                {
                    outV[rowi] = aV[rowi] * inv(matrixV[i]);
                    break;
                }
            }
        }
    );
}

template<typename ValueType, typename IndexType>
void Matrix<ValueType, IndexType>::negLUx(const Vector<ValueType>& a, Vector<ValueType>& out) const
{
    NF_ASSERT(nRows() == a.size(), "Dimension mismatch");
    NF_ASSERT(nRows() == out.size(), "Dimension mismatch");

    auto [rowOffsV, colIdxV, matrixV, outV] =
        views(sparsityPattern_->rowOffs(), sparsityPattern_->colIdxs(), values_, out);
    const auto aV = a.view();

    parallelFor(
        values_.exec(),
        {0, nRows()},
        NEON_LAMBDA(const localIdx rowi) {
            outV[rowi] = zero<ValueType>();
            for (auto i = rowOffsV[rowi]; i < rowOffsV[rowi + 1]; i++)
            {
                auto colI = colIdxV[i];
                if (rowi != colI)
                {
                    outV[rowi] -= matrixV[i] * aV[colI];
                }
            }
        }
    );
}


#define NN_DECLARE_CSRMATRIX(VALUETYPE, INTEGERTYPE)                                               \
    template class Matrix<VALUETYPE, la::SparsityPattern<INTEGERTYPE>> \
    template Vector<VALUETYPE>                                                                     \
    upper<VALUETYPE, INTEGERTYPE>(const CSRMatrix<VALUETYPE, INTEGERTYPE>&)

NN_DECLARE_CSRMATRIX(scalar, localIdx);
NN_DECLARE_CSRMATRIX(Vec3, int);

}
