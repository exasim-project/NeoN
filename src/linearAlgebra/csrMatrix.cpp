// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "NeoN/linearAlgebra/CSRMatrix.hpp"

namespace NeoN::la
{

template<typename ValueType, typename IndexType>
Vector<ValueType> CSRMatrix<ValueType, IndexType>::diag() const
{
    auto diag = Vector<ValueType>(values_.exec(), nRows());
    auto [diagV, rowOffsV, colIdxV, matrixV] =
        views(diag, sparsityPattern_->rowOffs(), sparsityPattern_->colIdxs(), values_);

    parallelFor(
        values_.exec(),
        {0, nRows()},
        NEON_LAMBDA(const std::size_t rowi) {
            for (auto i = rowOffsV[rowi]; i < rowOffsV[rowi + 1]; i++)
            {
                if (rowi == colIdxV[i])
                {
                    diagV[rowi] = matrixV[i];
                    break;
                }
            }
        }
    );
    return diag;
}

template<typename ValueType, typename IndexType>
Vector<ValueType> CSRMatrix<ValueType, IndexType>::scaledInverseDiag(const Vector<scalar>& a) const
{
    auto diag = Vector<ValueType>(values_.exec(), nRows());
    scaledInverseDiag(a, diag);
    return diag;
}

template<typename ValueType, typename IndexType>
void CSRMatrix<ValueType, IndexType>::scaledInverseDiag(
    const Vector<scalar>& a, Vector<ValueType>& out
) const
{
    NF_ASSERT(nRows() == a.size(), "Dimension mismatch");

    auto [outV, rowOffsV, colIdxV, matrixV, aV] =
        views(out, sparsityPattern_->rowOffs(), sparsityPattern_->colIdxs(), values_, a);

    parallelFor(
        values_.exec(),
        {0, nRows()},
        NEON_LAMBDA(const std::size_t rowi) {
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
void CSRMatrix<ValueType, IndexType>::negLUx(const Vector<ValueType>& a, Vector<ValueType>& out)
    const
{
    NF_ASSERT(nRows() == a.size(), "Dimension mismatch");
    NF_ASSERT(nRows() == out.size(), "Dimension mismatch");

    auto [rowOffsV, colIdxV, matrixV, outV] =
        views(sparsityPattern_->rowOffs(), sparsityPattern_->colIdxs(), values_, out);
    const auto aV = a.view();

    parallelFor(
        values_.exec(),
        {0, nRows()},
        NEON_LAMBDA(const std::size_t rowi) {
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
    template class CSRMatrix<VALUETYPE, INTEGERTYPE>

NN_DECLARE_CSRMATRIX(scalar, localIdx);
NN_DECLARE_CSRMATRIX(Vec3, int);

}
