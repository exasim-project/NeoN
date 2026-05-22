// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "NeoN/core/primitives/vec3.hpp" // for Vec3
#include "NeoN/core/primitives/label.hpp"
#include "NeoN/core/parallelAlgorithms.hpp"
#include "NeoN/linearAlgebra/matrix.hpp"

namespace NeoN::la
{

template<typename ValueType, typename SparsityType>
Vector<ValueType> Matrix<ValueType, SparsityType>::diag() const
{
    auto diag = Vector<ValueType>(values_.exec(), nRows());
    fill(diag, zero<ValueType>());
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


template<typename ValueType, typename SparsityType>
Matrix<ValueType, SparsityType> Matrix<ValueType, SparsityType>::copyToExecutor(Executor dstExec
) const
{
    if (dstExec == values_.exec())
    {
        return *this;
    }
    return {
        values_.copyToExecutor(dstExec),
        std::make_shared<const SparsityType>(this->sparsityPattern_->copyToExecutor(dstExec)),
        (faceToMatrixAddress_) ? std::make_shared<const FaceToMatrixAddress>(
            this->faceToMatrixAddress_->copyToExecutor(dstExec)
        )
                               : nullptr
    };
}


// Free functions

template<typename ValueType, typename IndexType>
Vector<ValueType> upper(const CSRMatrix<ValueType, IndexType>& mtx)
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


void negLUx(
    const CSRMatrix<Vec3, localIdx>& mtx,
    const Vector<Vec3>& a,
    const Vector<Vec3>& b,
    const Vector<scalar>& rAU,
    const Vector<scalar>& vol,
    Vector<Vec3>& out
)
{
    NF_ASSERT(mtx.nRows() == a.size(), "Dimension mismatch");
    NF_ASSERT(mtx.nRows() == out.size(), "Dimension mismatch");

    const auto [rowOffsV, colIdxV, matrixV, rAUV, volV, aV, bV] =
        views(mtx.sparsity()->rowOffs(), mtx.sparsity()->colIdxs(), mtx.values(), rAU, vol, a, b);
    auto outV = out.view();

    parallelFor(
        mtx.exec(),
        {0, mtx.nRows()},
        NEON_LAMBDA(const localIdx rowi) {
            outV[rowi] = zero<Vec3>();
            for (auto i = rowOffsV[rowi]; i < rowOffsV[rowi + 1]; i++)
            {
                auto colI = colIdxV[i];
                if (rowi != colI)
                {
                    outV[rowi] -= matrixV[i] * aV[colI];
                }
            }

            outV[rowi] += bV[rowi];
            outV[rowi] *= rAUV[rowi] / volV[rowi];
        }
    );
}

void scaledInvDiagNegLUx(
    const CSRMatrix<Vec3, localIdx>& mtx,
    const Vector<Vec3>& a,
    const Vector<Vec3>& b,
    const Vector<scalar>& vol,
    Vector<scalar>& rAU,
    Vector<Vec3>& out
)
{
    NF_ASSERT(mtx.nRows() == a.size(), "Dimension mismatch");
    NF_ASSERT(mtx.nRows() == out.size(), "Dimension mismatch");

    const auto [rowOffsV, colIdxV, matrixV, rAUV, volV, aV, bV] =
        views(mtx.sparsity()->rowOffs(), mtx.sparsity()->colIdxs(), mtx.values(), rAU, vol, a, b);
    auto outV = out.view();

    parallelFor(
        mtx.exec(),
        {0, mtx.nRows()},
        NEON_LAMBDA(const localIdx rowi) {
            outV[rowi] = zero<Vec3>();
            for (auto i = rowOffsV[rowi]; i < rowOffsV[rowi + 1]; i++)
            {
                auto colI = colIdxV[i];
                if (rowi == colI)
                {
                    rAUV[rowi] = volV[rowi] / matrixV[i][0];
                }
                else
                {
                    outV[rowi] -= matrixV[i] * aV[colI];
                }
            }

            outV[rowi] += bV[rowi];
            outV[rowi] *= rAUV[rowi] / volV[rowi];
        }
    );
}


Vector<scalar> scaledInverseDiag(const CSRMatrix<Vec3, localIdx>& mtx, const Vector<scalar>& a)
{
    auto diag = Vector<scalar>(mtx.exec(), mtx.nRows());
    scaledInverseDiag(mtx, a, diag);
    return diag;
}

void scaledInverseDiag(
    const CSRMatrix<Vec3, localIdx>& mtx, const Vector<scalar>& a, Vector<scalar>& out
)
{
    NF_ASSERT(mtx.nRows() == a.size(), "Dimension mismatch");

    auto [outV, rowOffsV, colIdxV, matrixV, aV] =
        views(out, mtx.sparsity()->rowOffs(), mtx.sparsity()->colIdxs(), mtx.values(), a);

    parallelFor(
        mtx.exec(),
        {0, mtx.nRows()},
        NEON_LAMBDA(const localIdx rowi) {
            for (auto i = rowOffsV[rowi]; i < rowOffsV[rowi + 1]; i++)
            {
                if (rowi == colIdxV[i])
                {
                    outV[rowi] = aV[rowi] * inv(matrixV[i][0]);
                    break;
                }
            }
        }
    );
}

Vector<scalar> scaledInverseDiag(
    const CSRMatrix<Vec3, localIdx>& mtx, const FaceToMatrixAddress& mi, const Vector<scalar>& a
)
{
    auto diag = Vector<scalar>(mtx.exec(), mtx.nRows());
    scaledInverseDiag(mtx, mi, a, diag);
    return diag;
}

void scaledInverseDiag(
    const CSRMatrix<Vec3, localIdx>& mtx,
    const FaceToMatrixAddress& mi,
    const Vector<scalar>& a,
    Vector<scalar>& out
)
{
    NF_ASSERT(mtx.nRows() == a.size(), "Dimension mismatch");

    auto [outV, rowOffsV, colIdxV, matrixV, aV] =
        views(out, mtx.sparsity()->rowOffs(), mtx.sparsity()->colIdxs(), mtx.values(), a);

    const auto diaOffsV = mi.diagOffset().view();

    parallelFor(
        mtx.exec(),
        {0, mtx.nRows()},
        NEON_LAMBDA(const localIdx rowi) {
            outV[rowi] = aV[rowi] / matrixV[rowOffsV[rowi] + diaOffsV[rowi]][0];
        }
    );
}

#define NN_DECLARE_MATRIX(VALUETYPE, INTEGERTYPE)                                                  \
    template class Matrix<VALUETYPE, la::CsrSparsityPattern<INTEGERTYPE>>;                         \
    template class Matrix<VALUETYPE, la::CooSparsityPattern<INTEGERTYPE>>;                         \
    template Vector<VALUETYPE>                                                                     \
    upper<VALUETYPE, INTEGERTYPE>(const CSRMatrix<VALUETYPE, INTEGERTYPE>&)

NN_DECLARE_MATRIX(scalar, localIdx);
NN_DECLARE_MATRIX(Vec3, localIdx);

}
