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
    auto [diagV, matrixV] = views(diag, values_);
    const auto sparsityV = sparsityPattern_->view();

    // Lenient: a missing diagonal keeps the zero fill() above instead of aborting.
    // Traversal lives in *SparsityView::findEntry(), not here.
    parallelFor(
        values_.exec(),
        {0, nRows()},
        NEON_LAMBDA(const localIdx rowi) {
            const auto offset = sparsityV.findEntry(rowi, rowi);
            if (offset != decltype(sparsityV)::invalidIndex())
            {
                diagV[rowi] = matrixV[offset];
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
    auto copiedValues = values_.copyToExecutor(dstExec);
    auto copiedSparsity =
        std::make_shared<const SparsityType>(sparsityPattern_->copyToExecutor(dstExec));

    if constexpr (requires(const FaceToMatrixAddress& address, const SparsityType& sp) {
                      address.view(sp.view());
                  })
    {
        if (faceToMatrixAddress_)
        {
            return {
                copiedValues,
                copiedSparsity,
                std::make_shared<const FaceToMatrixAddress>(
                    faceToMatrixAddress_->copyToExecutor(dstExec)
                )
            };
        }
    }
    // faceToMatrixAddress_ can only be set via the constrained 3-arg constructor above, so
    // it's null here whenever that constraint isn't met. Assert it so a future change can't
    // silently drop it.
    NF_ASSERT(!faceToMatrixAddress_, "Face address requires a supported sparsity format");
    return {copiedValues, copiedSparsity};
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

template<typename SparsityType>
void scaledInvDiagNegLUx(
    const Matrix<Vec3, SparsityType>& mtx,
    const Vector<Vec3>& a,
    const Vector<Vec3>& b,
    const Vector<scalar>& vol,
    Vector<scalar>& rAU,
    Vector<Vec3>& out
)
{
    NF_ASSERT(mtx.nRows() == a.size(), "Dimension mismatch");
    NF_ASSERT(mtx.nRows() == b.size(), "Dimension mismatch");
    NF_ASSERT(mtx.nRows() == vol.size(), "Dimension mismatch");
    NF_ASSERT(mtx.nRows() == rAU.size(), "Dimension mismatch");
    NF_ASSERT(mtx.nRows() == out.size(), "Dimension mismatch");

    const auto [matrixV, sparsity] = mtx.view();
    auto [rAUV, volV, aV, bV, outV] = views(rAU, vol, a, b, out);

    parallelFor(
        mtx.exec(),
        {0, mtx.nRows()},
        NEON_LAMBDA(const localIdx rowi) {
            outV[rowi] = zero<Vec3>();
            const auto rowSize = sparsity.rowSize(rowi);
            for (localIdx slot = 0; slot < rowSize; ++slot)
            {
                const auto idx = sparsity.linearIndex(rowi, slot);
                const auto col = sparsity.colIdxs[idx];
                if (col == decltype(sparsity)::invalidIndex()) break; // ELL padding
                if (rowi == col)
                {
                    rAUV[rowi] = volV[rowi] / matrixV[idx][0];
                }
                else
                {
                    outV[rowi] -= matrixV[idx] * aV[col];
                }
            }

            outV[rowi] += bV[rowi];
            outV[rowi] *= rAUV[rowi] / volV[rowi];
        }
    );
}

template<typename SparsityType>
void scaledInvDiagNegLUx(
    const Matrix<scalar, SparsityType>& mtx,
    const Vector<Vec3>& a,
    const Vector<Vec3>& b,
    const Vector<scalar>& vol,
    Vector<scalar>& rAU,
    Vector<Vec3>& out
)
{
    NF_ASSERT(mtx.nRows() == a.size(), "Dimension mismatch");
    NF_ASSERT(mtx.nRows() == b.size(), "Dimension mismatch");
    NF_ASSERT(mtx.nRows() == vol.size(), "Dimension mismatch");
    NF_ASSERT(mtx.nRows() == rAU.size(), "Dimension mismatch");
    NF_ASSERT(mtx.nRows() == out.size(), "Dimension mismatch");

    const auto [matrixV, sparsity] = mtx.view();
    auto [rAUV, volV, aV, bV, outV] = views(rAU, vol, a, b, out);

    parallelFor(
        mtx.exec(),
        {0, mtx.nRows()},
        NEON_LAMBDA(const localIdx rowi) {
            outV[rowi] = zero<Vec3>();
            const auto rowSize = sparsity.rowSize(rowi);
            for (localIdx slot = 0; slot < rowSize; ++slot)
            {
                const auto idx = sparsity.linearIndex(rowi, slot);
                const auto col = sparsity.colIdxs[idx];
                if (col == decltype(sparsity)::invalidIndex()) break; // ELL padding
                if (rowi == col)
                {
                    // scalar diagonal coefficient scales all components equally
                    rAUV[rowi] = volV[rowi] / matrixV[idx];
                }
                else
                {
                    // scalar * Vec3 broadcasts the off-diagonal coefficient to each component
                    outV[rowi] -= matrixV[idx] * aV[col];
                }
            }

            outV[rowi] += bV[rowi];
            outV[rowi] *= rAUV[rowi] / volV[rowi];
        }
    );
}


template<typename SparsityType>
Vector<scalar> scaledInverseDiag(const Matrix<Vec3, SparsityType>& mtx, const Vector<scalar>& a)
{
    auto diag = Vector<scalar>(mtx.exec(), mtx.nRows());
    scaledInverseDiag(mtx, a, diag);
    return diag;
}

template<typename SparsityType>
void scaledInverseDiag(
    const Matrix<Vec3, SparsityType>& mtx, const Vector<scalar>& a, Vector<scalar>& out
)
{
    NF_ASSERT(mtx.nRows() == a.size(), "Dimension mismatch");
    NF_ASSERT(mtx.nRows() == out.size(), "Dimension mismatch");

    const auto [matrixV, sparsity] = mtx.view();
    auto [outV, aV] = views(out, a);

    parallelFor(
        mtx.exec(),
        {0, mtx.nRows()},
        NEON_LAMBDA(const localIdx rowi) {
            const auto rowSize = sparsity.rowSize(rowi);
            for (localIdx slot = 0; slot < rowSize; ++slot)
            {
                const auto idx = sparsity.linearIndex(rowi, slot);
                const auto col = sparsity.colIdxs[idx];
                if (col == decltype(sparsity)::invalidIndex()) break; // ELL padding
                if (rowi == col)
                {
                    outV[rowi] = aV[rowi] * inv(matrixV[idx][0]);
                    break;
                }
            }
        }
    );
}

template<typename SparsityType>
Vector<scalar> scaledInverseDiag(
    const Matrix<Vec3, SparsityType>& mtx, const FaceToMatrixAddress& mi, const Vector<scalar>& a
)
{
    auto diag = Vector<scalar>(mtx.exec(), mtx.nRows());
    scaledInverseDiag(mtx, mi, a, diag);
    return diag;
}

template<typename SparsityType>
void scaledInverseDiag(
    const Matrix<Vec3, SparsityType>& mtx,
    const FaceToMatrixAddress& mi,
    const Vector<scalar>& a,
    Vector<scalar>& out
)
{
    NF_ASSERT(mtx.nRows() == a.size(), "Dimension mismatch");
    NF_ASSERT(mtx.nRows() == out.size(), "Dimension mismatch");

    const auto matrixV = mtx.values().view();
    const auto ma = mi.view(mtx.sparsity()->view());
    auto [outV, aV] = views(out, a);

    parallelFor(
        mtx.exec(),
        {0, mtx.nRows()},
        NEON_LAMBDA(const localIdx rowi) { outV[rowi] = aV[rowi] / matrixV[ma.diagIdx(rowi)][0]; }
    );
}

template<typename SparsityType>
Vector<scalar> scaledInverseDiag(
    const Matrix<scalar, SparsityType>& mtx, const FaceToMatrixAddress& mi, const Vector<scalar>& a
)
{
    auto diag = Vector<scalar>(mtx.exec(), mtx.nRows());
    scaledInverseDiag(mtx, mi, a, diag);
    return diag;
}

template<typename SparsityType>
void scaledInverseDiag(
    const Matrix<scalar, SparsityType>& mtx,
    const FaceToMatrixAddress& mi,
    const Vector<scalar>& a,
    Vector<scalar>& out
)
{
    NF_ASSERT(mtx.nRows() == a.size(), "Dimension mismatch");
    NF_ASSERT(mtx.nRows() == out.size(), "Dimension mismatch");

    const auto matrixV = mtx.values().view();
    const auto ma = mi.view(mtx.sparsity()->view());
    auto [outV, aV] = views(out, a);

    parallelFor(
        mtx.exec(),
        {0, mtx.nRows()},
        NEON_LAMBDA(const localIdx rowi) {
            // scalar diagonal coefficient: no per-component selection needed
            outV[rowi] = aV[rowi] / matrixV[ma.diagIdx(rowi)];
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

// ELL instantiated standalone, not via NN_DECLARE_MATRIX: upper() is CSR-shaped and has
// no ELL overload.
template class Matrix<scalar, la::EllSparsityPattern<localIdx>>;
template class Matrix<Vec3, la::EllSparsityPattern<localIdx>>;

// Momentum-predictor utilities (scaledInverseDiag / scaledInvDiagNegLUx), format-generic over
// SparsityType -- CSR and ELL both instantiated, unlike upper()/NN_DECLARE_MATRIX above.
#define NN_DECLARE_MOMENTUM_UTILS(SPARSITYTYPE)                                                                                                              \
    template Vector<scalar>                                                                                                                                  \
    scaledInverseDiag(const Matrix<Vec3, SPARSITYTYPE>&, const Vector<scalar>&);                                                                             \
    template void                                                                                                                                            \
    scaledInverseDiag(const Matrix<Vec3, SPARSITYTYPE>&, const Vector<scalar>&, Vector<scalar>&);                                                            \
    template Vector<scalar>                                                                                                                                  \
    scaledInverseDiag(const Matrix<Vec3, SPARSITYTYPE>&, const FaceToMatrixAddress&, const Vector<scalar>&);                                                 \
    template void                                                                                                                                            \
    scaledInverseDiag(const Matrix<Vec3, SPARSITYTYPE>&, const FaceToMatrixAddress&, const Vector<scalar>&, Vector<scalar>&);                                \
    template Vector<scalar>                                                                                                                                  \
    scaledInverseDiag(const Matrix<scalar, SPARSITYTYPE>&, const FaceToMatrixAddress&, const Vector<scalar>&);                                               \
    template void                                                                                                                                            \
    scaledInverseDiag(const Matrix<scalar, SPARSITYTYPE>&, const FaceToMatrixAddress&, const Vector<scalar>&, Vector<scalar>&);                              \
    template void                                                                                                                                            \
    scaledInvDiagNegLUx(const Matrix<Vec3, SPARSITYTYPE>&, const Vector<Vec3>&, const Vector<Vec3>&, const Vector<scalar>&, Vector<scalar>&, Vector<Vec3>&); \
    template void                                                                                                                                            \
    scaledInvDiagNegLUx(const Matrix<scalar, SPARSITYTYPE>&, const Vector<Vec3>&, const Vector<Vec3>&, const Vector<scalar>&, Vector<scalar>&, Vector<Vec3>&)

NN_DECLARE_MOMENTUM_UTILS(la::CsrSparsityPattern<localIdx>);
NN_DECLARE_MOMENTUM_UTILS(la::EllSparsityPattern<localIdx>);

}
