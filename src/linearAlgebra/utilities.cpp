// SPDX-FileCopyrightText: 2024 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "NeoN/core/macros.hpp"
#include "NeoN/core/parallelAlgorithms.hpp"
#include "NeoN/core/containerFreeFunctions.hpp"
#include "NeoN/linearAlgebra/utilities.hpp"


namespace NeoN::la
{

Vector<localIdx> unpackColIdx(
    const Vector<localIdx>& in,
    const Vector<localIdx>& unpackedRowOffs,
    const Vector<localIdx>& packedRowOffs
)
{
    const auto exec = in.exec();
    const auto inV = in.view();
    auto out = Vector<localIdx> {exec, 3 * in.size()};
    auto outV = out.view();
    auto rowV = unpackedRowOffs.view();
    auto oldRowV = packedRowOffs.view();

    NeoN::parallelFor(
        exec,
        {0, unpackedRowOffs.size() - 1},
        NEON_LAMBDA(const localIdx i) {
            const auto j {rowV[i]};        // new row start
            const auto l {oldRowV[i / 3]}; // original row start
            const auto length {rowV[i + 1] - rowV[i]};
            const auto offs = i % 3;
            // iterate all entries of the row
            // every column is shifted by a factor of 3
            // plus an offset based on the dimension 0,1,2
            for (auto k = 0; k < length; k++)
            {
                outV[j + k] = (3 * inV[l + k]) + offs;
            }
        },
        "computeUnpackedColIdx"
    );

    return out;
}

Vector<scalar> unpackVecValues(const Vector<Vec3>& in)
{
    const auto exec = in.exec();
    const auto inV = in.view();
    auto out = Vector<scalar> {exec, 3 * in.size()};
    auto outV = out.view();

    NeoN::parallelFor(
        exec,
        {0, in.size()},
        NEON_LAMBDA(const localIdx i) {
            localIdx j = 3 * i;
            outV[j + 0] = inV[i][0];
            outV[j + 1] = inV[i][1];
            outV[j + 2] = inV[i][2];
        },
        "computeUnpackedVecValues"
    );

    return out;
}

Vector<scalar> unpackMtxValues(
    const Vector<Vec3>& in, const Vector<localIdx>& rowOffs, const Vector<localIdx>& newRowOffs
)
{
    const auto exec = in.exec();
    const auto inV = in.view();
    auto out = Vector<scalar> {exec, 3 * in.size()};
    auto outV = out.view();
    auto rowV = rowOffs.view();
    auto newRowV = newRowOffs.view();

    NeoN::parallelFor(
        exec,
        {0, rowOffs.size() - 1},
        NEON_LAMBDA(const localIdx i) {
            const auto length {rowV[i + 1] - rowV[i]};
            for (auto k = 0; k < length; k++)
            {
                outV[newRowV[3 * i + 0] + k] = inV[rowV[i] + k][0];
                outV[newRowV[3 * i + 1] + k] = inV[rowV[i] + k][1];
                outV[newRowV[3 * i + 2] + k] = inV[rowV[i] + k][2];
            }
        },
        "computeUnpackedMtxValues"
    );

    return out;
}

Vector<localIdx> unpackRowOffs(const Vector<localIdx>& in)
{
    const auto exec = in.exec();
    const auto inV = in.view();
    // for a 3x3 matrix with 7 nnz, input is [0, 2, 5, 7] (4 entries = nRows+1)
    const localIdx nOldRows = static_cast<localIdx>(in.size() - 1);
    auto ret = Vector<localIdx>(exec, 3 * nOldRows + 1);
    auto retV = ret.view();

    // Closed-form expansion: for original row i with offset off and length len,
    // the 3 expanded rows start at: 3*off, 3*off+len, 3*off+2*len.
    // The sentinel (last element) is 3*inV[nOldRows].
    // Example: [0,2,5,7] -> [0,2,4, 6,9,12, 15,17,19, 21]
    NeoN::parallelFor(
        exec,
        {0, nOldRows + 1},
        NEON_LAMBDA(const localIdx i) {
            retV[3 * i] = 3 * inV[i];
            if (i < nOldRows)
            {
                const localIdx len = inV[i + 1] - inV[i];
                retV[3 * i + 1] = 3 * inV[i] + len;
                retV[3 * i + 2] = 3 * inV[i] + 2 * len;
            }
        },
        "computeUnpackedRowOffs"
    );
    return ret;
}


void packVecValues(const Vector<scalar>& in, Vector<Vec3>& out)
{
    const auto exec = in.exec();
    const auto inV = in.view();
    auto outV = out.view();

    NeoN::parallelFor(
        exec,
        {0, out.size()},
        NEON_LAMBDA(const localIdx i) {
            localIdx j = 3 * i;
            outV[i][0] = inV[j + 0];
            outV[i][1] = inV[j + 1];
            outV[i][2] = inV[j + 2];
        },
        "computePackedVecValues"
    );
}

template<typename MatrixType, typename ValueType>
void computeResidual(
    const MatrixType& mtx,
    const Vector<ValueType>& bV,
    const Vector<ValueType>& xV,
    Vector<ValueType>& resV
)
{
    auto [res, b, x] = views(resV, bV, xV);
    const auto [coeffs, sparsity] = mtx.view();

    NeoN::parallelFor(
        resV.exec(),
        {0, resV.size()},
        NEON_LAMBDA(const localIdx rowi) {
            auto rowStart = sparsity.rowOffs[rowi];
            auto rowEnd = sparsity.rowOffs[rowi + 1];
            // ValueType sum: scalar coeffs * Vec3 x broadcasts to each component for
            // the segregated vector-solve form (scalar matrix, Vec3 rhs)
            ValueType sum = zero<ValueType>();
            for (localIdx coli = rowStart; coli < rowEnd; coli++)
            {
                sum += coeffs[coli] * x[sparsity.colIdxs[coli]];
            }
            res[rowi] = sum - b[rowi];
        },
        "computeResidual"
    );
}

template void computeResidual<
    CSRMatrix<scalar, localIdx>,
    scalar>(const CSRMatrix<scalar, localIdx>&, const Vector<scalar>&, const Vector<scalar>&, Vector<scalar>&);

template void computeResidual<
    CSRMatrix<scalar, localIdx>,
    Vec3>(const CSRMatrix<scalar, localIdx>&, const Vector<Vec3>&, const Vector<Vec3>&, Vector<Vec3>&);

template<typename IndexType>
Vector<IndexType> rowsToRowOffs(const Vector<IndexType>& rows)
{
    auto rowsHost = rows.copyToHost();
    const auto rowsV = rowsHost.view();
    const auto nnz = rowsV.size();

    if (nnz == 0)
    {
        return Vector<IndexType>(SerialExecutor {}, 1, IndexType(0)).copyToExecutor(rows.exec());
    }

    // TODO can this be realized without copying to host?
    IndexType maxRow = 0;
    for (localIdx i = 0; i < nnz; i++)
    {
        if (rowsV[i] > maxRow) maxRow = rowsV[i];
    }

    const IndexType nRows = maxRow + 1;
    Vector<IndexType> rowOffs(SerialExecutor {}, nRows + 1, IndexType(0));
    auto rowOffsV = rowOffs.view();

    for (localIdx i = 0; i < nnz; i++)
    {
        rowOffsV[rowsV[i] + 1]++;
    }

    for (IndexType r = 0; r < nRows; r++)
    {
        rowOffsV[r + 1] += rowOffsV[r];
    }

    return rowOffs.copyToExecutor(rows.exec());
}

#define NN_INSTANTIATE_ROWS_TO_ROW_OFFS(TYPENAME)                                                  \
    template Vector<TYPENAME> rowsToRowOffs<TYPENAME>(const Vector<TYPENAME>&)

NN_FOR_ALL_INTEGER_TYPES(NN_INSTANTIATE_ROWS_TO_ROW_OFFS);

}
