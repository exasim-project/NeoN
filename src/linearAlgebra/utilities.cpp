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
    // for a 3x3 matrix with 7 nnz 4
    // 0, 2, 5, 7  -> size = 4
    auto length = Vector<localIdx> {exec, 3 * (in.size() - 1)};
    fill(length, 0);
    auto lengthV = length.view();

    // compute the length of each row and stretch it out
    // [0, 2, 5, 7] -> [2, 2, 2, ..., 2,2,2 ]
    NeoN::parallelFor(
        exec,
        {0, in.size() - 1},
        NEON_LAMBDA(const localIdx i) {
            localIdx j = 3 * i;
            const auto val = inV[i + 1] - inV[i];
            lengthV[j + 0] = val;
            lengthV[j + 1] = val;
            lengthV[j + 2] = val;
        },
        "computeUnpackedRowOffs"
    );

    auto ret = Vector<localIdx> {exec, 3 * (in.size() - 1) + 1};
    fill(ret, 0);
    auto retV = ret.view();

    // [2, 2, 2] -> [0, 2, 4, 6, 8]
    NeoN::parallelScan(
        exec,
        {1, length.size() + 1},
        NEON_LAMBDA(const NeoN::localIdx i, NeoN::localIdx& update, const bool final) {
            update += lengthV[i - 1];
            if (final)
            {
                retV[i] = update;
            }
        }
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

template<typename MatrixType>
void computeResidual(
    const MatrixType& mtx, const Vector<scalar>& bV, const Vector<scalar>& xV, Vector<scalar>& resV
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
            scalar sum = 0.0;
            for (localIdx coli = rowStart; coli < rowEnd; coli++)
            {
                sum += coeffs[coli] * x[sparsity.colIdxs[coli]];
            }
            res[rowi] = sum - b[rowi];
        },
        "computeResidual"
    );
}

template void computeResidual<CSRMatrix<
    scalar,
    localIdx>>(const CSRMatrix<scalar, localIdx>&, const Vector<scalar>&, const Vector<scalar>&, Vector<scalar>&);

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
