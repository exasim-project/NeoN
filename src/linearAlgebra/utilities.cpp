// SPDX-FileCopyrightText: 2024 - 2025 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "NeoN/core/parallelAlgorithms.hpp"
#include "NeoN/core/containerFreeFunctions.hpp"
#include "NeoN/linearAlgebra/utilities.hpp"


namespace NeoN::la
{

Vector<localIdx> convertColIdx(
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
        KOKKOS_LAMBDA(const localIdx i) {
            auto j {rowV[i]};        // new row start
            auto l {oldRowV[i / 3]}; // original row start
            auto length {rowV[i + 1] - rowV[i]};
            auto offs = i % 3;
            // iterate all entries of the row
            // every column is shifted by a factor of 3
            // plus an offset based on the dimension 0,1,2
            for (auto k = 0; k < length; k++)
            {
                outV[j + k] = (3 * inV[l + k]) + offs;
            }
        }
    );

    return out;
}

Vector<scalar> unpack(const Vector<Vec3>& in)
{
    const auto exec = in.exec();
    const auto inV = in.view();
    auto out = Vector<scalar> {exec, 3 * in.size()};
    auto outV = out.view();

    NeoN::parallelFor(
        exec,
        {0, in.size()},
        KOKKOS_LAMBDA(const localIdx i) {
            localIdx j = 3 * i;
            outV[j + 0] = inV[i][0];
            outV[j + 1] = inV[i][1];
            outV[j + 2] = inV[i][2];
        }
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
        KOKKOS_LAMBDA(const localIdx i) {
            auto length {rowV[i + 1] - rowV[i]};
            for (auto k = 0; k < length; k++)
            {
                outV[newRowV[3 * i + 0] + k] = inV[rowV[i] + k][0];
                outV[newRowV[3 * i + 1] + k] = inV[rowV[i] + k][1];
                outV[newRowV[3 * i + 2] + k] = inV[rowV[i] + k][2];
            }
        }
    );

    return out;
}

Vector<localIdx> unpackRowPtrs(const Vector<localIdx>& in)
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
        KOKKOS_LAMBDA(const localIdx i) {
            localIdx j = 3 * i;
            auto val = inV[i + 1] - inV[i];
            lengthV[j + 0] = val;
            lengthV[j + 1] = val;
            lengthV[j + 2] = val;
        }
    );

    auto ret = Vector<localIdx> {exec, 3 * (in.size() - 1) + 1};
    fill(ret, 0);
    auto retV = ret.view();

    // [2, 2, 2] -> [0, 2, 4, 6, 8]
    NeoN::parallelScan(
        exec,
        {1, length.size() + 1},
        KOKKOS_LAMBDA(const NeoN::localIdx i, NeoN::localIdx& update, const bool final) {
            update += lengthV[i - 1];
            if (final)
            {
                retV[i] = update;
            }
        }
    );
    return ret;
}


void pack(const Vector<scalar>& in, Vector<Vec3>& out)
{
    const auto exec = in.exec();
    const auto inV = in.view();
    auto outV = out.view();

    NeoN::parallelFor(
        exec,
        {0, out.size()},
        KOKKOS_LAMBDA(const localIdx i) {
            localIdx j = 3 * i;
            outV[i][0] = inV[j + 0];
            outV[i][1] = inV[j + 1];
            outV[i][2] = inV[j + 2];
        }
    );
}

void computeResidual(
    const CSRMatrix<scalar, localIdx>& mtx,
    const Vector<scalar>& bV,
    const Vector<scalar>& xV,
    Vector<scalar>& resV
)
{
    auto [res, b, x] = views(resV, bV, xV);
    const auto [coeffs, colIdxs, rowOffs] = mtx.view();

    NeoN::parallelFor(
        resV.exec(),
        {0, resV.size()},
        KOKKOS_LAMBDA(const localIdx rowi) {
            auto rowStart = rowOffs[rowi];
            auto rowEnd = rowOffs[rowi + 1];
            scalar sum = 0.0;
            for (localIdx coli = rowStart; coli < rowEnd; coli++)
            {
                sum += coeffs[coli] * x[colIdxs[coli]];
            }
            res[rowi] = sum - b[rowi];
        }
    );
}

}
