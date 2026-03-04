// SPDX-FileCopyrightText: 2024 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <span>
#include <vector>

#include <ginkgo/ginkgo.hpp>

#include "NeoN/core/primitives/label.hpp"
#include "NeoN/core/primitives/scalar.hpp"
#include "NeoN/core/primitives/vec3.hpp"
#include "NeoN/core/vector/vector.hpp"
#include "NeoN/linearAlgebra/matrix.hpp"


namespace NeoN::la
{

/**
 * @brief Convert a local-index CSR matrix (with ghost columns) to a Ginkgo matrix_data with global
 * indices.
 *
 * For each owned row i:
 *   globalRow = globalCellIds[i]
 *   for each entry j in the row:
 *     if localCol < nLocalCells: globalCol = globalCellIds[localCol]
 *     else:                      globalCol = ghostCellGlobalIds[localCol - nLocalCells]
 *
 * @param matrix       Host-accessible CSR matrix (call copyToHost() first if needed).
 * @param globalCellIds   Mapping from local cell index to global index (size nLocalCells).
 * @param ghostCellGlobalIds  Global IDs of ghost cells (size nGhostCells).
 * @param nLocalCells  Number of owned (local) cells == number of matrix rows.
 * @return gko::matrix_data with global row/column indices.
 */
inline gko::matrix_data<scalar, label> toGlobalMatrixData(
    const CSRMatrix<scalar, localIdx>& matrix,
    const std::vector<label>& globalCellIds,
    const std::vector<label>& ghostCellGlobalIds,
    localIdx nLocalCells
)
{
    const label nGlobal = static_cast<label>(globalCellIds.size() + ghostCellGlobalIds.size());
    gko::matrix_data<scalar, label> matData {
        gko::dim<2> {static_cast<std::size_t>(nGlobal), static_cast<std::size_t>(nGlobal)}
    };

    auto hostMatrix = matrix.copyToHost();
    const auto* colIdxs = hostMatrix.colIdxs().data();
    const auto* rowOffs = hostMatrix.rowOffs().data();
    const auto* values = hostMatrix.values().data();

    for (localIdx row = 0; row < nLocalCells; ++row)
    {
        label globalRow = globalCellIds[static_cast<std::size_t>(row)];
        for (localIdx j = rowOffs[row]; j < rowOffs[row + 1]; ++j)
        {
            localIdx localCol = colIdxs[j];
            label globalCol =
                (localCol < nLocalCells)
                    ? globalCellIds[static_cast<std::size_t>(localCol)]
                    : ghostCellGlobalIds[static_cast<std::size_t>(localCol - nLocalCells)];
            matData.nonzeros.emplace_back(globalRow, globalCol, values[j]);
        }
    }
    return matData;
}

/* @brief given a vector of column indices for vector matrices it creates the unpacked scalar
 * version.
 * @param[in] in the vector of column indices for a packed Csr<Vec3> matrix
 * @param[in] unpackedRowOffs corresponding rowOffsets of the unpacked matrix
 * @param[in] packedRowOffs corresponding rowOffsets of the packed matrix
 * @return A vector containing new unpacked column indices
 * @details Example input [0,1,0,1,2,1,2] this function returns [0,3,1,4,2,5,0,3,6,1,4,7 ...] see
 * example
 *
 * Example:
 *   0 1 2       0 1 2 3 4 5 6 7 9     column index
 *              [x . . x . . . . . ]
 *  [x x . ]    [. x . . x . . . . ]
 *  [x x x ] -> [. . x . . x . . . ]
 *  [. x x ]    [x . . x . . x . . ]
 *              [. x . . x . . x . ]
 *              [. . x . . x . . x ]
 *              [. . . x . . x . . ]
 *              [. . . . x . . x . ]
 *              [. . . . . x . . x ]
 *
 *  packed      unpacked
 *  sparsity    sparsity
 */
Vector<localIdx> unpackColIdx(
    const Vector<localIdx>& in,
    const Vector<localIdx>& unpackedRowOffs,
    const Vector<localIdx>& packedRowOffs
);

/* @brief computed unpacked rowOffs from packed rowOffs
 * @details given a sparsity pattern every row with Vec3 entries is unpacked
 * by copying its y and z components after the initial row. For
 * example [{x,y,z}] will result in
 * [x . .]
 * [. y .]
 * [. . z]
 * Thus, each row with given length will result 2 new entries of the same length
 * in the unpacked vector. E.g. a vector of packed rowOffs [0,2,5,7] returns
 * [0,2,4,6,9,12,15,17,19,21] where the last entries is the total number of rows
 *
 * @param[in] in the vector rowPtrs to unpack
 * @return A vector containing unpacked rowPtrs
 */
Vector<localIdx> unpackRowOffs(const Vector<localIdx>& in);

/* @brief given a vector of Vec3 (packed) this returns vector of consecutive scalars (unpacked)
 *
 * E.g. given a vector [{1,2,3},{4,5,6},{7,8,9}] this returns [1,2,3,4,5,6,7,8,9]
 *
 * @param[in] in the vector to unpack
 * @return A vector containing duplicated entries
 */
Vector<scalar> unpackVecValues(const Vector<Vec3>& in);

/* @brief given a vector [1,2,3,4,5,6,7,8,9] this packs it into [{1,2,3},{4,5,6},{7,8,9}]
 *
 * @param[in] in the vector to duplicate
 * @param[out] out the vector to duplicate
 */
void packVecValues(const Vector<scalar>& in, Vector<Vec3>& out);

/* @brief given a vector of packed matrix values this returns unpacked matrix values
 * @details
 *
 * Given an input row [{1,2,3},{4,5,6},{7,8,9}]
 * this returns [1,4,7,2,5,8,3,6,9]
 *
 * @param[in] in vector of packed matrix values
 * @param[in] rowOffs the rowOffs of the packed matrix
 * @param[in] newRowOffs the rowOffs of the unpacked matrix
 * @return A vector of the unpacked matrix values
 */
Vector<scalar> unpackMtxValues(
    const Vector<Vec3>& in, const Vector<localIdx>& rowOffs, const Vector<localIdx>& newRowOffs
);


/* @brief given a linear system consisting of A, b and x the operator computes the residual vector
 * Ax-b
 *
 * @param[in] mtx, the corresponding matrix
 * @param[in] b, rhs vector b
 * @param[in] x, initial guess vector x
 * @param[out]
 */
template<typename MatrixType>
void computeResidual(
    const MatrixType& mtx, const Vector<scalar>& b, const Vector<scalar>& x, Vector<scalar>& res
);

}
