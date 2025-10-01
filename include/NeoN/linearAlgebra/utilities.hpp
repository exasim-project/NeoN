// SPDX-FileCopyrightText: 2024 - 2025 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include "NeoN/core/primitives/scalar.hpp"
#include "NeoN/core/primitives/vec3.hpp"
#include "NeoN/core/vector/vector.hpp"
#include "NeoN/linearAlgebra/CSRMatrix.hpp"


namespace NeoN::la
{

/* @brief given a vector a [0,1,0,1,2,1,2] this returns [0,1,2,3,4,5,0,1,2,...]
 *
 * @param[in] in the vector to duplicate
 * @return A vector containing duplicated entries
 */
Vector<localIdx> duplicateColIdx(
    const Vector<localIdx>& in,
    const Vector<localIdx>& newRowOffs,
    const Vector<localIdx>& oldRowOffs
);

/* @brief given a vector of rowOffs [0,3,6] this returns [0,3,6,9,12,15,18,21]
 *
 * @param[in] in the vector to duplicate
 * @return A vector containing duplicated entries
 */
Vector<localIdx> stretchRowPtrs(const Vector<localIdx>& in);

/* @brief given a vector [{1,2,3},{4,5,6},{7,8,9}] this returns [1,2,3,4,5,6,7,8,9]
 *
 * @param[in] in the vector to duplicate
 * @return A vector containing duplicated entries
 */
Vector<scalar> flatten(const Vector<Vec3>& in);

/* @brief given a vector [{1,2,3},{4,5,6},{7,8,9}] this returns [1,2,3,4,5,6,7,8,9]
 *
 * @param[in] in the vector to duplicate
 * @return A vector containing duplicated entries
 */
void pack(const Vector<scalar>& in, Vector<Vec3>& out);
void pack(const Vector<scalar>& in, Vector<scalar>& out);

/* @brief given a vector [{1,2,3},{4,5,6},{7,8,9}] this returns [1,2,3,4,5,6,7,8,9]
 *
 * @param[in] in the vector to duplicate
 * @return A vector containing duplicated entries
 */
Vector<scalar> unpack(const Vector<Vec3>& in);

/* @brief given a vector [{1,2,3},{4,5,6},{7,8,9}]
 * this returns [1,4,7,2,5,8,3,6,9]
 *
 * @param[in] in the vector to duplicate
 * @return A vector containing duplicated entries
 */
Vector<scalar> unpackMtxValues(
    const Vector<Vec3>& in, const Vector<localIdx>& rowOffs, const Vector<localIdx>& newRowOffs
);


/* @brief given a linear system consisting of A, b and x the operator computes the residual vector *
 * Ax-b
 *
 * @param[in] mtx, the corresponding matrix
 * @param[in] b, rhs vector b
 * @param[in] x, initial guess vector x
 * @param[out]
 */
void computeResidual(
    const CSRMatrix<scalar, localIdx>& mtx,
    const Vector<scalar>& b,
    const Vector<scalar>& x,
    Vector<scalar>& res
);

}
