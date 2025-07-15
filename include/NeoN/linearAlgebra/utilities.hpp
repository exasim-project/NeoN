// SPDX-FileCopyrightText: 2024 - 2025 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include "NeoN/core/primitives/scalar.hpp"
#include "NeoN/core/vector/vector.hpp"
#include "NeoN/linearAlgebra/CSRMatrix.hpp"


namespace NeoN::la
{

/* @brief given a linear system consisting of A, b and x the operator computes the residual vector
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
