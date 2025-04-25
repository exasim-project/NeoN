// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024-2025 NeoN authors

#pragma once

#include "NeoN/core/primitives/scalar.hpp"
#include "NeoN/core/vector.hpp"
#include "NeoN/linearAlgebra/CSRMatrix.hpp"


namespace NeoN::la
{

/* @brief given a linear system consisting of A, b and x the operator computes Ax-b
 *
 */
void computeResidual(
    const CSRMatrix<scalar, localIdx>& mtx,
    const Vector<scalar> b,
    const Vector<scalar>& x,
    Vector<scalar>& res
);

}
