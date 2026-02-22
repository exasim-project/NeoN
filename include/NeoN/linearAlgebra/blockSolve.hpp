// SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include "NeoN/core/dictionary.hpp"
#include "NeoN/linearAlgebra/blockMatrix.hpp"
#include "NeoN/linearAlgebra/blockVector.hpp"
#include "NeoN/linearAlgebra/solver.hpp"

namespace NeoN::la
{

/**
 * @brief Solve a block linear system by flattening to monolithic CSR and delegating to Solver.
 *
 * Boundary conditions are assumed to be already incorporated into the matrix and RHS.
 * The solution BlockVector serves as the initial guess (typically zero for first solve).
 *
 * @param matrix The block-structured sparse matrix.
 * @param rhs The block right-hand side vector.
 * @param solution The block solution vector (input: initial guess, output: solution).
 * @param solverDict Dictionary with solver configuration (solver backend, type, criteria).
 * @return SolverStats Convergence statistics from the solve.
 */
SolverStats solve(
    const BlockMatrix& matrix,
    const BlockVector& rhs,
    BlockVector& solution,
    const Dictionary& solverDict
);

} // namespace NeoN::la
