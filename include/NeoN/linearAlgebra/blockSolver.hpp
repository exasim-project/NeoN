// SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include "NeoN/core/dictionary.hpp"
#include "NeoN/core/executor/executor.hpp"
#include "NeoN/linearAlgebra/blockMatrix.hpp"
#include "NeoN/linearAlgebra/blockVector.hpp"
#include "NeoN/linearAlgebra/solver.hpp"

namespace NeoN::la
{

/**
 * @class BlockSolver
 * @brief Solver for block-coupled linear systems that calls Ginkgo directly with zero-copy views.
 *
 * Unlike the free function la::solve() which deep-copies through toCSR() and LinearSystem,
 * BlockSolver wraps BlockMatrix/BlockVector data directly into Ginkgo arrays and matrices
 * without any intermediate copies.
 */
class BlockSolver
{

public:

    /**
     * @brief Construct from executor and solver configuration dictionary.
     * @param exec The executor (serial, CPU, or GPU).
     * @param solverDict Dictionary with solver backend, type, and criteria.
     */
    BlockSolver(const Executor& exec, const Dictionary& solverDict);

    /**
     * @brief Solve the block linear system A * solution = rhs with zero-copy Ginkgo views.
     *
     * The solution vector serves as the initial guess on input.
     *
     * @param matrix The block-structured sparse matrix (values already in monolithic CSR order).
     * @param rhs The block right-hand side vector.
     * @param solution The block solution vector (input: initial guess, output: solution).
     * @return SolverStats Convergence statistics from the solve.
     */
    SolverStats
    solve(const BlockMatrix& matrix, const BlockVector& rhs, BlockVector& solution) const;

private:

    Executor exec_;
    Dictionary solverDict_;
};

} // namespace NeoN::la
