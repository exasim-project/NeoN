// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <AMReX_MultiFab.H>

#include <cstddef>

#include "NeoN/blockAmr/linearAlgebra/coefficients.hpp"
#include "NeoN/blockAmr/linearAlgebra/matrix.hpp"
#include "NeoN/blockAmr/linearAlgebra/operator.hpp"
#include "NeoN/core/executor/executor.hpp"

namespace blockamr::la
{

/* @class LinearSystem
 * @brief Pure data: a matrix and a right-hand side, the two things a solve needs
 *        and the two things an operator contributes to.
 *
 * No BCs, no Geometry, no discretisation knowledge -- the operators folded all of
 * that in before it got here. What it adds over holding the pair loosely is
 * `operator+=`: A and b are written together, by one object, through one call.
 *
 * Non-owning, per the design: it holds `Matrix*` and `amrex::MultiFab*` and both
 * must outlive it. That is not a shortcut -- the rhs a caller passes IS the rhs
 * the solve reads, so an operator's contribution to it is visible to the caller
 * without a copy-back step.
 *
 * This class is the sole friend of `Coefficients` (coefficients.hpp) and the sole
 * friend of `Operator::assemble` (operator.hpp). `operator+=` below is where both
 * friendships are finally used, and it is the only mutating entry point on the
 * linear-algebra side.
 */
class LinearSystem
{
public:

    LinearSystem(Matrix& matrix, amrex::MultiFab& rhs) : matrix_(&matrix), rhs_(&rhs) {}

    // The accumulation. Builds the one Coefficients the operator will ever see --
    // the matrix's coefficient handles plus the rhs, which lives here rather than
    // on the Matrix -- and hands it over. The operator receives that and nothing
    // else: not this class, not the Matrix, not the format.
    //
    // ACCUMULATES. Operators add to what is already there, so a system is zeroed
    // once (zero(), or at construction by the format) and then written by however
    // many operators contribute to it.
    LinearSystem& operator+=(const Operator& op)
    {
        op.assemble(Coefficients {matrix_->coefficients(), CellView {rhs_}, matrix_->executor()});
        return *this;
    }

    // Coefficients AND rhs, together: they are one system, and zeroing half of it
    // is never what a caller means.
    void zero()
    {
        matrix_->zero();
        rhs_->setVal(0.0);
    }

    const Matrix& matrix() const { return *matrix_; }

    const amrex::MultiFab& rhs() const { return *rhs_; }

    // From the MATRIX, which is the only object that knows how many rows this rank
    // owns. Never boxArray().numPts(), which counts every rank's cells and differs
    // under MPI (faceCoeffMatrix.hpp).
    std::size_t localRows() const { return matrix_->localRows(); }

    const NeoN::Executor& executor() const { return matrix_->executor(); }

private:

    Matrix* matrix_;
    amrex::MultiFab* rhs_;
};

} // namespace blockamr::la
