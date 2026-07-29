// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <AMReX_MultiFab.H>

#include <cstddef>

#include "NeoN/blockAmr/core/fieldLevel.hpp"
#include "NeoN/blockAmr/linearAlgebra/coefficients.hpp"
#include "NeoN/blockAmr/linearAlgebra/matrix.hpp"
#include "NeoN/blockAmr/linearAlgebra/operator.hpp"
#include "NeoN/core/executor/executor.hpp"

namespace blockamr::la
{

/* @class LinearSystem
 * @brief Pure data: a matrix and a right-hand side. No BCs, Geometry or
 *        discretisation knowledge -- the operators fold that in beforehand.
 *
 * INVARIANT: non-owning. It holds `Matrix*` and `amrex::MultiFab*`, both of which
 * must outlive it; the rhs a caller passes IS the rhs the solve reads, so an
 * operator's contribution needs no copy-back.
 *
 * Sole friend of `Coefficients` (coefficients.hpp) and of `Operator::assemble`
 * (operator.hpp); `operator+=` is the only mutating entry point on the
 * linear-algebra side.
 */
class LinearSystem
{
public:

    LinearSystem(Matrix& matrix, amrex::MultiFab& rhs) : matrix_(&matrix), rhs_(&rhs) {}

    // ACCUMULATES: operators add to what is already there, so a system is zeroed
    // once (zero(), or at construction by the format) and then written by however
    // many operators contribute. The operator sees only a Coefficients -- not this
    // class, not the Matrix, not the format.
    LinearSystem& operator+=(const Operator& op)
    {
        // nonOwning: the rhs belongs to the caller, so the handle borrows it.
        op.assemble(Coefficients {
            matrix_->coefficients(), CellFieldLevel {nonOwning(*rhs_)}, matrix_->executor()
        });
        return *this;
    }

    // Coefficients AND rhs together: zeroing half a system is never meant.
    void zero()
    {
        matrix_->zero();
        rhs_->setVal(0.0);
    }

    const Matrix& matrix() const { return *matrix_; }

    const amrex::MultiFab& rhs() const { return *rhs_; }

    // Rank-local, never boxArray().numPts(): see localCount in transfer.hpp.
    std::size_t localRows() const { return matrix_->localRows(); }

    const NeoN::Executor& executor() const { return matrix_->executor(); }

private:

    Matrix* matrix_;
    amrex::MultiFab* rhs_;
};

} // namespace blockamr::la
