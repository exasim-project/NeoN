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
 * @brief Pure data: a matrix and a right-hand side, non-owning -- both must outlive it,
 *        and the rhs a caller passes IS the one the solve reads. No BC, Geometry or
 *        discretisation knowledge. Sole friend of Coefficients and Operator::assemble.
 */
class LinearSystem
{
public:

    LinearSystem(Matrix& matrix, amrex::MultiFab& rhs) : matrix_(&matrix), rhs_(&rhs) {}

    // ACCUMULATES: a system is zeroed once, then written by however many operators
    // contribute. The operator sees only a Coefficients.
    LinearSystem& operator+=(const Operator& op)
    {
        // The one site the six fields are ORDERED, hence the only place a transposition could
        // hide. nonOwning: the rhs belongs to the caller, so the handle borrows it.
        op.assemble(Coefficients {
            matrix_->mesh(),
            matrix_->alpha(),
            matrix_->upper(),
            matrix_->lower(),
            CellFieldLevel {nonOwning(*rhs_)},
            matrix_->executor()
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
