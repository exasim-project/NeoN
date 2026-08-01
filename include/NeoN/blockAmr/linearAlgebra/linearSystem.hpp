// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <AMReX_MultiFab.H>

#include <cstddef>

#include "NeoN/blockAmr/linearAlgebra/faceCoeffMatrix.hpp"
#include "NeoN/core/executor/executor.hpp"

namespace blockamr::la
{

/* @class LinearSystem
 * @brief Pure data: a matrix and a right-hand side, non-owning -- both must outlive it,
 *        and the rhs a caller passes IS the one the solve reads. No BC, Geometry or
 *        discretisation knowledge of its own; the operator reads those off the matrix.
 */
class LinearSystem
{
public:

    LinearSystem(MFFaceCoeffs& matrix, amrex::MultiFab& rhs) : matrix_(&matrix), rhs_(&rhs) {}

    // ACCUMULATES: a system is zeroed once, then written by however many operators
    // contribute. A template, so the operator's SIGNATURE is the whole contract -- there is
    // no erasure and no virtual call between `+=` and the kernels.
    template<class Op>
    LinearSystem& operator+=(const Op& op)
    {
        op.assemble(*this);
        return *this;
    }

    // Coefficients AND rhs together: zeroing half a system is never meant.
    void zero()
    {
        matrix_->zero();
        rhs_->setVal(0.0);
    }

    // Plain accessors: nothing is derived from the coefficients on this side, so there is
    // nothing a write has to invalidate.
    MFFaceCoeffs& matrix() { return *matrix_; }

    const MFFaceCoeffs& matrix() const { return *matrix_; }

    amrex::MultiFab& rhs() { return *rhs_; }

    const amrex::MultiFab& rhs() const { return *rhs_; }

    // Rank-local, never boxArray().numPts(): see localCount in transfer.hpp.
    std::size_t localRows() const { return matrix_->localRows(); }

    const NeoN::Executor& executor() const { return matrix_->exec; }

private:

    MFFaceCoeffs* matrix_;
    amrex::MultiFab* rhs_;
};

} // namespace blockamr::la
