// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_MLMG.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Vector.H>

#include <ginkgo/ginkgo.hpp>

#include <cstddef>
#include <memory>
#include <vector>

#include "NeoN/blockAmr/linearAlgebra/matrixFree/linOpBase.hpp"
#include "NeoN/blockAmr/core/types.hpp"

namespace blockamr::la
{

// Matrix-free SPD operator: x = sign*(L_inhom(b) - c0), with MLMG::apply as
// the mat-vec and c0 = L_inhom(0) the affine BC offset recorded once at
// construction.
class AmrexOp : public AmrexLinOpBase<AmrexOp>
{
public:

    // Required by the polymorphic-object machinery (create_default / clear).
    explicit AmrexOp(std::shared_ptr<const gko::Executor> exec);

    AmrexOp(
        std::shared_ptr<const gko::Executor> exec,
        MLMG* mlmg,
        const amrex::BoxArray& ba,
        const amrex::DistributionMapping& dm,
        gko::size_type n,
        double sign
    );

protected:

    // Keeps the base's advanced apply_impl(alpha, b, beta, x) from being hidden.
    using AmrexLinOpBase<AmrexOp>::apply_impl;

    void apply_impl(const gko::LinOp* b, gko::LinOp* x) const override;

private:

    MLMG* mlmg_ = nullptr;
    double sign_ = -1.0;
    std::shared_ptr<amrex::MultiFab> in_;
    std::shared_ptr<amrex::MultiFab> out_;
    std::shared_ptr<amrex::MultiFab> c0_;
};

// Multi-level (composite AMR) AmrexOp: the Ginkgo vector concatenates all levels'
// valid cells (coarsest first, in the gather/scatter per-box order) and the mat-vec
// is the COMPOSITE multi-level MLMG::apply — fine coarse/fine ghosts interpolated
// from the coarse `in`, the coarse residual refluxed at the interface (cancelling
// any dependence on covered coarse cells), the covered coarse output overwritten by
// average_down of the fine one. Hence, on the concatenated vector: covered coarse
// columns are ZERO (index-1 singular, nullspace = covered-cell perturbations,
// disjoint from the range), so a consistent rhs (covered coarse rhs = average_down
// of the fine rhs, enforced by the caller) is solvable and the covered solution
// entries are fixed by a final average_down; and the operator is NOT symmetric (the
// c/f interpolation is not the adjoint of the reflux), so bicgstab/gmres are the
// safe solvers (CG may still work in practice — measured by the caller/tests).
// Affine offset c0 = L_inhom(0) recorded per level, as in AmrexOp.
class CompositeAmrexOp : public AmrexLinOpBase<CompositeAmrexOp>
{
public:

    explicit CompositeAmrexOp(std::shared_ptr<const gko::Executor> exec);

    CompositeAmrexOp(
        std::shared_ptr<const gko::Executor> exec,
        MLMG* mlmg,
        const std::vector<amrex::BoxArray>& bas,
        const std::vector<amrex::DistributionMapping>& dms,
        gko::size_type n,
        double sign
    );

protected:

    // Keeps the base's advanced apply_impl(alpha, b, beta, x) from being hidden.
    using AmrexLinOpBase<CompositeAmrexOp>::apply_impl;

    void apply_impl(const gko::LinOp* b, gko::LinOp* x) const override;

private:

    static amrex::Vector<amrex::MultiFab*>
    ptrs(const std::vector<std::shared_ptr<amrex::MultiFab>>& v);

    MLMG* mlmg_ = nullptr;
    double sign_ = 1.0;
    std::vector<std::shared_ptr<amrex::MultiFab>> in_;
    std::vector<std::shared_ptr<amrex::MultiFab>> out_;
    std::vector<std::shared_ptr<amrex::MultiFab>> c0_;
    std::vector<long> off_;
};

// Multigrid preconditioner: z = M^{-1} r from a FIXED small number of MLMG V-cycles
// (setFixedIter, which ignores the tolerances passed to solve()) on a
// caller-supplied equivalent operator, so the Krylov iteration count stays ~flat in
// N while the outer mat-vec stays matrix-free. A V-cycle with (red-black)
// Gauss-Seidel smoothing is only approximately symmetric — classic CG tolerates it
// here (measured), bicgstab/gmres are the fallback if it degrades.
class MlmgPrecond : public AmrexLinOpBase<MlmgPrecond>
{
public:

    explicit MlmgPrecond(std::shared_ptr<const gko::Executor> exec);

    MlmgPrecond(
        std::shared_ptr<const gko::Executor> exec,
        MLMG* mlmg,
        const amrex::BoxArray& ba,
        const amrex::DistributionMapping& dm,
        gko::size_type n,
        int n_cycles
    );

protected:

    // Keeps the base's advanced apply_impl(alpha, b, beta, x) from being hidden.
    using AmrexLinOpBase<MlmgPrecond>::apply_impl;

    void apply_impl(const gko::LinOp* b, gko::LinOp* x) const override;

private:

    MLMG* mlmg_ = nullptr;
    std::shared_ptr<amrex::MultiFab> in_;
    std::shared_ptr<amrex::MultiFab> out_;
};

} // namespace blockamr::la
