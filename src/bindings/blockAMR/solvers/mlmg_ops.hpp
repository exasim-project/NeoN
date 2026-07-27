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

#include "../../../blockAmrSolvers/common/linop_base.hpp"
#include "../../../blockAmrSolvers/common/types.hpp"

namespace blockamr::solvers
{

// Matrix-free SPD operator: x = sign*(L_inhom(b) - c0), with MLMG::apply as
// the mat-vec and c0 = L_inhom(0) the affine BC offset recorded once at
// construction.
class AmrexOp : public AmrexLinOpBase<AmrexOp>
{
public:

    // Exec-only constructor required by the polymorphic-object machinery
    // (create_default / clear).
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

    // Keeps the base's advanced apply_impl(alpha, b, beta, x) visible in this
    // scope (the declaration below would otherwise hide it).
    using AmrexLinOpBase<AmrexOp>::apply_impl;

    void apply_impl(const gko::LinOp* b, gko::LinOp* x) const override;

private:

    MLMG* mlmg_ = nullptr;
    double sign_ = -1.0;
    std::shared_ptr<amrex::MultiFab> in_;
    std::shared_ptr<amrex::MultiFab> out_;
    std::shared_ptr<amrex::MultiFab> c0_;
};

// Multi-level (composite AMR) generalisation of AmrexOp: the Ginkgo vector is
// the concatenation of all levels' valid cells (coarsest first, each level in
// the gather/scatter per-box flattening order) and the mat-vec is the
// multi-level MLMG::apply — the COMPOSITE operator: per level
// out[l] = L(in) with the fine level's coarse/fine boundary interpolated from
// the coarse `in`, the coarse residual refluxed at the coarse/fine interface
// (which cancels any dependence on coarse cells covered by the fine patch),
// and the covered coarse output overwritten by average_down of the fine
// output. Consequences for the linear system on the full concatenated vector:
//   - columns belonging to covered coarse cells are ZERO (index-1 singular;
//     nullspace = covered-cell perturbations, disjoint from the range), so a
//     consistent rhs (covered coarse rhs = average_down of the fine rhs —
//     enforced by the caller) is solvable and the covered solution entries
//     are fixed afterwards by a final average_down;
//   - the composite operator is NOT symmetric (the c/f ghost interpolation is
//     not the adjoint of the reflux), so bicgstab/gmres are the safe solvers
//     (CG may still work in practice — measured by the caller/tests).
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

    // Keeps the base's advanced apply_impl(alpha, b, beta, x) visible in this
    // scope (the declaration below would otherwise hide it).
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

// Multigrid preconditioner: z = M^{-1} r approximated by a FIXED small number
// of MLMG V-cycles (setFixedIter) on a caller-supplied equivalent operator.
// Used as the generated preconditioner of the matrix-free Krylov solve, so the
// iteration count stays ~flat in N (MG) while the outer mat-vec stays
// matrix-free. The loose tolerances passed to solve() are ignored in
// fixed-iter mode. NOTE: a V-cycle with (red-black) Gauss-Seidel smoothing is
// only approximately symmetric — classic CG tolerates it here (measured), but
// bicgstab/gmres are the fallback if it ever degrades.
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

    // Keeps the base's advanced apply_impl(alpha, b, beta, x) visible in this
    // scope (the declaration below would otherwise hide it).
    using AmrexLinOpBase<MlmgPrecond>::apply_impl;

    void apply_impl(const gko::LinOp* b, gko::LinOp* x) const override;

private:

    MLMG* mlmg_ = nullptr;
    std::shared_ptr<amrex::MultiFab> in_;
    std::shared_ptr<amrex::MultiFab> out_;
};

} // namespace blockamr::solvers
