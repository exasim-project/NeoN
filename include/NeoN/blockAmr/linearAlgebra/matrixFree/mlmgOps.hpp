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

// Multi-level (composite AMR) AmrexOp: the Ginkgo vector concatenates all levels' valid cells
// (coarsest first) and the mat-vec is the COMPOSITE MLMG::apply. Covered coarse columns are
// ZERO and the operator is NOT symmetric: report/blockamr-linear-algebra-notes.md
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

// Multigrid preconditioner: z = M^{-1} r from a FIXED number of MLMG V-cycles (setFixedIter,
// which ignores solve()'s tolerances) on a caller-supplied equivalent operator. A V-cycle with
// Gauss-Seidel smoothing is only approximately symmetric -- CG tolerates it here (measured).
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
