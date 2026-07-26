// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <ginkgo/ginkgo.hpp>

#include <memory>
#include <stdexcept>
#include <string>

#include "bc_geom.hpp"
#include "gmg_kernels.hpp"
#include "linop_base.hpp"
#include "profiling.hpp"
#include "transfer.hpp"

// ---------------------------------------------------------------------------
// The coarsest multigrid level as a Ginkgo operator, and the solver selection
// that runs on it.
//
// WHY THIS EXISTS. The V-cycle's bottom "solve" was a fixed number of smoother
// sweeps with no residual test (gmg_coarsest_sweeps). That is cheap and, being
// fixed work, it is exactly stationary -- which matters, because the V-cycle is
// used as a CG preconditioner and CG assumes the preconditioner is the SAME
// linear operator on every apply. But a smoother cannot touch the coarse grid's
// near-null modes at all: a consistent polynomial smoother has p(0) = 1, so the
// constant mode survives every sweep, and no number of sweeps converges it.
// MLMG solves its bottom with a Krylov method for this reason.
//
// WHY IT IS GINKGO RATHER THAN A HAND-ROLLED CG. A hand-rolled CG would be ~60
// lines and avoid a Ginkgo round trip on a grid of a few dozen cells. It would
// also be a second implementation of something Ginkgo already has, and -- more
// to the point -- CG is only valid for a SYMMETRIC operator. The moment the
// operator can be asymmetric (convection), the bottom needs BiCGStab or GMRES
// too, and then a hand-rolled path is three implementations. Naming a Ginkgo
// solver in one dispatch is the maintainable form of "the user chooses".
//
// ASYMMETRY IS NOT SPECIAL-CASED HERE. GmgBottomOp reads its own upper and lower
// coefficient array per direction (gmgApply*), so it represents an asymmetric
// operator exactly. Which SOLVER is legal on it is the caller's decision, made
// explicit by the `symmetric` flag rather than inferred -- see GmgPrecondT.
// ---------------------------------------------------------------------------

namespace blockamr::solvers
{

// A single GMG level exposed as a gko::LinOp: y = A x on that level's
// rediscretised coefficients, with the level's own geometry and boundary
// conditions applied to x's ghosts first.
//
// The value type is the HIERARCHY's type T, not the outer Krylov's double: the
// bottom solve happens entirely inside the V-cycle, on level fields that are
// already stored in T, so converting to double at the boundary would cost two
// full passes to buy precision that the surrounding V-cycle does not carry.
template<class T>
class GmgBottomOp : public AmrexLinOpBase<GmgBottomOp<T>, T>
{
public:

    // Required by Ginkgo's polymorphic-object machinery (create_default/clear).
    explicit GmgBottomOp(std::shared_ptr<const gko::Executor> exec)
        : AmrexLinOpBase<GmgBottomOp<T>, T>(exec)
    {}

    GmgBottomOp(
        std::shared_ptr<const gko::Executor> exec,
        gko::size_type n,
        const amrex::BoxArray& ba,
        const amrex::DistributionMapping& dm,
        amrex::Geometry geom,
        std::shared_ptr<GmgFab<T>> alpha,
        std::shared_ptr<GmgFab<T>> ux,
        std::shared_ptr<GmgFab<T>> lx,
        std::shared_ptr<GmgFab<T>> uy,
        std::shared_ptr<GmgFab<T>> ly,
        std::shared_ptr<GmgFab<T>> uz,
        std::shared_ptr<GmgFab<T>> lz,
        BcArray bc,
        bool onDevice
    )
        : AmrexLinOpBase<GmgBottomOp<T>, T>(exec, gko::dim<2> {n, n}), geom_(std::move(geom)),
          alpha_(std::move(alpha)), ux_(std::move(ux)), lx_(std::move(lx)), uy_(std::move(uy)),
          ly_(std::move(ly)), uz_(std::move(uz)), lz_(std::move(lz)), bc_(bc),
          hasPhysBc_(std::any_of(bc.begin(), bc.end(), [](int b) { return b != 0; })),
          onDevice_(onDevice)
    {
        // Own work fabs rather than borrowing the level's sol/rhs: the Krylov
        // method applies this operator to its own search directions, which are
        // not the level's solution, and aliasing them would corrupt the cycle.
        const amrex::MFInfo info =
            onDevice_ ? amrex::MFInfo() : amrex::MFInfo().SetArena(amrex::The_Pinned_Arena());
        in_ = std::make_shared<GmgFab<T>>(ba, dm, 1, 1, info);  // 1 ghost for the stencil
        out_ = std::make_shared<GmgFab<T>>(ba, dm, 1, 0, info); // valid-only
        in_->setVal(T(0));
        out_->setVal(T(0));
    }

protected:

    // Keeps the base's advanced apply_impl(alpha, b, beta, x) visible here.
    using AmrexLinOpBase<GmgBottomOp<T>, T>::apply_impl;

    void apply_impl(const gko::LinOp* b, gko::LinOp* x) const override
    {
        prof::Timer tAll("gmg.bottom.apply");
        auto exec = this->get_executor();
        if (onDevice_)
        {
            exec->synchronize(); // b written by Ginkgo on its own stream
            scatter_device(localValues<T>(b), *in_);
        }
        else
        {
            scatter(localValues<T>(b), *in_);
        }

        in_->FillBoundary(geom_.periodicity());
        if (!onDevice_)
        {
            amrex::Gpu::streamSynchronize(); // FillBoundary before host loops
        }
        if (hasPhysBc_)
        {
            if (onDevice_)
            {
                fillDomainBcGhostsDevice(*in_, geom_.Domain(), bc_);
            }
            else
            {
                fillDomainBcGhostsHost(*in_, geom_.Domain(), bc_);
            }
        }

        if (onDevice_)
        {
            gmgApplyDevice(*in_, *out_, *ux_, *lx_, *uy_, *ly_, *uz_, *lz_, *alpha_);
            gather_device(*out_, localValues<T>(x), 1.0);
            amrex::Gpu::streamSynchronize(); // x read by Ginkgo next
        }
        else
        {
            gmgApplyHost(*in_, *out_, *ux_, *lx_, *uy_, *ly_, *uz_, *lz_, *alpha_);
            gather(*out_, localValues<T>(x), 1.0);
        }
    }

private:

    amrex::Geometry geom_;
    std::shared_ptr<GmgFab<T>> alpha_, ux_, lx_, uy_, ly_, uz_, lz_;
    std::shared_ptr<GmgFab<T>> in_, out_;
    BcArray bc_ {};
    bool hasPhysBc_ = false;
    bool onDevice_ = false;
};

// Which bottom solver a caller asked for. "smoother" keeps the historical
// fixed-sweep behaviour, and is the DEFAULT precisely because it is stationary:
// see makeBottomSolver's note.
inline void validateBottomSolver(const std::string& kind, bool symmetric)
{
    static const char* kAll = "'smoother', 'cg', 'fcg', 'bicgstab', 'gmres' or 'gcr'";
    if (kind != "smoother" && kind != "cg" && kind != "fcg" && kind != "bicgstab" && kind != "gmres"
        && kind != "gcr")
    {
        throw std::runtime_error(
            "GmgPrecond: unknown gmg_bottom_solver '" + kind + "' (expected " + kAll + ")"
        );
    }
    // Refused, not warned: a CG bottom on an asymmetric operator does not fail
    // loudly, it converges to the wrong correction or stalls, and the caller
    // sees only a worse outer iteration count.
    if (!symmetric && (kind == "cg" || kind == "fcg"))
    {
        throw std::runtime_error(
            "GmgPrecond: gmg_bottom_solver='" + kind
            + "' needs a symmetric operator, but symmetric=False was set. Use "
              "'bicgstab', 'gmres' or 'gcr' for an asymmetric bottom."
        );
    }
}

// Generate the bottom solver on `op`.
//
// STATIONARITY. A residual-tested Krylov bottom takes a different number of
// iterations on different right-hand sides, which makes the V-cycle a DIFFERENT
// linear operator on each apply. gko::solver::Cg and Bicgstab both assume a
// fixed preconditioner, so an outer CG over a V-cycle with an adaptive bottom is
// outside its theory. Two ways to stay inside it, both available here:
//   - solve the bottom essentially exactly (a tight `rtol`), so the variation is
//     below what the outer solver can see. The bottom is a handful of cells, so
//     this is cheap -- it is the recommended setting.
//   - drive the outer solve with a FLEXIBLE method (solver='gcr' or 'fcg'),
//     which tolerates a preconditioner that varies between applies.
// The DEFAULT bottom is still 'smoother' (fixed sweeps), which is stationary by
// construction and reproduces the historical behaviour exactly.
template<class T>
std::shared_ptr<const gko::LinOp> makeBottomSolver(
    const std::string& kind,
    std::shared_ptr<const gko::Executor> exec,
    std::shared_ptr<const gko::LinOp> op,
    int max_iter,
    double rtol
)
{
    if (kind == "smoother")
    {
        return nullptr;
    }
    std::vector<std::shared_ptr<const gko::stop::CriterionFactory>> criteria;
    criteria.push_back(
        gko::stop::Iteration::build().with_max_iters(static_cast<gko::size_type>(max_iter)).on(exec)
    );
    criteria.push_back(gko::stop::ResidualNorm<T>::build()
                           .with_baseline(gko::stop::mode::rhs_norm)
                           .with_reduction_factor(static_cast<gko::remove_complex<T>>(rtol))
                           .on(exec));

    if (kind == "cg")
    {
        return gko::share(gko::solver::Cg<T>::build().with_criteria(criteria).on(exec)->generate(op)
        );
    }
    if (kind == "fcg")
    {
        return gko::share(gko::solver::Fcg<T>::build().with_criteria(criteria).on(exec)->generate(op
        ));
    }
    if (kind == "bicgstab")
    {
        return gko::share(
            gko::solver::Bicgstab<T>::build().with_criteria(criteria).on(exec)->generate(op)
        );
    }
    if (kind == "gmres")
    {
        return gko::share(
            gko::solver::Gmres<T>::build().with_criteria(criteria).on(exec)->generate(op)
        );
    }
    if (kind == "gcr")
    {
        return gko::share(gko::solver::Gcr<T>::build().with_criteria(criteria).on(exec)->generate(op
        ));
    }
    throw std::runtime_error("GmgPrecond: unknown gmg_bottom_solver '" + kind + "'");
}

} // namespace blockamr::solvers
