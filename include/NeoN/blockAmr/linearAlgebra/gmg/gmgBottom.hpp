// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <ginkgo/ginkgo.hpp>

#include <memory>
#include <stdexcept>
#include <string>

#include "NeoN/blockAmr/core/bc.hpp"
#include "NeoN/blockAmr/linearAlgebra/matrixFree/linOpBase.hpp"
#include "NeoN/blockAmr/core/profiling.hpp"
#include "NeoN/blockAmr/linearAlgebra/transfer.hpp"
#include "NeoN/blockAmr/linearAlgebra/gmg/gmgKernels.hpp"

// The coarsest multigrid level as a Ginkgo operator, and the solver selection on it.
// Why a Krylov bottom, why Ginkgo, and the symmetry rules:
// report/blockamr-gmg-notes.md#bottom (NeoFOAM repo).

namespace blockamr::la
{

// A single GMG level as a gko::LinOp: y = A x on that level's rediscretised coefficients,
// with the level's geometry and BCs applied to x's ghosts first. The value type is the
// HIERARCHY's T, not the outer Krylov's double — the bottom runs inside the V-cycle.
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
        // Own work fabs, not the level's sol/rhs: aliasing would corrupt the cycle.
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

        const FaceCoeffs<T> fc {
            alpha_.get(), ux_.get(), lx_.get(), uy_.get(), ly_.get(), uz_.get(), lz_.get()
        };
        gmgApply(*in_, *out_, fc, onDevice_);
        if (onDevice_)
        {
            gather_device(*out_, localValues<T>(x), 1.0);
            amrex::Gpu::streamSynchronize(); // x read by Ginkgo next
        }
        else
        {
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
    // Refused, not warned: a CG bottom on an asymmetric operator fails silently.
    if (!symmetric && (kind == "cg" || kind == "fcg"))
    {
        throw std::runtime_error(
            "GmgPrecond: gmg_bottom_solver='" + kind
            + "' needs a symmetric operator, but symmetric=False was set. Use "
              "'bicgstab', 'gmres' or 'gcr' for an asymmetric bottom."
        );
    }
}

// Generate the bottom solver on `op`. A residual-tested Krylov bottom is NOT stationary,
// which an outer CG assumes: keep rtol tight or use a flexible outer method — the DEFAULT
// 'smoother' (fixed sweeps) is stationary (report/blockamr-gmg-notes.md#bottom).
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

} // namespace blockamr::la
