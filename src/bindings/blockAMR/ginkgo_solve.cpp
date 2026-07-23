// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

// Matrix-free Ginkgo CG solve of an AMReX MLLinOp system (single-level, CPU
// serial). The mat-vec is MLMG::apply, which computes out = L(in) — the raw
// Laplacian, negative-definite. CG needs SPD, so the custom LinOp returns
// -L(in) and the RHS is negated to match; the solution is unchanged.

#include <nanobind/nanobind.h>

#include <AMReX_Arena.H>
#include <AMReX_MultiFab.H>
#include <AMReX_MLLinOp.H>
#include <AMReX_MLMG.H>

#include <ginkgo/ginkgo.hpp>

#include <cstdint>
#include <memory>
#include <vector>

#include "bindings.hpp"

namespace nb = nanobind;

namespace
{

using MLMG = amrex::MLMGT<amrex::MultiFab>;
using Dense = gko::matrix::Dense<double>;

// Flat-vector <-> MultiFab transfer (component 0, valid cells only).
// gather and scatter MUST traverse cells in the identical order: MFIter
// without tiling, then k,j,i over the valid box. MultiFabs live in device
// memory by default in GPU builds, so access is staged through explicit
// host copies unless the arena is host-accessible. `scale` lets gather
// apply the SPD sign flip (-L) in the same pass.
void gather(const amrex::MultiFab& mf, double* buf, double scale)
{
    const bool hostOk = mf.arena()->isHostAccessible();
    amrex::Gpu::streamSynchronize();
    std::size_t idx = 0;
    for (amrex::MFIter mfi(mf); mfi.isValid(); ++mfi)
    {
        const amrex::Box& vbx = mfi.validbox();
        const auto& fab = mf[mfi];
        const amrex::Box& fbx = fab.box();
        std::vector<double> stage;
        auto arr = fab.const_array();
        if (!hostOk)
        {
            // Component 0 occupies the first numPts() elements of the fab.
            stage.resize(static_cast<std::size_t>(fbx.numPts()));
            amrex::Gpu::dtoh_memcpy(stage.data(), fab.dataPtr(), stage.size() * sizeof(double));
            arr = amrex::makeArray4<const double>(stage.data(), fbx, 1);
        }
        const auto lo = amrex::lbound(vbx);
        const auto hi = amrex::ubound(vbx);
        for (int k = lo.z; k <= hi.z; ++k)
        {
            for (int j = lo.y; j <= hi.y; ++j)
            {
                for (int i = lo.x; i <= hi.x; ++i)
                {
                    buf[idx++] = scale * arr(i, j, k);
                }
            }
        }
    }
}

void scatter(const double* buf, amrex::MultiFab& mf)
{
    const bool hostOk = mf.arena()->isHostAccessible();
    amrex::Gpu::streamSynchronize();
    std::size_t idx = 0;
    for (amrex::MFIter mfi(mf); mfi.isValid(); ++mfi)
    {
        const amrex::Box& vbx = mfi.validbox();
        auto& fab = mf[mfi];
        const amrex::Box& fbx = fab.box();
        std::vector<double> stage;
        auto arr = fab.array();
        if (!hostOk)
        {
            // Round-trip the full fab so ghost values survive the update.
            stage.resize(static_cast<std::size_t>(fbx.numPts()));
            amrex::Gpu::dtoh_memcpy(stage.data(), fab.dataPtr(), stage.size() * sizeof(double));
            arr = amrex::makeArray4<double>(stage.data(), fbx, 1);
        }
        const auto lo = amrex::lbound(vbx);
        const auto hi = amrex::ubound(vbx);
        for (int k = lo.z; k <= hi.z; ++k)
        {
            for (int j = lo.y; j <= hi.y; ++j)
            {
                for (int i = lo.x; i <= hi.x; ++i)
                {
                    arr(i, j, k) = buf[idx++];
                }
            }
        }
        if (!hostOk)
        {
            amrex::Gpu::htod_memcpy(fab.dataPtr(), stage.data(), stage.size() * sizeof(double));
        }
    }
}

// Matrix-free SPD operator: x = -L(b), with MLMG::apply as the mat-vec.
class AmrexOp : public gko::EnableLinOp<AmrexOp>, public gko::EnableCreateMethod<AmrexOp>
{
public:

    // Exec-only constructor required by the polymorphic-object machinery
    // (create_default / clear).
    explicit AmrexOp(std::shared_ptr<const gko::Executor> exec) : gko::EnableLinOp<AmrexOp>(exec) {}

    AmrexOp(
        std::shared_ptr<const gko::Executor> exec,
        MLMG* mlmg,
        const amrex::BoxArray& ba,
        const amrex::DistributionMapping& dm,
        gko::size_type n
    )
        : gko::EnableLinOp<AmrexOp>(exec, gko::dim<2> {n, n}), mlmg_(mlmg),
          // shared_ptr, not values: MultiFab is move-only, but
          // EnablePolymorphicAssignment needs AmrexOp copy-assignable.
          // MLMG::apply needs >= 1 ghost cell on the input, hence ng=1 on in_.
          // Pinned (host-accessible) arena: gather/scatter run host-side every
          // mat-vec, and GPU kernels can address pinned memory directly.
          in_(std::make_shared<amrex::MultiFab>(
              ba, dm, 1, 1, amrex::MFInfo().SetArena(amrex::The_Pinned_Arena())
          )),
          out_(std::make_shared<amrex::MultiFab>(
              ba, dm, 1, 0, amrex::MFInfo().SetArena(amrex::The_Pinned_Arena())
          ))
    {
        in_->setVal(0.0);
        out_->setVal(0.0);
        amrex::Gpu::streamSynchronize();
    }

protected:

    void apply_impl(const gko::LinOp* b, gko::LinOp* x) const override
    {
        auto denseB = gko::as<Dense>(b);
        auto denseX = gko::as<Dense>(x);
        scatter(denseB->get_const_values(), *in_);
        mlmg_->apply({out_.get()}, {in_.get()});
        // Negate on gather: x = -L(in) makes the operator SPD.
        gather(*out_, denseX->get_values(), -1.0);
    }

    void apply_impl(
        const gko::LinOp* alpha, const gko::LinOp* b, const gko::LinOp* beta, gko::LinOp* x
    ) const override
    {
        auto denseX = gko::as<Dense>(x);
        auto tmp = denseX->clone();
        this->apply_impl(b, tmp.get());
        denseX->scale(beta);
        denseX->add_scaled(alpha, tmp);
    }

private:

    MLMG* mlmg_ = nullptr;
    std::shared_ptr<amrex::MultiFab> in_;
    std::shared_ptr<amrex::MultiFab> out_;
};

} // namespace

void registerGinkgoSolve(nb::module_& m)
{
    using namespace amrex;

    using MLLinOp = MLLinOpT<MultiFab>;

    m.def(
        "ginkgo_solve",
        [](MLLinOp& lp, MultiFab& sol, const MultiFab& rhs, int max_iter, double rtol)
        {
            MLMG mlmg(lp);

            auto exec = gko::ReferenceExecutor::create();
            const auto n = static_cast<gko::size_type>(sol.boxArray().numPts());

            auto op =
                gko::share(AmrexOp::create(exec, &mlmg, sol.boxArray(), sol.DistributionMap(), n));

            // b = -rhs (SPD sign flip, matching -L in AmrexOp); x0 = sol.
            auto b = Dense::create(exec, gko::dim<2> {n, 1});
            gather(rhs, b->get_values(), -1.0);
            auto x = Dense::create(exec, gko::dim<2> {n, 1});
            gather(sol, x->get_values(), 1.0);

            auto logger = gko::share(gko::log::Convergence<double>::create());
            auto solver =
                gko::solver::Cg<double>::build()
                    .with_criteria(
                        gko::stop::Iteration::build().with_max_iters(
                            static_cast<gko::size_type>(max_iter)
                        ),
                        gko::stop::ResidualNorm<double>::build().with_reduction_factor(rtol)
                    )
                    .on(exec)
                    ->generate(op);
            solver->add_logger(logger);
            solver->apply(b, x);

            scatter(x->get_const_values(), sol);

            // Explicit final residual ||b - A x||_2 for reporting.
            auto res = b->clone();
            auto one = gko::initialize<Dense>({1.0}, exec);
            auto negOne = gko::initialize<Dense>({-1.0}, exec);
            op->apply(negOne, x, one, res);
            auto norm = Dense::create(exec, gko::dim<2> {1, 1});
            res->compute_norm2(norm);

            nb::dict result;
            result["num_iters"] = static_cast<std::int64_t>(logger->get_num_iterations());
            result["res_norm"] = norm->at(0, 0);
            return result;
        },
        nb::arg("lp"),
        nb::arg("sol"),
        nb::arg("rhs"),
        nb::arg("max_iter") = 1000,
        nb::arg("rtol") = 1e-10
    );
}
