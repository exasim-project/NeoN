// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "mlmg_ops.hpp"

#include <AMReX_GpuDevice.H>

#include <cstddef>

#include "../common/transfer.hpp"

namespace blockamr::solvers
{

AmrexOp::AmrexOp(std::shared_ptr<const gko::Executor> exec) : AmrexLinOpBase<AmrexOp>(exec) {}

AmrexOp::AmrexOp(
    std::shared_ptr<const gko::Executor> exec,
    MLMG* mlmg,
    const amrex::BoxArray& ba,
    const amrex::DistributionMapping& dm,
    gko::size_type n,
    double sign
)
    : AmrexLinOpBase<AmrexOp>(exec, gko::dim<2> {n, n}), mlmg_(mlmg), sign_(sign),
      // shared_ptr, not values: MultiFab is move-only, but
      // EnablePolymorphicAssignment needs AmrexOp copy-assignable.
      // MLMG::apply needs >= 1 ghost cell on the input, hence ng=1 on in_.
      // Default (device) arena: on a Cuda executor the pack/unpack kernels
      // and MLMG::apply all run on the GPU with no host copies; on the
      // reference (CPU) path gather/scatter stage these via dtoh/htod.
      in_(std::make_shared<amrex::MultiFab>(ba, dm, 1, 1)),
      out_(std::make_shared<amrex::MultiFab>(ba, dm, 1, 0)),
      c0_(std::make_shared<amrex::MultiFab>(ba, dm, 1, 0))
{
    in_->setVal(0.0);
    out_->setVal(0.0);
    // Affine offset c0 = L_inhom(0): captures the set_level_bc
    // contribution so apply_impl can subtract it and stay linear.
    mlmg_->apply({c0_.get()}, {in_.get()});
    // apply overwrites in_'s ghost cells; restore the all-zero state.
    in_->setVal(0.0);
    amrex::Gpu::streamSynchronize();
}

void AmrexOp::apply_impl(const gko::LinOp* b, gko::LinOp* x) const
{
    auto exec = this->get_executor();
    // A ReferenceExecutor is its own master; a device (Cuda) executor has a
    // distinct host master. On device, pack/unpack run as AMReX kernels
    // straight against the Ginkgo device pointers, so the entire mat-vec
    // stays on the GPU. On host, stage through host clones.
    const bool onDevice = exec->get_master().get() != exec.get();
    if (onDevice)
    {
        // Ordering across the two libraries' streams is done host-side:
        // wait for Ginkgo's writes to b, run the AMReX mat-vec, then wait
        // for its writes to x before Ginkgo reads them.
        exec->synchronize();
        scatter_device(localValues<double>(b), *in_);
        mlmg_->apply({out_.get()}, {in_.get()});
        amrex::MultiFab::Subtract(*out_, *c0_, 0, 0, 1, 0);
        gather_device(*out_, localValues<double>(x), sign_);
        amrex::Gpu::streamSynchronize();
    }
    else
    {
        auto host = exec->get_master();
        auto bHost = gko::clone(host, localView<double>(b));
        scatter(bHost->get_const_values(), *in_);
        mlmg_->apply({out_.get()}, {in_.get()});
        // Remove the affine BC offset, then apply the SPD sign on gather:
        // x = sign*(L_inhom(in) - c0).
        amrex::MultiFab::Subtract(*out_, *c0_, 0, 0, 1, 0);
        auto xHost = Dense::create(host, gko::dim<2> {localRows(x), 1});
        gather(*out_, xHost->get_values(), sign_);
        localView<double>(x)->copy_from(xHost);
    }
}

CompositeAmrexOp::CompositeAmrexOp(std::shared_ptr<const gko::Executor> exec)
    : AmrexLinOpBase<CompositeAmrexOp>(exec)
{}

CompositeAmrexOp::CompositeAmrexOp(
    std::shared_ptr<const gko::Executor> exec,
    MLMG* mlmg,
    const std::vector<amrex::BoxArray>& bas,
    const std::vector<amrex::DistributionMapping>& dms,
    gko::size_type n,
    double sign
)
    : AmrexLinOpBase<CompositeAmrexOp>(exec, gko::dim<2> {n, n}), mlmg_(mlmg), sign_(sign)
{
    long off = 0;
    for (std::size_t lev = 0; lev < bas.size(); ++lev)
    {
        // shared_ptr for copy-assignability (see AmrexOp); MLMG::apply
        // needs >= 1 ghost cell on the input, hence ng=1 on in_.
        in_.push_back(std::make_shared<amrex::MultiFab>(bas[lev], dms[lev], 1, 1));
        out_.push_back(std::make_shared<amrex::MultiFab>(bas[lev], dms[lev], 1, 0));
        c0_.push_back(std::make_shared<amrex::MultiFab>(bas[lev], dms[lev], 1, 0));
        in_.back()->setVal(0.0);
        out_.back()->setVal(0.0);
        off_.push_back(off);
        off += bas[lev].numPts();
    }
    // Affine offset c0 = L_inhom(0) per level (set_level_bc contribution).
    mlmg_->apply(ptrs(c0_), ptrs(in_));
    // apply overwrites in_'s ghost cells; restore the all-zero state.
    for (auto& mf : in_)
    {
        mf->setVal(0.0);
    }
    amrex::Gpu::streamSynchronize();
}

void CompositeAmrexOp::apply_impl(const gko::LinOp* b, gko::LinOp* x) const
{
    auto exec = this->get_executor();
    const bool onDevice = exec->get_master().get() != exec.get();
    if (onDevice)
    {
        exec->synchronize(); // b written by Ginkgo
        const double* bv = localValues<double>(b);
        for (std::size_t lev = 0; lev < in_.size(); ++lev)
        {
            scatter_device(bv + off_[lev], *in_[lev]);
        }
        mlmg_->apply(ptrs(out_), ptrs(in_));
        double* xv = localValues<double>(x);
        for (std::size_t lev = 0; lev < out_.size(); ++lev)
        {
            amrex::MultiFab::Subtract(*out_[lev], *c0_[lev], 0, 0, 1, 0);
            gather_device(*out_[lev], xv + off_[lev], sign_);
        }
        amrex::Gpu::streamSynchronize(); // x complete before Ginkgo reads it
    }
    else
    {
        auto host = exec->get_master();
        auto bHost = gko::clone(host, localView<double>(b));
        for (std::size_t lev = 0; lev < in_.size(); ++lev)
        {
            scatter(bHost->get_const_values() + off_[lev], *in_[lev]);
        }
        mlmg_->apply(ptrs(out_), ptrs(in_));
        auto xHost = Dense::create(host, gko::dim<2> {localRows(x), 1});
        for (std::size_t lev = 0; lev < out_.size(); ++lev)
        {
            amrex::MultiFab::Subtract(*out_[lev], *c0_[lev], 0, 0, 1, 0);
            gather(*out_[lev], xHost->get_values() + off_[lev], sign_);
        }
        localView<double>(x)->copy_from(xHost);
    }
}

amrex::Vector<amrex::MultiFab*>
CompositeAmrexOp::ptrs(const std::vector<std::shared_ptr<amrex::MultiFab>>& v)
{
    amrex::Vector<amrex::MultiFab*> p;
    for (const auto& m : v)
    {
        p.push_back(m.get());
    }
    return p;
}

MlmgPrecond::MlmgPrecond(std::shared_ptr<const gko::Executor> exec)
    : AmrexLinOpBase<MlmgPrecond>(exec)
{}

MlmgPrecond::MlmgPrecond(
    std::shared_ptr<const gko::Executor> exec,
    MLMG* mlmg,
    const amrex::BoxArray& ba,
    const amrex::DistributionMapping& dm,
    gko::size_type n,
    int n_cycles
)
    : AmrexLinOpBase<MlmgPrecond>(exec, gko::dim<2> {n, n}), mlmg_(mlmg),
      // shared_ptr for copy-assignability (see AmrexOp). Default (device)
      // arena: on a Cuda executor pack/unpack and the V-cycles all run on
      // the GPU; the reference path stages via scatter/gather.
      in_(std::make_shared<amrex::MultiFab>(ba, dm, 1, 1)),
      out_(std::make_shared<amrex::MultiFab>(ba, dm, 1, 1))
{
    in_->setVal(0.0);
    out_->setVal(0.0);
    mlmg_->setVerbose(0);
    mlmg_->setBottomVerbose(0);
    // Exactly n_cycles V-cycles per apply; solve() tolerances are ignored.
    mlmg_->setFixedIter(n_cycles);
    amrex::Gpu::streamSynchronize();
}

void MlmgPrecond::apply_impl(const gko::LinOp* b, gko::LinOp* x) const
{
    auto exec = this->get_executor();
    const bool onDevice = exec->get_master().get() != exec.get();
    if (onDevice)
    {
        exec->synchronize(); // b written by Ginkgo
        scatter_device(localValues<double>(b), *in_);
        out_->setVal(0.0); // z0 = 0: apply M^{-1}, not a warm-started solve
        mlmg_->solve({out_.get()}, {in_.get()}, 1e-4, 0.0);
        gather_device(*out_, localValues<double>(x), 1.0);
        amrex::Gpu::streamSynchronize(); // x complete before Ginkgo reads it
    }
    else
    {
        auto host = exec->get_master();
        auto bHost = gko::clone(host, localView<double>(b));
        scatter(bHost->get_const_values(), *in_);
        out_->setVal(0.0);
        mlmg_->solve({out_.get()}, {in_.get()}, 1e-4, 0.0);
        auto xHost = Dense::create(host, gko::dim<2> {localRows(x), 1});
        gather(*out_, xHost->get_values(), 1.0);
        localView<double>(x)->copy_from(xHost);
    }
}

} // namespace blockamr::solvers
