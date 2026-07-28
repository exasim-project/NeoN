// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <ginkgo/ginkgo.hpp>

#include <memory>

#include "NeoN/blockAmr/core/distVec.hpp"
#include "NeoN/blockAmr/core/profiling.hpp"

namespace blockamr::solvers
{

// CRTP base for the matrix-free Ginkgo operators in this directory. It bundles
// the two mixins every one of them needs (gko::EnableLinOp for the LinOp
// plumbing, gko::EnableCreateMethod for the static create()) and supplies the
// ONE implementation of the "advanced" apply_impl:
//   x = alpha * op(b) + beta * x,
// expressed through the derived class' simple apply_impl(b, x) on a temporary.
// Every derived operator previously carried a byte-identical copy of it.
//
// A derived class D:
//   - derives as `public AmrexLinOpBase<D>` (in place of the former
//     `public gko::EnableLinOp<D>, public gko::EnableCreateMethod<D>`),
//   - forwards to `AmrexLinOpBase<D>(exec)` / `AmrexLinOpBase<D>(exec, size)`
//     in its constructors (in place of `gko::EnableLinOp<D>(...)`),
//   - implements only `apply_impl(const gko::LinOp* b, gko::LinOp* x) const`,
//     preceded by `using AmrexLinOpBase<D>::apply_impl;` so that declaration
//     does not hide the advanced overload (nvcc warning 611 / -Woverloaded-
//     virtual; the code is correct either way, the using-declaration only
//     keeps the build log clean).
// The exec-only constructor stays required by the polymorphic-object machinery
// (create_default / clear), which does `new D(exec)`.
// V is the value type of the Dense vectors this operator is applied to -- double
// for the fp64 Krylov path, float for the mixed-precision one. It appears only in
// the Dense casts below; gko::EnableLinOp carries no value type, so a derived
// operator is a plain gko::LinOp either way and Cg<float> accepts it directly.
template<class D, class V = double>
class AmrexLinOpBase : public gko::EnableLinOp<D>, public gko::EnableCreateMethod<D>
{
protected:

    using DenseV = gko::matrix::Dense<V>;


    explicit AmrexLinOpBase(std::shared_ptr<const gko::Executor> exec) : gko::EnableLinOp<D>(exec)
    {}

    AmrexLinOpBase(std::shared_ptr<const gko::Executor> exec, const gko::dim<2>& size)
        : gko::EnableLinOp<D>(exec, size)
    {}

    // Supplied by the derived class; re-declared here so it stays visible in
    // this scope for the advanced overload below to call.
    void apply_impl(const gko::LinOp* b, gko::LinOp* x) const override = 0;

    // x = alpha * op(b) + beta * x.
    //
    // Every derived apply_impl(b, x) OVERWRITES the whole of x — each one ends
    // in a gather across the entire flat vector — so the intermediate needs no
    // initial contents. It is therefore a reused scratch buffer rather than a
    // clone of x: cloning cost an allocation plus a full device copy of x per
    // call, and both were discarded by the very next line.
    //
    // The beta branches skip whole vector passes. beta == 1 is the case Ginkgo's
    // Ir and the Krylov initial residual take (r = b - A x), where scale(beta)
    // is a read-modify-write pass that multiplies by one. beta == 0 discards x
    // entirely, so op(b) can be written straight into it with no scratch at all.
    void apply_impl(
        const gko::LinOp* alpha, const gko::LinOp* b, const gko::LinOp* beta, gko::LinOp* x
    ) const override
    {
        prof::Timer tAll("adv.apply");
        // A view, so scale/add_scaled below hit x's own memory. Both are
        // elementwise, so doing them on the local part is the whole operation.
        auto denseX = localView<V>(x);
        const double alphaVal = hostScalar(alpha);
        const double betaVal = hostScalar(beta);

        if (betaVal == 0.0)
        {
            this->apply_impl(b, x);
            if (alphaVal != 1.0)
            {
                prof::Timer t("adv.scale");
                denseX->scale(alpha);
            }
            return;
        }

        const gko::dim<2> size = denseX->get_size();
        if (!scratch_ || scratch_->get_size() != size)
        {
            prof::Timer t("adv.alloc");
            scratch_ = DenseV::create(this->get_executor(), size);
        }
        this->apply_impl(b, scratch_.get());
        if (betaVal != 1.0)
        {
            prof::Timer t("adv.scale");
            denseX->scale(beta);
        }
        {
            prof::Timer t("adv.addscaled");
            denseX->add_scaled(alpha, scratch_);
        }
    }

private:

    // alpha/beta are 1x1 Dense on the solve executor; a device value is staged
    // through the host master to read it (cf. ResidualHistoryLogger::readScalar).
    static double hostScalar(const gko::LinOp* s)
    {
        auto d = gko::as<DenseV>(s);
        auto exec = d->get_executor();
        if (exec->get_master().get() != exec.get())
        {
            return gko::clone(exec->get_master(), d)->at(0, 0);
        }
        return d->at(0, 0);
    }

    // shared_ptr, not unique_ptr: Ginkgo's EnablePolymorphicAssignment gives
    // these operators a copy-assignment, which a move-only member would delete.
    // Sharing a scratch buffer between copies is harmless — it holds no state
    // across calls.
    mutable std::shared_ptr<DenseV> scratch_;
};

} // namespace blockamr::solvers
