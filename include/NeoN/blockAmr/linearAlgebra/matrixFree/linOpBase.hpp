// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <ginkgo/ginkgo.hpp>

#include <memory>

#include "NeoN/blockAmr/linearAlgebra/distVec.hpp"
#include "NeoN/blockAmr/core/profiling.hpp"

namespace blockamr::la
{

// CRTP base for the matrix-free Ginkgo operators here: it bundles gko::EnableLinOp and
// gko::EnableCreateMethod and supplies the one advanced apply_impl(alpha, b, beta, x) through
// the derived class' apply_impl(b, x). How to derive: report/blockamr-linear-algebra-notes.md
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

    // Supplied by the derived class; re-declared so the advanced overload can call it.
    void apply_impl(const gko::LinOp* b, gko::LinOp* x) const override = 0;

    // x = alpha * op(b) + beta * x. Every derived apply_impl(b, x) OVERWRITES all of x, so the
    // intermediate is a reused scratch rather than a clone of x. The beta branches skip whole
    // vector passes: beta == 1 needs no scale, beta == 0 discards x and needs no scratch.
    void apply_impl(
        const gko::LinOp* alpha, const gko::LinOp* b, const gko::LinOp* beta, gko::LinOp* x
    ) const override
    {
        prof::Timer tAll("adv.apply");
        // A view, so scale/add_scaled hit x's own memory; both are elementwise.
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

    // alpha/beta are 1x1 Dense on the solve executor; a device value is staged through the
    // host master.
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

    // shared_ptr, not unique_ptr: EnablePolymorphicAssignment gives these operators a
    // copy-assignment, which a move-only member would delete. No state is kept across calls.
    mutable std::shared_ptr<DenseV> scratch_;
};

} // namespace blockamr::la
