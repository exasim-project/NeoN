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

// CRTP base for the matrix-free Ginkgo operators in this directory: it bundles the
// two mixins each needs (gko::EnableLinOp, gko::EnableCreateMethod) and supplies the
// ONE implementation of the "advanced" apply_impl, x = alpha * op(b) + beta * x,
// expressed through the derived class' simple apply_impl(b, x) on a temporary —
// every derived operator used to carry a byte-identical copy of it.
//
// A derived class D derives as `public AmrexLinOpBase<D>`, forwards to
// `AmrexLinOpBase<D>(exec[, size])` in its constructors, and implements only
// apply_impl(b, x), preceded by `using AmrexLinOpBase<D>::apply_impl;` so that
// declaration does not hide the advanced overload (nvcc warning 611 /
// -Woverloaded-virtual; cosmetic, the code is correct either way). The exec-only
// constructor is required by create_default / clear, which do `new D(exec)`.
//
// V is the value type of the Dense vectors only; gko::EnableLinOp carries no value
// type, so a derived operator is a plain gko::LinOp and Cg<float> accepts it.
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

    // x = alpha * op(b) + beta * x.
    //
    // Every derived apply_impl(b, x) OVERWRITES the whole of x, so the intermediate
    // needs no initial contents: a reused scratch buffer rather than a clone of x,
    // which cost an allocation plus a full device copy discarded by the next line.
    // The beta branches skip whole vector passes: at beta == 1 (Ginkgo's Ir and the
    // Krylov initial residual r = b - A x) scale(beta) would multiply by one, and
    // beta == 0 discards x, so op(b) goes straight into it with no scratch.
    void apply_impl(
        const gko::LinOp* alpha, const gko::LinOp* b, const gko::LinOp* beta, gko::LinOp* x
    ) const override
    {
        prof::Timer tAll("adv.apply");
        // A view, so scale/add_scaled hit x's own memory; both are elementwise, so
        // the local part is the whole operation.
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

    // alpha/beta are 1x1 Dense on the solve executor; a device value is staged through
    // the host master (cf. ResidualHistoryLogger::readScalar).
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

    // shared_ptr, not unique_ptr: Ginkgo's EnablePolymorphicAssignment gives these
    // operators a copy-assignment, which a move-only member would delete. Sharing is
    // harmless — this holds no state across calls.
    mutable std::shared_ptr<DenseV> scratch_;
};

} // namespace blockamr::la
