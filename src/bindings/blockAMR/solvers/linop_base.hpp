// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <ginkgo/ginkgo.hpp>

#include <memory>

#include "types.hpp"

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
template<class D>
class AmrexLinOpBase : public gko::EnableLinOp<D>, public gko::EnableCreateMethod<D>
{
protected:

    explicit AmrexLinOpBase(std::shared_ptr<const gko::Executor> exec) : gko::EnableLinOp<D>(exec)
    {}

    AmrexLinOpBase(std::shared_ptr<const gko::Executor> exec, const gko::dim<2>& size)
        : gko::EnableLinOp<D>(exec, size)
    {}

    // Supplied by the derived class; re-declared here so it stays visible in
    // this scope for the advanced overload below to call.
    void apply_impl(const gko::LinOp* b, gko::LinOp* x) const override = 0;

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
};

} // namespace blockamr::solvers
