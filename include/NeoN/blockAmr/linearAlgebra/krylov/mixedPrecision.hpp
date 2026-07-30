// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <ginkgo/ginkgo.hpp>

#include <memory>
#include <utility>

#include "NeoN/blockAmr/linearAlgebra/matrixFree/linOpBase.hpp"
#include "NeoN/blockAmr/core/profiling.hpp"
#include "NeoN/blockAmr/core/types.hpp"

// An FP32 inner solver wearing an FP64 LinOp's clothes so gko::solver::Ir<double> can
// drive it; the answer stays in the outer fp64 loop. Measured and REJECTED (1.41x slower
// than fp64 CG); mp_inner_max_iter is the knob. report/blockamr-precision-measurements.md

namespace blockamr::la
{

class MixedPrecisionSolve : public AmrexLinOpBase<MixedPrecisionSolve>
{
public:

    using Dense32 = gko::matrix::Dense<float>;

    // Required by Ginkgo's polymorphic-object machinery (create_default / clear).
    explicit MixedPrecisionSolve(std::shared_ptr<const gko::Executor> exec)
        : AmrexLinOpBase<MixedPrecisionSolve>(exec)
    {}

    MixedPrecisionSolve(
        std::shared_ptr<const gko::Executor> exec,
        gko::size_type n,
        std::shared_ptr<const gko::LinOp> inner
    )
        : AmrexLinOpBase<MixedPrecisionSolve>(exec, gko::dim<2> {n, n}), inner_(std::move(inner))
    {}

protected:

    using AmrexLinOpBase<MixedPrecisionSolve>::apply_impl;

    void apply_impl(const gko::LinOp* b, gko::LinOp* x) const override
    {
        prof::Timer tAll("mp.apply");
        const gko::dim<2> size {localRows(b), 1};
        if (!b32_ || b32_->get_size() != size)
        {
            prof::Timer t("mp.alloc");
            b32_ = Dense32::create(this->get_executor(), size);
            x32_ = Dense32::create(this->get_executor(), size);
            // What the inner Cg<float> is handed: the buffers are sized by THIS rank's
            // rows, so a plain Dense would make its dots and norms rank-local.
            b32Global_ = makeGlobalVec(this->get_executor(), this->get_size()[0], b32_.get());
            x32Global_ = makeGlobalVec(this->get_executor(), this->get_size()[0], x32_.get());
        }
        {
            // Ginkgo rounds per element on the device: one pass, no host round-trip.
            prof::Timer t("mp.down");
            localView<double>(b)->convert_to(b32_);
        }
        {
            // Zero guess, not the incoming x: this operator IS the correction S(r).
            prof::Timer t("mp.zero");
            x32_->fill(0.0F);
        }
        {
            prof::Timer t("mp.inner");
            inner_->apply(b32Global_, x32Global_);
        }
        {
            prof::Timer t("mp.up");
            x32_->convert_to(localView<double>(x).get());
        }
    }

private:

    std::shared_ptr<const gko::LinOp> inner_;
    // shared_ptr for the same reason as AmrexLinOpBase::scratch_ (copy-assignment).
    mutable std::shared_ptr<Dense32> b32_, x32_;
    // Non-owning views of the two above; rebuilt whenever they are.
    mutable std::shared_ptr<gko::LinOp> b32Global_, x32Global_;
};

} // namespace blockamr::la
