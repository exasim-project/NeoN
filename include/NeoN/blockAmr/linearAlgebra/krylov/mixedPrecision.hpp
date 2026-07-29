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

// An FP32 solver wearing an FP64 LinOp's clothes so gko::solver::Ir<double> can drive
// it: b down to fp32, a preconditioned Cg<float> from a zero guess, the result back
// up. Everything deciding the ANSWER stays in the outer fp64 loop, so a weak inner
// solve costs outer iterations and never accuracy.
//
// Measured and REJECTED: the fp32 iteration is the predicted 1.18x cheaper, but
// refinement needs 1.5x the preconditioner applies — 1.41x slower than plain fp64 CG.
// The inner stopping criterion is not trustworthy, so mp_inner_max_iter (not
// mp_inner_rtol) is the knob for this path. Measurements:
// report/blockamr-precision-measurements.md in the NeoFOAM repo.

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
            // What the inner Cg<float> is actually handed: the buffers above are sized
            // by THIS rank's rows, so on >1 rank a plain Dense would make the inner
            // solver's dots and norms rank-local (the defect the fp64 Krylov path had).
            // Ginkgo clones its work vectors from these, so the view propagates.
            b32Global_ = makeGlobalVec(this->get_executor(), this->get_size()[0], b32_.get());
            x32Global_ = makeGlobalVec(this->get_executor(), this->get_size()[0], x32_.get());
        }
        {
            // Ginkgo rounds per element on the device: one pass, no host round-trip.
            prof::Timer t("mp.down");
            localView<double>(b)->convert_to(b32_);
        }
        {
            // Zero guess, not the incoming x: this operator IS the correction S(r), so
            // a warm start would add x to itself once the Ir outer loop accumulates.
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
    // Non-owning views of the two above (a distributed::Vector on >1 rank, the buffer
    // itself on one); rebuilt whenever they are.
    mutable std::shared_ptr<gko::LinOp> b32Global_, x32Global_;
};

} // namespace blockamr::la
