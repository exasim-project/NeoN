// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <ginkgo/ginkgo.hpp>

#include <memory>
#include <utility>

#include "linop_base.hpp"
#include "profiling.hpp"
#include "types.hpp"

// ---------------------------------------------------------------------------
// An FP32 solver wearing an FP64 LinOp's clothes, so that gko::solver::Ir<double>
// can drive it.
//
// WHY. The profile of the tuned 256^3 solve puts 74.2 ms of 188.9 on the Krylov
// side, and every kernel there is at 83-100% of the machine's measured 479 GB/s.
// There is no kernel-quality headroom left; the only lever is bytes. Every one of
// those vectors is fp64:
//
//     cg::step_1 / cg::step_2   25.6 ms    p = r + beta p; x += alpha p; r -= alpha q
//     matvec (FaceCoeffOp)      17.7 ms
//     dot / nrm2 (cuBLAS)       15.1 ms
//     pack/unpack Ginkgo<->AMReX 9.4 ms    the V-cycle interface
//     linf stopping norm         6.5 ms
//
// Halving the element width halves the traffic of all of it. What it cannot do is
// halve the ACCURACY the caller asked for: fp32 CG stagnates once the residual
// approaches fp32's rounding of A x, around 1e-7 relative, and the tolerances here
// go to 1e-10 and below.
//
// Iterative refinement is the standard resolution and the reason this class exists.
// The outer loop keeps everything that decides the ANSWER in fp64 --
//
//     r = b - A x        (fp64 operator, fp64 residual, fp64 stopping test)
//     x <- x + S(r)      (S approximate, precision irrelevant to the fixed point)
//
// -- and S is where the time goes, so S is what gets narrowed. A wrong S costs
// outer iterations; it cannot cost accuracy, exactly as a preconditioner cannot.
//
// WHAT THIS CLASS IS. gko::solver::Ir<double> calls its inner solver through a
// gko::LinOp with Dense<double> arguments. This is that LinOp: it converts b down
// to fp32, runs a preconditioned Cg<float> from a zero guess, and converts the
// result back up. The two conversions are 3 * n * 4 bytes per apply -- 0.2 ms at
// 256^3 against the ~9 ms per outer iteration they buy back.
//
// THE INNER TOLERANCE IS THE WHOLE DESIGN. Solving the inner system tightly wastes
// fp32 iterations on digits the outer loop is about to recompute; solving it too
// loosely makes the outer loop a Richardson iteration. The default (1e-2, i.e.
// two digits per outer step) comes from the standard analysis -- the outer
// contraction factor IS the inner tolerance -- and is a knob because the right
// value depends on how much the V-cycle already contracts.
//
// WHAT IT MEASURED: THE BYTES ARRIVE, THE VEHICLE LOSES THEM
//
// A NEGATIVE RESULT, kept wired and tested for the same reason bf16.hpp's is. The
// answers are right -- every configuration below converges and agrees with the fp64
// CG's solution to ~1e-14 -- and it is slower than the fp64 CG at every setting.
//
// 256^3, one box, fp32 hierarchy, varying b, rtol 1e-10 in linf. `applies` counts
// PRECONDITIONER applies, which is the unit of work here (the V-cycle is 58% of a
// solve): Ginkgo's Cg applies the preconditioner before its stopping check, so a
// solve reported as k iterations performed k+1 of them. The inner tolerance is set
// unreachable so mp_inner_max_iter = K is the exact inner count, which makes the
// column exact rather than inferred:
//
//     config        outer   K   applies    ms    ms/apply
//     cg fp64         9     --     10     213.8    21.4
//     mpir K=1       13      1     26     535.1    20.6
//     mpir K=2        5      2     15     302.4    20.2
//     mpir K=3        5      3     20     384.8    19.2
//     mpir K=4        5      4     25     468.8    18.8
//     mpir K=6        5      6     35     644.2    18.4
//     mpir K=8        5      8     45     815.6    18.1
//
// Read the last column first: the fp32 inner iteration really is cheaper, 18.1 ms
// against 21.4, i.e. 1.18x -- essentially the 1.20x the profile predicted for
// halving the Krylov width, arriving as predicted. (The trend from 20.6 to 18.1 is
// the per-outer-step fp64 residual amortising over more inner iterations.)
//
// Then read the `applies` column: the cheapest refinement schedule needs 15 where CG
// needed 10. A restart is not free -- it re-pays the initial residual AND the
// pre-check preconditioner apply, and it discards the Krylov space, which is why
// K=1 needs 13 outer steps where CG needed 9 iterations. 1.5x the work at 0.85x the
// unit cost is 1.27x slower, and measured it is 1.41x.
//
// So refinement cannot cash a 1.18x saving that costs 1.5x more applies. What would
// is a Krylov method that runs its recurrence in fp32 WITHOUT restarting, keeping
// only the solution update in fp64. That is a different algorithm, not a different
// setting, and Ginkgo does not offer it.
//
// Two things ruled out along the way, so they are not re-litigated:
//   * The over-relaxed smoother (gmg_omega=1.1) breaks the V-cycle's
//     self-adjointness, which fp32 CG might have tolerated less well than fp64's.
//     It is not the cause: at omega=1.0 the fp32 floor is unchanged (0.1239 against
//     0.1226) and mpir is worse still (415 ms at K=2 against 302).
//   * "Just run fp32 CG to the tolerance" is not an option, which is why refinement
//     was the right idea: one fp32 solve alone stops at a linf residual of 0.123
//     where the fp64 CG reaches 3.3e-10.
//
// One loose end, flagged rather than fixed because no conclusion here rests on it:
// that lone fp32 solve stops itself after ~12 iterations, far above fp32's rounding
// floor, so the INNER stopping criterion is not trustworthy. It does not affect the
// table above, whose inner counts are fixed by iteration cap with the tolerance
// switched off -- but it does mean mp_inner_max_iter, not mp_inner_rtol, is the knob
// to drive this path with.
// ---------------------------------------------------------------------------

namespace blockamr::solvers
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
            // What the inner Cg<float> is actually handed. The buffers above are
            // sized by THIS rank's rows, so on >1 rank a plain Dense would make the
            // inner solver's dots and norms rank-local -- the same defect the fp64
            // Krylov path had. Ginkgo clones its work vectors from these, so the
            // distributed view propagates through the whole inner solve.
            b32Global_ = makeGlobalVec(this->get_executor(), this->get_size()[0], b32_.get());
            x32Global_ = makeGlobalVec(this->get_executor(), this->get_size()[0], x32_.get());
        }
        {
            // Dense::convert_to is the narrowing copy; Ginkgo rounds per element on
            // the device, so this is one pass and no host round-trip.
            prof::Timer t("mp.down");
            localView<double>(b)->convert_to(b32_);
        }
        {
            // Zero guess, not the incoming x: this operator IS the correction
            // S(r), so a warm start would add x to itself once the Ir outer loop
            // accumulates the result.
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
    // Same shared_ptr-not-unique_ptr reason as AmrexLinOpBase::scratch_: Ginkgo
    // gives these operators a copy-assignment, which a move-only member deletes.
    mutable std::shared_ptr<Dense32> b32_, x32_;
    // Non-owning views of the two above: the buffer on one rank, a
    // distributed::Vector over it on several. Rebuilt whenever they are.
    mutable std::shared_ptr<gko::LinOp> b32Global_, x32Global_;
};

} // namespace blockamr::solvers
