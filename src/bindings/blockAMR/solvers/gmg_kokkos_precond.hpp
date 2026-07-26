// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <ginkgo/ginkgo.hpp>

#include <memory>
#include <stdexcept>
#include <utility>

#include "../bench/gmg_apply.hpp"
#include "linop_base.hpp"
#include "profiling.hpp"
#include "types.hpp"

// ---------------------------------------------------------------------------
// The optimised Kokkos V-cycle as a Ginkgo preconditioner.
//
// This is the whole Ginkgo side of it: a gko::LinOp whose apply hands two device
// pointers to bench::KokkosGmgApply. All the multigrid lives on the other side of
// that handle, in the non-RDC object library where the Kokkos kernels compile (see
// bench/gmg_apply.hpp for why the fence exists).
//
// It sits beside GmgPrecondT rather than inside it. GmgPrecondT is the shipped
// preconditioner and the baseline every measurement is read against, so it is left
// untouched; `precond="gmg"` and `precond="gmg_kokkos"` are two independent objects
// and bench_solvers.py can run both in one process and compare them.
//
// What this one does NOT carry, because the ported V-cycle does not: physical
// boundary conditions (periodic only), the Chebyshev smoother, and the host
// (ReferenceExecutor) path. Each is rejected at construction rather than silently
// ignored.
// ---------------------------------------------------------------------------

namespace blockamr::solvers
{

// V is the value type of the Krylov vectors this preconditioner is applied to --
// double for the ordinary solvers, float inside the mixed-precision refinement. It
// is independent of the HIERARCHY's storage type: KokkosGmgApply converts on the
// way in and out, so an fp32 Krylov can drive an fp64 hierarchy and vice versa.
template<class V>
class GmgKokkosPrecondT : public AmrexLinOpBase<GmgKokkosPrecondT<V>, V>
{
public:

    // Required by Ginkgo's polymorphic-object machinery (create_default / clear).
    explicit GmgKokkosPrecondT(std::shared_ptr<const gko::Executor> exec)
        : AmrexLinOpBase<GmgKokkosPrecondT<V>, V>(exec)
    {}

    GmgKokkosPrecondT(
        std::shared_ptr<const gko::Executor> exec,
        gko::size_type n,
        std::shared_ptr<bench::KokkosGmgApply> vcycle
    )
        : AmrexLinOpBase<GmgKokkosPrecondT<V>, V>(exec, gko::dim<2> {n, n}),
          vcycle_(std::move(vcycle))
    {
        if (exec->get_master().get() == exec.get())
        {
            throw std::runtime_error(
                "GmgKokkosPrecond: the Kokkos V-cycle is a device path; use executor='cuda'"
            );
        }
    }

    [[nodiscard]] int nlevels() const { return vcycle_ ? vcycle_->nlevels() : 0; }

protected:

    // Keeps the base's advanced apply_impl(alpha, b, beta, x) visible in this scope.
    using AmrexLinOpBase<GmgKokkosPrecondT<V>, V>::apply_impl;

    void apply_impl(const gko::LinOp* b, gko::LinOp* x) const override
    {
        prof::Timer tAll("gmgk.apply");
        {
            prof::Timer t("gmgk.sync_gko");
            this->get_executor()->synchronize(); // b written by Ginkgo
        }
        prof::Timer t("gmgk.vcycle");
        // shared_ptr<T> in a const method still yields a non-const T*, so the
        // handle's mutating apply is reachable without a cast.
        using DenseV = gko::matrix::Dense<V>;
        vcycle_->apply(gko::as<DenseV>(b)->get_const_values(), gko::as<DenseV>(x)->get_values());
    }

private:

    // shared_ptr for the same reason as AmrexLinOpBase::scratch_: Ginkgo gives these
    // operators a copy-assignment, which a move-only member would delete.
    std::shared_ptr<bench::KokkosGmgApply> vcycle_;
};

using GmgKokkosPrecond = GmgKokkosPrecondT<double>;
using GmgKokkosPrecond32 = GmgKokkosPrecondT<float>;

} // namespace blockamr::solvers
