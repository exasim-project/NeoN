// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <ginkgo/ginkgo.hpp>

#include <memory>
#include <stdexcept>
#include <utility>

#include "NeoN/blockAmr/linearAlgebra/matrixFree/linOpBase.hpp"
#include "NeoN/blockAmr/core/profiling.hpp"
#include "NeoN/blockAmr/core/types.hpp"
#include "NeoN/blockAmr/linearAlgebra/gmgKokkos/apply.hpp"

// The optimised Kokkos V-cycle as a Ginkgo preconditioner: a gko::LinOp whose apply
// hands two device pointers to blockamr::KokkosGmgApply. All the multigrid lives behind
// that handle, in the non-RDC object library where the Kokkos kernels compile
// (apply.hpp).
//
// It sits beside GmgPrecondT rather than inside it, because GmgPrecondT is the shipped
// preconditioner and the baseline every measurement is read against: `precond="gmg"`
// and `precond="gmg_kokkos"` are independent objects, so bench_solvers.py can run both
// in one process and compare them.
//
// What this one does NOT carry, because the ported V-cycle does not: the Chebyshev
// smoother, and the host (ReferenceExecutor) path -- rejected below rather than
// ignored.

namespace blockamr::la
{

// V is the value type of the Krylov vectors -- double for the ordinary solvers, float
// inside the mixed-precision refinement. Independent of the HIERARCHY's storage type:
// KokkosGmgApply converts on the way in and out, so either can drive the other.
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
        std::shared_ptr<blockamr::KokkosGmgApply> vcycle
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
        // shared_ptr<T> in a const method still yields a non-const T*, so the handle's
        // mutating apply is reachable without a cast.
        vcycle_->apply(localValues<V>(b), localValues<V>(x));
    }

private:

    // shared_ptr like AmrexLinOpBase::scratch_: Ginkgo copy-assigns these operators,
    // which a move-only member would delete.
    std::shared_ptr<blockamr::KokkosGmgApply> vcycle_;
};

using GmgKokkosPrecond = GmgKokkosPrecondT<double>;
using GmgKokkosPrecond32 = GmgKokkosPrecondT<float>;

} // namespace blockamr::la
