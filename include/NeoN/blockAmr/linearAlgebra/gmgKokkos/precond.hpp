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

// The optimised Kokkos V-cycle as a Ginkgo preconditioner: a gko::LinOp whose apply hands two
// device pointers to blockamr::KokkosGmgApply. Why it sits beside GmgPrecondT, and what it does
// not carry: report/blockamr-gmg-notes.md#kokkos-handle.

namespace blockamr::la
{

// V is the value type of the Krylov vectors -- double normally, float inside the
// mixed-precision refinement. Independent of the HIERARCHY's type; KokkosGmgApply converts.
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
        // shared_ptr<T> in a const method still yields a non-const T*, so no cast is needed.
        vcycle_->apply(localValues<V>(b), localValues<V>(x));
    }

private:

    // shared_ptr like AmrexLinOpBase::scratch_: Ginkgo copy-assigns these operators.
    std::shared_ptr<blockamr::KokkosGmgApply> vcycle_;
};

using GmgKokkosPrecond = GmgKokkosPrecondT<double>;
using GmgKokkosPrecond32 = GmgKokkosPrecondT<float>;

} // namespace blockamr::la
