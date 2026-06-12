// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#if NF_WITH_GINKGO

#include <cstddef>
#include <memory>

#include <ginkgo/ginkgo.hpp>

#include "NeoN/core/primitives/scalar.hpp"
#include "NeoN/core/primitives/label.hpp"

namespace NeoN::la::ginkgo
{

/** @brief Ginkgo-native aDIC preconditioner.
 *
 * Same approximate-DIC algorithm as ADICPreconditioner, but the kernels run on Ginkgo's own
 * executor/stream instead of through Kokkos. That removes the per-apply Kokkos::fence (a
 * device-wide synchronisation on every Krylov iteration) that the Kokkos version needs because
 * Kokkos and Ginkgo use different CUDA streams.
 *
 * Because a CSR matrix stores both A(i,j) and A(j,i) and aDIC targets symmetric matrices, every
 * sweep is a gather (each row writes only its own entry): no atomics. Symmetric/positive-definite
 * only (pair with the GinkgoSolver negateSystem flag). A frozen deep copy of the matrix is taken at
 * construction so the preconditioner is self-contained and safe to cache/reuse. Single RHS.
 */

// CUDA kernel launchers (defined in adicGinkgoKernels.cpp; real on a CUDA build, no-op otherwise).
// CUstream_st* is Ginkgo's forward-declared CUDA stream type (= CUstream); passing the executor's
// stream keeps the kernels ordered with the rest of the solve, no explicit synchronisation.
void adicGkoGenerateCuda(
    std::size_t n,
    const scalar* vals,
    const localIdx* col,
    const localIdx* row,
    scalar* diag,
    scalar* rd,
    CUstream_st* stream
);
void adicGkoApplyCuda(
    std::size_t n,
    const scalar* vals,
    const localIdx* col,
    const localIdx* row,
    const scalar* rd,
    const scalar* b,
    scalar* x,
    scalar* work,
    CUstream_st* stream
);

namespace detail
{

// rd[i] = 1 / ( diag[i] - sum_{j<i} A(i,j)^2 / diag[j] ). `diag` is a separate scratch holding the
// original diagonal (read-only in pass 2) to avoid the read-after-write hazard of writing rd[i].
inline void adicGkoGenerateHost(
    std::size_t n,
    const scalar* vals,
    const localIdx* col,
    const localIdx* row,
    scalar* diag,
    scalar* rd
)
{
    for (std::size_t i = 0; i < n; ++i)
    {
        scalar d = scalar {0};
        for (auto k = row[i]; k < row[i + 1]; ++k)
        {
            if (static_cast<std::size_t>(col[k]) == i)
            {
                d = vals[k];
                break;
            }
        }
        diag[i] = d;
    }
    for (std::size_t i = 0; i < n; ++i)
    {
        scalar s = diag[i];
        for (auto k = row[i]; k < row[i + 1]; ++k)
        {
            const auto j = static_cast<std::size_t>(col[k]);
            if (j < i)
            {
                s -= vals[k] * vals[k] / diag[j];
            }
        }
        rd[i] = scalar {1} / s;
    }
}

// x = M^{-1} b: diagonal scale, forward gather (lower), backward gather (upper).
inline void adicGkoApplyHost(
    std::size_t n,
    const scalar* vals,
    const localIdx* col,
    const localIdx* row,
    const scalar* rd,
    const scalar* b,
    scalar* x,
    scalar* work
)
{
    for (std::size_t i = 0; i < n; ++i)
    {
        x[i] = rd[i] * b[i];
    }
    for (std::size_t i = 0; i < n; ++i)
    {
        scalar s = x[i];
        for (auto k = row[i]; k < row[i + 1]; ++k)
        {
            const auto j = static_cast<std::size_t>(col[k]);
            if (j < i)
            {
                s -= rd[i] * vals[k] * x[j];
            }
        }
        work[i] = s;
    }
    for (std::size_t i = 0; i < n; ++i)
    {
        scalar s = work[i];
        for (auto k = row[i]; k < row[i + 1]; ++k)
        {
            const auto j = static_cast<std::size_t>(col[k]);
            if (j > i)
            {
                s -= rd[i] * vals[k] * work[j];
            }
        }
        x[i] = s;
    }
}

} // namespace detail


class ADICGinkgoPreconditioner :
    public gko::EnableLinOp<ADICGinkgoPreconditioner>,
    public gko::EnableCreateMethod<ADICGinkgoPreconditioner>
{
    friend class gko::EnablePolymorphicObject<ADICGinkgoPreconditioner, gko::LinOp>;
    friend class gko::EnableCreateMethod<ADICGinkgoPreconditioner>;

public:

    using Csr = gko::matrix::Csr<scalar, localIdx>;
    using Dense = gko::matrix::Dense<scalar>;

    ADICGinkgoPreconditioner(
        std::shared_ptr<const gko::Executor> exec, std::shared_ptr<const Csr> mtx
    )
        : gko::EnableLinOp<ADICGinkgoPreconditioner>(exec, mtx ? mtx->get_size() : gko::dim<2> {}),
          mtx_(mtx ? gko::share(gko::clone(exec, mtx)) : nullptr),
          rd_(exec, mtx ? mtx->get_size()[0] : 0), work_(exec, mtx ? mtx->get_size()[0] : 0)
    {
        if (mtx_)
        {
            generate();
        }
    }

protected:

    explicit ADICGinkgoPreconditioner(std::shared_ptr<const gko::Executor> exec)
        : gko::EnableLinOp<ADICGinkgoPreconditioner>(exec), rd_(exec), work_(exec)
    {}

    void generate()
    {
        const auto n = mtx_->get_size()[0];
        gko::array<scalar> diag {this->get_executor(), n};

        struct generate_op : gko::Operation
        {
            generate_op(
                std::size_t n,
                const scalar* vals,
                const localIdx* col,
                const localIdx* row,
                scalar* diag,
                scalar* rd
            )
                : n(n), vals(vals), col(col), row(row), diag(diag), rd(rd)
            {}

            std::size_t n;
            const scalar* vals;
            const localIdx* col;
            const localIdx* row;
            scalar* diag;
            scalar* rd;

            void run(std::shared_ptr<const gko::ReferenceExecutor>) const override
            {
                detail::adicGkoGenerateHost(n, vals, col, row, diag, rd);
            }
            void run(std::shared_ptr<const gko::OmpExecutor>) const override
            {
                detail::adicGkoGenerateHost(n, vals, col, row, diag, rd);
            }
            void run(std::shared_ptr<const gko::CudaExecutor> exec) const override
            {
                adicGkoGenerateCuda(n, vals, col, row, diag, rd, exec->get_stream());
            }
        };

        this->get_executor()->run(generate_op(
            n,
            mtx_->get_const_values(),
            mtx_->get_const_col_idxs(),
            mtx_->get_const_row_ptrs(),
            diag.get_data(),
            rd_.get_data()
        ));
    }

    void apply_impl(const gko::LinOp* b, gko::LinOp* x) const override
    {
        auto denseB = gko::as<Dense>(b);
        auto denseX = gko::as<Dense>(x);
        const auto n = mtx_->get_size()[0];

        struct apply_op : gko::Operation
        {
            apply_op(
                std::size_t n,
                const scalar* vals,
                const localIdx* col,
                const localIdx* row,
                const scalar* rd,
                const scalar* b,
                scalar* x,
                scalar* work
            )
                : n(n), vals(vals), col(col), row(row), rd(rd), b(b), x(x), work(work)
            {}

            std::size_t n;
            const scalar* vals;
            const localIdx* col;
            const localIdx* row;
            const scalar* rd;
            const scalar* b;
            scalar* x;
            scalar* work;

            void run(std::shared_ptr<const gko::ReferenceExecutor>) const override
            {
                detail::adicGkoApplyHost(n, vals, col, row, rd, b, x, work);
            }
            void run(std::shared_ptr<const gko::OmpExecutor>) const override
            {
                detail::adicGkoApplyHost(n, vals, col, row, rd, b, x, work);
            }
            void run(std::shared_ptr<const gko::CudaExecutor> exec) const override
            {
                adicGkoApplyCuda(n, vals, col, row, rd, b, x, work, exec->get_stream());
            }
        };

        this->get_executor()->run(apply_op(
            n,
            mtx_->get_const_values(),
            mtx_->get_const_col_idxs(),
            mtx_->get_const_row_ptrs(),
            rd_.get_const_data(),
            denseB->get_const_values(),
            denseX->get_values(),
            work_.get_data()
        ));
    }

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

private:

    std::shared_ptr<const Csr> mtx_;  // frozen deep copy of the system matrix
    gko::array<scalar> rd_;           // reciprocal preconditioned diagonal
    mutable gko::array<scalar> work_; // apply scratch (forward-sweep result)
};

} // namespace NeoN::la::ginkgo

#endif
