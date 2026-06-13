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
 * sweep is a gather (each row writes only its own entry): no atomics. The frozen deep copy of the
 * matrix is column-sorted at construction and the diagonal's per-row CSR index is cached, so the
 * forward (lower) and backward (upper) sweeps each iterate only their half of the row -- half the
 * memory traffic and no per-entry triangle branch, the LDU split OpenFOAM gets for free.
 * Symmetric/positive-definite only (pair with the GinkgoSolver negateSystem flag). The deep copy
 * makes the preconditioner self-contained and safe to cache/reuse. Single RHS.
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
    localIdx* diagPos,
    CUstream_st* stream
);
void adicGkoApplyCuda(
    std::size_t n,
    const scalar* vals,
    const localIdx* col,
    const localIdx* row,
    const localIdx* diagPos,
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
// Columns are sorted ascending (see the constructor), so diagPos[i] is the index of the diagonal
// entry and splits each row into its lower part [row[i], diagPos[i]) and upper part
// [diagPos[i]+1, row[i+1]); the lower-sum loop then visits only the j<i entries with no branch.
inline void adicGkoGenerateHost(
    std::size_t n,
    const scalar* vals,
    const localIdx* col,
    const localIdx* row,
    scalar* diag,
    scalar* rd,
    localIdx* diagPos
)
{
    for (std::size_t i = 0; i < n; ++i)
    {
        auto dp = row[i];
        scalar d = scalar {0};
        for (auto k = row[i]; k < row[i + 1]; ++k)
        {
            if (static_cast<std::size_t>(col[k]) == i)
            {
                dp = k;
                d = vals[k];
                break;
            }
        }
        diagPos[i] = dp;
        diag[i] = d;
    }
    for (std::size_t i = 0; i < n; ++i)
    {
        scalar s = diag[i];
        for (auto k = row[i]; k < diagPos[i]; ++k)
        {
            const auto j = static_cast<std::size_t>(col[k]);
            s -= vals[k] * vals[k] / diag[j];
        }
        rd[i] = scalar {1} / s;
    }
}

// x = M^{-1} b: diagonal scale, forward gather (lower), backward gather (upper). With sorted
// columns and the per-row diagonal index diagPos, each sweep visits only its half of the row -- the
// lower part [row[i], diagPos[i]) or the upper part [diagPos[i]+1, row[i+1]) -- so it touches half
// the matrix entries and needs no per-entry j<i / j>i branch.
inline void adicGkoApplyHost(
    std::size_t n,
    const scalar* vals,
    const localIdx* col,
    const localIdx* row,
    const localIdx* diagPos,
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
        for (auto k = row[i]; k < diagPos[i]; ++k)
        {
            s -= rd[i] * vals[k] * x[static_cast<std::size_t>(col[k])];
        }
        work[i] = s;
    }
    for (std::size_t i = 0; i < n; ++i)
    {
        scalar s = work[i];
        for (auto k = diagPos[i] + 1; k < row[i + 1]; ++k)
        {
            s -= rd[i] * vals[k] * work[static_cast<std::size_t>(col[k])];
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
          rd_(exec, mtx ? mtx->get_size()[0] : 0), diagPos_(exec, mtx ? mtx->get_size()[0] : 0),
          work_(exec, mtx ? mtx->get_size()[0] : 0)
    {
        if (mtx)
        {
            // The split sweeps need each row's columns in ascending order so the lower part
            // [row[i], diagPos[i]) and upper part [diagPos[i]+1, row[i+1]) are contiguous. Sort the
            // private deep copy (gko::clone strips const, so it is mutable); never the caller's
            // mtx.
            auto sorted = gko::clone(exec, mtx);
            sorted->sort_by_column_index();
            mtx_ = gko::share(std::move(sorted));
            generate();
        }
    }

protected:

    explicit ADICGinkgoPreconditioner(std::shared_ptr<const gko::Executor> exec)
        : gko::EnableLinOp<ADICGinkgoPreconditioner>(exec), rd_(exec), diagPos_(exec), work_(exec)
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
                scalar* rd,
                localIdx* diagPos
            )
                : n(n), vals(vals), col(col), row(row), diag(diag), rd(rd), diagPos(diagPos)
            {}

            std::size_t n;
            const scalar* vals;
            const localIdx* col;
            const localIdx* row;
            scalar* diag;
            scalar* rd;
            localIdx* diagPos;

            void run(std::shared_ptr<const gko::ReferenceExecutor>) const override
            {
                detail::adicGkoGenerateHost(n, vals, col, row, diag, rd, diagPos);
            }
            void run(std::shared_ptr<const gko::OmpExecutor>) const override
            {
                detail::adicGkoGenerateHost(n, vals, col, row, diag, rd, diagPos);
            }
            void run(std::shared_ptr<const gko::CudaExecutor> exec) const override
            {
                adicGkoGenerateCuda(n, vals, col, row, diag, rd, diagPos, exec->get_stream());
            }
        };

        this->get_executor()->run(generate_op(
            n,
            mtx_->get_const_values(),
            mtx_->get_const_col_idxs(),
            mtx_->get_const_row_ptrs(),
            diag.get_data(),
            rd_.get_data(),
            diagPos_.get_data()
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
                const localIdx* diagPos,
                const scalar* rd,
                const scalar* b,
                scalar* x,
                scalar* work
            )
                : n(n), vals(vals), col(col), row(row), diagPos(diagPos), rd(rd), b(b), x(x),
                  work(work)
            {}

            std::size_t n;
            const scalar* vals;
            const localIdx* col;
            const localIdx* row;
            const localIdx* diagPos;
            const scalar* rd;
            const scalar* b;
            scalar* x;
            scalar* work;

            void run(std::shared_ptr<const gko::ReferenceExecutor>) const override
            {
                detail::adicGkoApplyHost(n, vals, col, row, diagPos, rd, b, x, work);
            }
            void run(std::shared_ptr<const gko::OmpExecutor>) const override
            {
                detail::adicGkoApplyHost(n, vals, col, row, diagPos, rd, b, x, work);
            }
            void run(std::shared_ptr<const gko::CudaExecutor> exec) const override
            {
                adicGkoApplyCuda(n, vals, col, row, diagPos, rd, b, x, work, exec->get_stream());
            }
        };

        this->get_executor()->run(apply_op(
            n,
            mtx_->get_const_values(),
            mtx_->get_const_col_idxs(),
            mtx_->get_const_row_ptrs(),
            diagPos_.get_const_data(),
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

    std::shared_ptr<const Csr> mtx_;  // frozen deep copy of the system matrix (columns sorted)
    gko::array<scalar> rd_;           // reciprocal preconditioned diagonal
    gko::array<localIdx> diagPos_;    // per-row CSR index of the diagonal entry (splits L/U)
    mutable gko::array<scalar> work_; // apply scratch (forward-sweep result)
};

} // namespace NeoN::la::ginkgo

#endif
