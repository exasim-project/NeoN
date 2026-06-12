// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#if NF_WITH_GINKGO

#include "NeoN/linearAlgebra/preconditioner/aDIC.hpp"
#include "NeoN/core/parallelAlgorithms.hpp"

namespace NeoN::la::ginkgo
{

ADICPreconditioner::ADICPreconditioner(std::shared_ptr<const gko::Executor> gkoExec)
    : gko::EnableLinOp<ADICPreconditioner>(gkoExec), exec_(SerialExecutor {}), n_(0),
      values_(exec_, 0), colIdx_(exec_, 0), rowOffs_(exec_, 0), rD_(exec_, 0), work_(exec_, 0)
{}

ADICPreconditioner::ADICPreconditioner(
    std::shared_ptr<const gko::Executor> gkoExec,
    Executor exec,
    const CSRMatrix<scalar, localIdx>& mtx
)
    : gko::EnableLinOp<ADICPreconditioner>(
        gkoExec,
        gko::dim<2> {
            static_cast<gko::size_type>(mtx.nRows()), static_cast<gko::size_type>(mtx.nRows())
        }
    ),
      exec_(exec), n_(mtx.nRows()), values_(exec, mtx.values()), colIdx_(exec, mtx.colIdxs()),
      rowOffs_(exec, mtx.rowOffs()), rD_(exec, mtx.nRows()), work_(exec, mtx.nRows())
{
    computeReciprocalD();
}

void ADICPreconditioner::computeReciprocalD()
{
    scalar* rD = rD_.data();
    scalar* work = work_.data();
    const scalar* vals = values_.data();
    const localIdx* col = colIdx_.data();
    const localIdx* row = rowOffs_.data();

    // rD = diag, work (= rDtmp) = diag
    parallelFor(
        exec_,
        {0, n_},
        NEON_LAMBDA(const localIdx i) {
            scalar d = 0.0;
            for (localIdx k = row[i]; k < row[i + 1]; ++k)
            {
                if (col[k] == i)
                {
                    d = vals[k];
                    break;
                }
            }
            rD[i] = d;
            work[i] = d;
        },
        "aDIC::initD"
    );

    // Single Jacobi approximation sweep: rDtmp[j] -= A(i,j)^2 / diag[i] for j > i, reading the
    // ORIGINAL diagonal rD[i] (the decoupling that makes this parallel instead of a recurrence).
    parallelFor(
        exec_,
        {0, n_},
        NEON_LAMBDA(const localIdx i) {
            for (localIdx k = row[i]; k < row[i + 1]; ++k)
            {
                const localIdx j = col[k];
                if (j > i)
                {
                    Kokkos::atomic_add(&work[j], -vals[k] * vals[k] / rD[i]);
                }
            }
        },
        "aDIC::calcD"
    );

    // reciprocal of the preconditioned diagonal
    parallelFor(
        exec_, {0, n_}, NEON_LAMBDA(const localIdx i) { rD[i] = 1.0 / work[i]; }, "aDIC::recipD"
    );

    fence(exec_);
}

void ADICPreconditioner::apply_impl(const gko::LinOp* b, gko::LinOp* x) const
{
    using vec = gko::matrix::Dense<scalar>;
    const scalar* bPtr = gko::as<vec>(b)->get_const_values();
    scalar* xPtr = gko::as<vec>(x)->get_values();

    const scalar* rD = rD_.data();
    scalar* work = work_.data();
    const scalar* vals = values_.data();
    const localIdx* col = colIdx_.data();
    const localIdx* row = rowOffs_.data();

    // x = rD * b   (diagonal scaling)
    parallelFor(
        exec_, {0, n_}, NEON_LAMBDA(const localIdx i) { xPtr[i] = rD[i] * bPtr[i]; }, "aDIC::diag"
    );

    // Forward sweep: work = x; work[j] -= rD[j]*A(i,j)*x[i] for j > i; x = work.
    // Reads the (constant) diagonal-scaled x and scatters into work -> atomics on work[j].
    parallelFor(
        exec_, {0, n_}, NEON_LAMBDA(const localIdx i) { work[i] = xPtr[i]; }, "aDIC::fwdInit"
    );
    parallelFor(
        exec_,
        {0, n_},
        NEON_LAMBDA(const localIdx i) {
            for (localIdx k = row[i]; k < row[i + 1]; ++k)
            {
                const localIdx j = col[k];
                if (j > i)
                {
                    Kokkos::atomic_add(&work[j], -rD[j] * vals[k] * xPtr[i]);
                }
            }
        },
        "aDIC::fwd"
    );
    parallelFor(
        exec_, {0, n_}, NEON_LAMBDA(const localIdx i) { xPtr[i] = work[i]; }, "aDIC::fwdStore"
    );

    // Backward sweep: work[i] -= rD[i]*A(i,j)*x[j] for j > i; x = work.
    // Each row writes only work[i] (gather), so no atomics; work already holds the forward result.
    parallelFor(
        exec_,
        {0, n_},
        NEON_LAMBDA(const localIdx i) {
            scalar acc = work[i];
            for (localIdx k = row[i]; k < row[i + 1]; ++k)
            {
                const localIdx j = col[k];
                if (j > i)
                {
                    acc -= rD[i] * vals[k] * xPtr[j];
                }
            }
            work[i] = acc;
        },
        "aDIC::bwd"
    );
    parallelFor(
        exec_, {0, n_}, NEON_LAMBDA(const localIdx i) { xPtr[i] = work[i]; }, "aDIC::bwdStore"
    );

    fence(exec_);
}

void ADICPreconditioner::apply_impl(
    const gko::LinOp* alpha, const gko::LinOp* b, const gko::LinOp* beta, gko::LinOp* x
) const
{
    using vec = gko::matrix::Dense<scalar>;
    auto denseX = gko::as<vec>(x);
    auto tmp = denseX->clone();
    this->apply_impl(b, tmp.get());
    denseX->scale(beta);
    denseX->add_scaled(alpha, tmp);
}

} // namespace NeoN::la::ginkgo

#endif
