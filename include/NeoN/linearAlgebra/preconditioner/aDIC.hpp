// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#if NF_WITH_GINKGO

#include <memory>

#include <ginkgo/ginkgo.hpp>

#include "NeoN/core/executor/executor.hpp"
#include "NeoN/core/vector/vector.hpp"
#include "NeoN/core/primitives/scalar.hpp"
#include "NeoN/core/primitives/label.hpp"
#include "NeoN/linearAlgebra/matrix.hpp"

namespace NeoN::la::ginkgo
{

/** @brief GPU-friendly approximate-DIC preconditioner as a Ginkgo LinOp.
 *
 * Port of OpenFOAM/SPUMA's `aDIC`: the standard DIC preconditioner with its two inherently
 * sequential triangular solves replaced by single, fully parallel Jacobi-style sweeps (scatter
 * with atomics). Unlike Ginkgo's `Ic`, applying the factor needs no exact sparse triangular solve
 * (no per-solve analysis phase, no level scheduling) -- it is a handful of `parallelFor`s, so it
 * stays saturated on the GPU. The kernels run through NeoN's executor abstraction, so the same code
 * serves CPU and GPU.
 *
 * Symmetric matrices only (the OpenFOAM pressure Laplacian): for a structurally symmetric matrix
 * the CSR entry (i, j) with j > i plays the role of the LDU "upper" coefficient, and value symmetry
 * A(i,j) == A(j,i) is assumed. It is the caller's responsibility to hand a positive-definite matrix
 * (the assembled pressure Laplacian is negative-definite; pair this with the GinkgoSolver
 * `negateSystem` flag, as DIC/Ic do).
 *
 * The preconditioner copies the values + sparsity and stores the reciprocal preconditioned diagonal
 * at construction, so it is self-contained and safe to cache/reuse across solves even as the source
 * matrix is overwritten.
 */
class ADICPreconditioner :
    public gko::EnableLinOp<ADICPreconditioner>,
    public gko::EnableCreateMethod<ADICPreconditioner>
{
    friend class gko::EnablePolymorphicObject<ADICPreconditioner, gko::LinOp>;
    friend class gko::EnableCreateMethod<ADICPreconditioner>;

public:

    using value_type = scalar;

    /** @brief Build from a NeoN CSR matrix: copies values + sparsity and computes the reciprocal
     *         preconditioned diagonal. @p gkoExec must match @p exec (see getGkoExecutor). */
    ADICPreconditioner(
        std::shared_ptr<const gko::Executor> gkoExec,
        Executor exec,
        const CSRMatrix<scalar, localIdx>& mtx
    );

protected:

    /** @brief Empty preconditioner; only used by Ginkgo's polymorphic-object machinery. */
    explicit ADICPreconditioner(std::shared_ptr<const gko::Executor> gkoExec);

    void apply_impl(const gko::LinOp* b, gko::LinOp* x) const override;

    void apply_impl(
        const gko::LinOp* alpha, const gko::LinOp* b, const gko::LinOp* beta, gko::LinOp* x
    ) const override;

private:

    /** @brief One parallel Jacobi approximation sweep yielding the reciprocal preconditioned
     *         diagonal rD (port of aDIC's calcReciprocalD). */
    void computeReciprocalD();

    Executor exec_;
    localIdx n_;
    Vector<scalar> values_;       //!< owned copy of the CSR values (frozen)
    Vector<localIdx> colIdx_;     //!< owned copy of the CSR column indices
    Vector<localIdx> rowOffs_;    //!< owned copy of the CSR row offsets
    Vector<scalar> rD_;           //!< reciprocal preconditioned diagonal
    mutable Vector<scalar> work_; //!< apply workspace (avoids per-apply allocation)
};

} // namespace NeoN::la::ginkgo

#endif
