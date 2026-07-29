// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>

#include <ginkgo/ginkgo.hpp>

#include <memory>
#include <vector>

#include "NeoN/blockAmr/core/bc.hpp"
#include "NeoN/blockAmr/linearAlgebra/matrixFree/linOpBase.hpp"
#include "NeoN/blockAmr/core/types.hpp"
#include "NeoN/core/executor/executor.hpp"

namespace blockamr::la
{

// The fine-level matrix diagonal, alpha - (aE+aW+aN+aS+aT+aB), written per valid
// cell of `diag`. This is the negSumDiag convention the stencils used to evaluate
// inline; S7 computes it once instead. The association order is exactly the one
// the two stencils used, so the stored value is bitwise what the derivation
// produced. `diag` is cell-centred with alpha's BoxArray/DistributionMapping.
//
// It takes NO BcArray, deliberately: domain BCs enter the mat-vec through the
// ghost reflection, i.e. through the OFF-diagonal term, so alpha - sum(faces) is
// BC-independent and storing it is arithmetically identical to deriving it.
void computeFaceCoeffDiag(
    const NeoN::Executor& exec,
    amrex::MultiFab& diag,
    const amrex::MultiFab& alpha,
    const amrex::MultiFab& ux,
    const amrex::MultiFab& lx,
    const amrex::MultiFab& uy,
    const amrex::MultiFab& ly,
    const amrex::MultiFab& uz,
    const amrex::MultiFab& lz
);

// General matrix-free face-coefficient operator on a structured single-level grid. The
// matrix is carried as OpenFOAM-style pieces given as AMReX fields:
//   alpha  : cell-centred diagonal SOURCE (ddt/Sp/reaction), NOT the full
//            diagonal — the face part is folded in by computeFaceCoeffDiag above.
//   u{x,y,z}, l{x,y,z} : face-centred upper/lower off-diagonal coefficients.
//             u* is the owner-row->neighbour coupling on the cell's HIGH face,
//            l* the neighbour-row->owner coupling on the cell's LOW face. For a
//            symmetric matrix pass the same MultiFab for u* and l*.
// The mat-vec is the OpenFOAM Amul in pull form (each cell reads its 6
// neighbours), against the STORED diagonal
//   diag = alpha - (aE+aW+aN+aS+aT+aB)               (negSumDiag)
// which the constructor computes once (or takes from the caller) rather than
// re-deriving per cell per apply. The face coeffs still feed both the
// off-diagonal and the diagonal. This is exact whenever the flux part
// annihilates a constant (divergence-free flux / pure diffusion); any
// non-conservative diagonal contribution must be folded into alpha.
//
// V is the value type of the flat Ginkgo vectors it is applied to -- double for
// the fp64 Krylov solvers, float for the inner solve of the mixed-precision
// refinement. The COEFFICIENTS are amrex::MultiFab either way, so an fp32
// instantiation narrows the Krylov vectors and nothing else; the operator it
// applies is the same one, evaluated in V. V = float is a DEVICE path only (the
// host stencil below stays double), which the constructor rejects rather than
// silently downgrades.
template<class V>
class FaceCoeffOpT : public AmrexLinOpBase<FaceCoeffOpT<V>, V>
{
public:

    explicit FaceCoeffOpT(std::shared_ptr<const gko::Executor> exec);

    // nexec is the NeoN executor the kernel launches run under; it is carried
    // alongside the Ginkgo executor rather than derived from it, because the launch
    // seam (blockamr::parallelFor) dispatches on the NeoN variant.
    FaceCoeffOpT(
        std::shared_ptr<const gko::Executor> exec,
        const NeoN::Executor& nexec,
        const amrex::BoxArray& ba,
        const amrex::DistributionMapping& dm,
        amrex::Geometry geom,
        gko::size_type n,
        const amrex::MultiFab* alpha,
        const amrex::MultiFab* ux,
        const amrex::MultiFab* lx,
        const amrex::MultiFab* uy,
        const amrex::MultiFab* ly,
        const amrex::MultiFab* uz,
        const amrex::MultiFab* lz,
        BcArray bc = {},
        const amrex::MultiFab* bcData = nullptr,
        // The stored fine-level diagonal, already computed by the caller. null
        // (every legacy call site) means "compute it here, once, from the
        // coefficients as handed in" — see the staleness note on owned_ below.
        // blockamr::la::MFFaceCoeffs owns one and passes it, so it survives the
        // per-solve rebuild of this operator.
        const amrex::MultiFab* diag = nullptr
    );

    // c0 = L(0), the constant offset that inhomogeneous domain BCs add to the
    // otherwise linear operator: the same stencil, applied to `zero` (a zero
    // vector the caller supplies as scratch) with the INHOMOGENEOUS ghost fill
    // instead of the reflecting one. `out` receives it.
    //
    // This is what keeps `apply` linear. With bc_data set the boundary operator
    // is affine, L(x) = A x + c0, and Ginkgo's Krylov solvers are entitled to
    // assume linearity; so the solve runs on A x = rhs - c0 with `apply` still
    // computing A alone, and c0 folded into the right-hand side once per solve.
    // Requires a bc_data operator; throws otherwise.
    void applyBcOffset(const gko::LinOp* zero, gko::LinOp* out) const;

protected:

    // Keeps the base's advanced apply_impl(alpha, b, beta, x) visible in this
    // scope (the declaration below would otherwise hide it).
    using AmrexLinOpBase<FaceCoeffOpT<V>, V>::apply_impl;

    void apply_impl(const gko::LinOp* b, gko::LinOp* x) const override;

private:

    // The shared body of apply_impl and applyBcOffset; `inhom` picks which
    // domain-BC ghost fill runs, and nothing else differs between them.
    void applyWith(const gko::LinOp* b, gko::LinOp* x, bool inhom) const;

    amrex::Geometry geom_;
    // The executor the stencil launch is dispatched on. Defaulted for the
    // exec-only constructor Ginkgo's create_default/clear uses, which builds an
    // operator with no fields to launch over.
    NeoN::Executor nexec_ {NeoN::SerialExecutor {}};
    BcArray bc_ {};
    bool hasPhysBc_ = false;
    bool onDevice_ = false;
    // Host path: owns pinned copies of the coefficient fields. Device path:
    // empty, and the pointers below reference the caller's device-resident
    // fields directly.
    //
    // On the device path that used to mean an external in-place update to the
    // coefficients was picked up by the next apply with no reassembly. Since S7
    // it does NOT: diag_ below is computed once, so after an in-place write to
    // alpha or a face field this operator's diagonal is stale until a new
    // operator is built over the same fields. blockamr::la::MFFaceCoeffs is where
    // that is handled — it owns the diagonal, refreshes it when the coefficient
    // handles are handed out, and builds a fresh operator per solve. A caller
    // constructing a FaceCoeffOp directly (FaceCoeffSolver, solveFaceCoeffs) must
    // reconstruct it after changing coefficients. bcData_ is unaffected: it is
    // still referenced and still picked up in place.
    std::vector<std::shared_ptr<amrex::MultiFab>> owned_;
    const amrex::MultiFab* ux_ = nullptr;
    const amrex::MultiFab* lx_ = nullptr;
    const amrex::MultiFab* uy_ = nullptr;
    const amrex::MultiFab* ly_ = nullptr;
    const amrex::MultiFab* uz_ = nullptr;
    const amrex::MultiFab* lz_ = nullptr;
    // The stored fine-level diagonal the stencils read. Points either at the
    // caller's field or at diagOwned_/a pinned copy in owned_; never null on an
    // operator built with fields.
    const amrex::MultiFab* diag_ = nullptr;
    // Holds the diagonal this operator computed for itself (the null-`diag`
    // constructor argument), on the device path. On the host path the pinned copy
    // in owned_ is what the stencil reads and this is released.
    std::shared_ptr<amrex::MultiFab> diagOwned_;
    // Inhomogeneous domain-BC data (null = homogeneous, the default and the
    // historical behaviour); staged to pinned memory in owned_ on the host path
    // exactly like the coefficients. dx_ scales the Neumann datum.
    const amrex::MultiFab* bcData_ = nullptr;
    amrex::Real dx_[3] {};
    std::shared_ptr<amrex::MultiFab> in_;
    std::shared_ptr<amrex::MultiFab> out_;
};

// The fp64 operator every existing caller means by "FaceCoeffOp", and its fp32
// twin. Both are explicitly instantiated in faceCoeffOp.cpp.
using FaceCoeffOp = FaceCoeffOpT<double>;
using FaceCoeffOp32 = FaceCoeffOpT<float>;

} // namespace blockamr::la
