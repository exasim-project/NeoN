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
#include "NeoN/blockAmr/core/fieldLevel.hpp"
#include "NeoN/blockAmr/core/meshLevel.hpp"
#include "NeoN/blockAmr/linearAlgebra/matrixFree/linOpBase.hpp"
#include "NeoN/blockAmr/core/types.hpp"
#include "NeoN/core/executor/executor.hpp"

namespace blockamr::la
{

// The fine-level diagonal, diag = alpha - (aE+aW+aN+aS+aT+aB) (negSumDiag), per
// valid cell of `diag` (cell-centred, alpha's BoxArray/DistributionMapping), in
// exactly the association order the stencils use — so the stored value is bitwise
// what an inline derivation gives. No BcArray, deliberately: domain BCs enter the
// mat-vec through the ghost reflection, i.e. the OFF-diagonal term, so this is
// BC-independent. `diag` by value and the coefficients by const& is the signature
// saying which of them this writes (core/fieldLevel.hpp).
void computeFaceCoeffDiag(
    const NeoN::Executor& exec,
    CellFieldLevel diag,
    const CellFieldLevel& alpha,
    const FaceFieldLevel& upper,
    const FaceFieldLevel& lower
);

// General matrix-free face-coefficient operator on a structured single-level grid,
// its matrix carried as OpenFOAM-style pieces given as AMReX fields:
//   alpha  : cell-centred diagonal SOURCE (ddt/Sp/reaction), NOT the full
//            diagonal — the face part is the negSumDiag term above.
//   upper, lower : the three face-centred off-diagonal direction fields each, one
//            FaceFieldLevel each (core/fieldLevel.hpp). upper[d] is the
//            owner-row->neighbour coupling on the cell's HIGH face in direction d,
//            lower[d] the neighbour-row->owner coupling on its LOW face; for a
//            symmetric matrix pass the same fields for both.
// The mat-vec is the OpenFOAM Amul in pull form (each cell reads its 6 neighbours)
// against diag = alpha - (aE+aW+aN+aS+aT+aB), so the face coeffs feed both the
// off-diagonal and the diagonal. Exact whenever the flux part annihilates a
// constant (divergence-free flux / pure diffusion); any non-conservative diagonal
// contribution must be folded into alpha. Homogeneous domain BCs are applied on
// every apply by the ghost reflection, hence through the off-diagonal term alone
// (the operator's half of that contract: operators/laplacian.hpp).
//
// PROTOTYPE (C1): faceCoeffOp.cpp currently BYPASSES the stored diagonal — the
// `diag` constructor argument is ignored and both stencils recompute
// alpha - sum(faces) inline, in that same association order.
//
// V is the value type of the flat Ginkgo vectors only (double for the fp64 Krylov
// solvers, float for the mixed-precision inner solve); the COEFFICIENTS stay
// amrex::MultiFab, so an fp32 instantiation narrows the vectors and nothing else.
// V = float is a DEVICE path only (the host stencil stays double), which the
// constructor rejects rather than silently downgrades.
template<class V>
class FaceCoeffOpT : public AmrexLinOpBase<FaceCoeffOpT<V>, V>
{
public:

    explicit FaceCoeffOpT(std::shared_ptr<const gko::Executor> exec);

    // nexec is carried alongside the Ginkgo executor rather than derived from it
    // because the launch seam (blockamr::parallelFor) dispatches on the NeoN variant.
    FaceCoeffOpT(
        std::shared_ptr<const gko::Executor> exec,
        const NeoN::Executor& nexec,
        // Allocation layout for the scratch/in-out fields plus the geometry the
        // stencil's dx and ghost fill come from (core/meshLevel.hpp). Only `geom`
        // outlives the constructor, hence geom_ below rather than a whole MeshLevel.
        const MeshLevel& mesh,
        gko::size_type n,
        // const&: a by-value handle would give this constructor write access to the
        // caller's coefficients.
        const CellFieldLevel& alpha,
        const FaceFieldLevel& upper,
        const FaceFieldLevel& lower,
        BcArray bc = {},
        // A bare pointer: read-only ghost-fill data whose source
        // (SolverConfig::bcData) is a const amrex::MultiFab*, so there is no mutable
        // handle to build a CellFieldLevel from.
        const amrex::MultiFab* bcData = nullptr,
        // The caller's stored fine-level diagonal — blockamr::la::MFFaceCoeffs owns
        // one, so it survives the per-solve rebuild of this operator; an empty handle
        // (every legacy call site) means compute it here, once. Ignored while the
        // PROTOTYPE (C1) path above is live.
        const CellFieldLevel& diag = {}
    );

    // c0 = L(0) into `out`: the constant inhomogeneous domain BCs add, i.e. the same
    // stencil applied to a caller-supplied zero vector with the INHOMOGENEOUS ghost
    // fill instead of the reflecting one. This is what keeps `apply` linear as
    // Ginkgo's Krylov solvers assume — with bc_data the operator is affine,
    // L(x) = A x + c0, so the solve runs A x = rhs - c0 with c0 folded into the
    // right-hand side once per solve. Requires a bc_data operator; throws otherwise.
    void applyBcOffset(const gko::LinOp* zero, gko::LinOp* out) const;

protected:

    // Keeps the base's advanced apply_impl(alpha, b, beta, x) from being hidden.
    using AmrexLinOpBase<FaceCoeffOpT<V>, V>::apply_impl;

    void apply_impl(const gko::LinOp* b, gko::LinOp* x) const override;

private:

    // Shared body of apply_impl and applyBcOffset; `inhom` picks the domain-BC ghost
    // fill and nothing else differs between them.
    void applyWith(const gko::LinOp* b, gko::LinOp* x, bool inhom) const;

    amrex::Geometry geom_;
    // Stencil-launch executor. Defaulted for the exec-only constructor used by
    // create_default/clear, which builds an operator with no fields to launch over.
    NeoN::Executor nexec_ {NeoN::SerialExecutor {}};
    BcArray bc_ {};
    bool hasPhysBc_ = false;
    bool onDevice_ = false;
    // Host path: owns pinned copies of the coefficient fields, so a caller's
    // in-place write is NOT observed until a new operator is built. Device path:
    // empty, and the pointers below reference the caller's device-resident fields
    // directly, so an in-place update IS picked up by the next apply — but only
    // while the diagonal is recomputed inline (PROTOTYPE C1 above); with a stored
    // diagonal, a write to alpha or a face field leaves it stale. That is what
    // blockamr::la::MFFaceCoeffs handles: it owns the diagonal, refreshes it when
    // the coefficient handles are handed out, and builds a fresh operator per solve.
    // A direct FaceCoeffOp caller (FaceCoeffSolver, solveFaceCoeffs) must
    // reconstruct. bcData_ is referenced either way and always picked up in place.
    std::vector<std::shared_ptr<amrex::MultiFab>> owned_;
    const amrex::MultiFab* ux_ = nullptr;
    const amrex::MultiFab* lx_ = nullptr;
    const amrex::MultiFab* uy_ = nullptr;
    const amrex::MultiFab* ly_ = nullptr;
    const amrex::MultiFab* uz_ = nullptr;
    const amrex::MultiFab* lz_ = nullptr;
    // The diagonal field the stencils read: the caller's field, diagOwned_, or a
    // pinned copy in owned_; never null on an operator built with fields. Under
    // PROTOTYPE (C1) it is alpha itself, the stencils subtracting the face sum.
    const amrex::MultiFab* diag_ = nullptr;
    // The diagonal this operator computed for itself (null-`diag` argument), device
    // path only; on the host path the pinned copy in owned_ is read and this is
    // released.
    std::shared_ptr<amrex::MultiFab> diagOwned_;
    // Inhomogeneous domain-BC data (null = homogeneous, the default); staged to
    // pinned memory in owned_ on the host path like the coefficients. dx_ scales the
    // Neumann datum.
    const amrex::MultiFab* bcData_ = nullptr;
    amrex::Real dx_[3] {};
    std::shared_ptr<amrex::MultiFab> in_;
    std::shared_ptr<amrex::MultiFab> out_;
};

// The fp64 operator every caller means by "FaceCoeffOp", and its fp32 twin; both
// explicitly instantiated in faceCoeffOp.cpp.
using FaceCoeffOp = FaceCoeffOpT<double>;
using FaceCoeffOp32 = FaceCoeffOpT<float>;

} // namespace blockamr::la
