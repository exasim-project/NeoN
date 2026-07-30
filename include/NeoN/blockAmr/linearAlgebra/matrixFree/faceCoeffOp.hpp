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

// The fine-level diagonal diag = alpha - (aE+aW+aN+aS+aT+aB) (negSumDiag) per valid cell, in
// exactly the association order the stencils use, so it is bitwise what they derive inline.
// BC-independent: domain BCs enter through the ghost reflection, i.e. the off-diagonal.
void computeFaceCoeffDiag(
    const NeoN::Executor& exec,
    CellFieldLevel diag,
    const CellFieldLevel& alpha,
    const FaceFieldLevel& upper,
    const FaceFieldLevel& lower
);

// Matrix-free face-coefficient operator on one structured level: the mat-vec is Amul in pull
// form against diag = alpha - (aE+aW+aN+aS+aT+aB); `alpha` is the diagonal SOURCE, and
// upper[d]/lower[d] the HIGH/LOW face couplings -- pass the same fields for both if symmetric.

// PROTOTYPE (C1): the `diag` argument is IGNORED -- both stencils recompute alpha - sum(faces)
// inline, in that same association order (faceCoeffOp.cpp).

// V types the flat Ginkgo vectors only, not the coefficients; V = float is a DEVICE path (the
// host stencil is double) and the constructor rejects it on the host. Measurements:
// report/blockamr-precision-measurements.md
template<class V>
class FaceCoeffOpT : public AmrexLinOpBase<FaceCoeffOpT<V>, V>
{
public:

    explicit FaceCoeffOpT(std::shared_ptr<const gko::Executor> exec);

    // nexec is carried alongside the Ginkgo executor because the launch seam
    // (blockamr::parallelFor) dispatches on the NeoN variant.
    FaceCoeffOpT(
        std::shared_ptr<const gko::Executor> exec,
        const NeoN::Executor& nexec,
        // Allocation layout plus the geometry the stencil's dx and ghost fill come from;
        // only `geom` outlives the constructor, hence geom_ below.
        const MeshLevel& mesh,
        gko::size_type n,
        // const&: a by-value handle would give this constructor write access.
        const CellFieldLevel& alpha,
        const FaceFieldLevel& upper,
        const FaceFieldLevel& lower,
        BcArray bc = {},
        // A bare pointer: its source (SolverConfig::bcData) is a const amrex::MultiFab*.
        const amrex::MultiFab* bcData = nullptr,
        // The caller's stored diagonal (MFFaceCoeffs owns one, so it survives this
        // operator's per-solve rebuild); an empty handle means compute it here, once.
        // Ignored while PROTOTYPE (C1) is live.
        const CellFieldLevel& diag = {}
    );

    // c0 = L(0) into `out`: the constant the INHOMOGENEOUS domain BCs add. Keeps `apply`
    // linear as Ginkgo's Krylov solvers assume -- L(x) = A x + c0, so the solve runs
    // A x = rhs - c0. Requires a bcData operator; throws otherwise.
    void applyBcOffset(const gko::LinOp* zero, gko::LinOp* out) const;

protected:

    // Keeps the base's advanced apply_impl(alpha, b, beta, x) from being hidden.
    using AmrexLinOpBase<FaceCoeffOpT<V>, V>::apply_impl;

    void apply_impl(const gko::LinOp* b, gko::LinOp* x) const override;

private:

    // Shared body of apply_impl and applyBcOffset; `inhom` picks the domain-BC ghost fill.
    void applyWith(const gko::LinOp* b, gko::LinOp* x, bool inhom) const;

    amrex::Geometry geom_;
    // Stencil-launch executor, defaulted for the exec-only constructor (create_default/clear),
    // which builds an operator with no fields to launch over.
    NeoN::Executor nexec_ {NeoN::SerialExecutor {}};
    // Homogeneous domain BCs are applied ON EVERY APPLY by the ghost reflection these drive,
    // hence through the off-diagonal alone (the operator's half: operators/laplacian.hpp).
    BcArray bc_ {};
    bool hasPhysBc_ = false;
    bool onDevice_ = false;
    // Host path: pinned copies, so a caller's in-place write is NOT observed until a new
    // operator is built. Device path: empty, the pointers below reference the caller's fields
    // directly. bcData_ is referenced either way and always picked up in place.
    std::vector<std::shared_ptr<amrex::MultiFab>> owned_;
    const amrex::MultiFab* ux_ = nullptr;
    const amrex::MultiFab* lx_ = nullptr;
    const amrex::MultiFab* uy_ = nullptr;
    const amrex::MultiFab* ly_ = nullptr;
    const amrex::MultiFab* uz_ = nullptr;
    const amrex::MultiFab* lz_ = nullptr;
    // The diagonal field the stencils read; never null on an operator built with fields. Under
    // PROTOTYPE (C1) it is alpha itself, the stencils subtracting the face sum.
    const amrex::MultiFab* diag_ = nullptr;
    // The diagonal this operator would compute for itself (device path); never assigned while
    // PROTOTYPE (C1) is live.
    std::shared_ptr<amrex::MultiFab> diagOwned_;
    // Inhomogeneous domain-BC data (null = homogeneous, the default), staged to pinned memory
    // on the host path like the coefficients. dx_ scales the Neumann datum.
    const amrex::MultiFab* bcData_ = nullptr;
    amrex::Real dx_[3] {};
    std::shared_ptr<amrex::MultiFab> in_;
    std::shared_ptr<amrex::MultiFab> out_;
};

// The fp64 operator every caller means by "FaceCoeffOp", and its fp32 twin.
using FaceCoeffOp = FaceCoeffOpT<double>;
using FaceCoeffOp32 = FaceCoeffOpT<float>;

} // namespace blockamr::la
