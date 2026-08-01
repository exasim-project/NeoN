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

#include "NeoN/blockAmr/core/bc.hpp"
#include "NeoN/blockAmr/core/fieldLevel.hpp"
#include "NeoN/blockAmr/core/meshLevel.hpp"
#include "NeoN/blockAmr/linearAlgebra/matrixFree/linOpBase.hpp"
#include "NeoN/blockAmr/linearAlgebra/faceCoeffLevel.hpp"
#include "NeoN/blockAmr/core/types.hpp"
#include "NeoN/core/executor/executor.hpp"

namespace blockamr::la
{

/* @brief The domain boundary condition ONE operator applies: the per-side spec, whose
 *        homogeneous ghost reflection runs on every apply, and the optional INHOMOGENEOUS
 *        datum, which only applyBcOffset reads. One argument because the datum is
 *        meaningless without the sides that say where it is read -- and because a null
 *        `data` next to a `sides` is exactly the pair every caller passes.
 */
struct DomainBc
{
    BcArray sides {};
    // A bare pointer: its source (SolverConfig::bcData) is a const amrex::MultiFab*.
    const amrex::MultiFab* data = nullptr;
};

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
        // The coefficients (`alpha` the diagonal SOURCE, upper/lower the HIGH/LOW face
        // couplings) together with the allocation layout and the geometry the stencil's dx and
        // ghost fill come from. const&: a by-value bundle would give this write access.
        // level.mesh.ba also FIXES the row count: numPts(), the global one every rank agrees
        // on (la::globalRows), which is what every call site passed by hand.
        const FaceCoeffLevel& level,
        DomainBc bc = {},
        // The caller's stored diagonal; an empty handle means compute it here, once.
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

    // Fused path: the stencil reads the interior from the flat Ginkgo vectors.
    void applyFused(const gko::LinOp* b, gko::LinOp* x, bool inhom) const;

    // Everything through the ghosted scratch MultiFab, computing in double.
    void applyStaged(const gko::LinOp* b, gko::LinOp* x, bool inhom) const;

    // Periodic/internal ghosts plus the domain-BC reflection on the device scratch.
    void fillGhostsDevice(bool inhom) const;

    // Copy the coefficients (and any bc data) into pinned memory and hold those instead.
    void stagePinned(const FaceCoeffLevel& level, const amrex::MultiFab* bcData);

    // Stencil-launch executor, defaulted for the exec-only constructor (create_default/clear),
    // which builds an operator with no fields to launch over.
    NeoN::Executor nexec_ {NeoN::SerialExecutor {}};
    // Homogeneous domain BCs are applied ON EVERY APPLY by the ghost reflection these drive,
    // hence through the off-diagonal alone (the operator's half: operators/laplacian.hpp).
    BcArray bc_ {};
    bool hasPhysBc_ = false;
    bool onDevice_ = false;
    // What the stencils read, plus the mesh they and the ghost fill run over; empty handles on
    // an operator built without fields. Host path: the coefficients are pinned copies, so a
    // caller's in-place write is NOT observed until a new operator is built. Device path: the
    // caller's own handles. Under PROTOTYPE (C1) level_.alpha is the field the centre term
    // comes from, the stencils subtracting the face sum themselves.
    FaceCoeffLevel level_;
    // The diagonal this operator would compute for itself (device path); never assigned while
    // PROTOTYPE (C1) is live.
    std::shared_ptr<amrex::MultiFab> diagOwned_;
    // Inhomogeneous domain-BC data (null = homogeneous, the default), staged to pinned memory
    // on the host path like the coefficients. dx_ scales the Neumann datum.
    const amrex::MultiFab* bcData_ = nullptr;
    std::shared_ptr<amrex::MultiFab> bcDataOwned_;
    amrex::Real dx_[3] {};
    std::shared_ptr<amrex::MultiFab> in_;
    std::shared_ptr<amrex::MultiFab> out_;
};

// The fp64 operator every caller means by "FaceCoeffOp", and its fp32 twin.
using FaceCoeffOp = FaceCoeffOpT<double>;
using FaceCoeffOp32 = FaceCoeffOpT<float>;

} // namespace blockamr::la
