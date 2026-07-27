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

#include "../common/bc.hpp"
#include "../common/linop_base.hpp"
#include "../common/types.hpp"

namespace blockamr::solvers
{

// General matrix-free face-coefficient operator on a structured single-level grid. The
// matrix is carried as OpenFOAM-style pieces given as AMReX fields:
//   alpha  : cell-centred diagonal SOURCE (ddt/Sp/reaction), NOT the full
//            diagonal — the face part is derived below (negSumDiag).
//   u{x,y,z}, l{x,y,z} : face-centred upper/lower off-diagonal coefficients.
//             u* is the owner-row->neighbour coupling on the cell's HIGH face,
//            l* the neighbour-row->owner coupling on the cell's LOW face. For a
//            symmetric matrix pass the same MultiFab for u* and l*.
// The mat-vec is the OpenFOAM Amul in pull form (each cell reads its 6
// neighbours), with the diagonal assembled on the fly as
//   diag = alpha - (aE+aW+aN+aS+aT+aB)               (negSumDiag)
// so no cell-diagonal array is stored — the face coeffs feed both the
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

    FaceCoeffOpT(
        std::shared_ptr<const gko::Executor> exec,
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
        const amrex::MultiFab* bcData = nullptr
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
    BcArray bc_ {};
    bool hasPhysBc_ = false;
    bool onDevice_ = false;
    // Host path: owns pinned copies of the coefficient fields. Device path:
    // empty, and the pointers below reference the caller's device-resident
    // fields directly, so an external in-place update to the coefficients is
    // picked up by the next apply with no reassembly.
    std::vector<std::shared_ptr<amrex::MultiFab>> owned_;
    const amrex::MultiFab* alpha_ = nullptr;
    const amrex::MultiFab* ux_ = nullptr;
    const amrex::MultiFab* lx_ = nullptr;
    const amrex::MultiFab* uy_ = nullptr;
    const amrex::MultiFab* ly_ = nullptr;
    const amrex::MultiFab* uz_ = nullptr;
    const amrex::MultiFab* lz_ = nullptr;
    // Inhomogeneous domain-BC data (null = homogeneous, the default and the
    // historical behaviour); staged to pinned memory in owned_ on the host path
    // exactly like the coefficients. dx_ scales the Neumann datum.
    const amrex::MultiFab* bcData_ = nullptr;
    amrex::Real dx_[3] {};
    std::shared_ptr<amrex::MultiFab> in_;
    std::shared_ptr<amrex::MultiFab> out_;
};

// The fp64 operator every existing caller means by "FaceCoeffOp", and its fp32
// twin. Both are explicitly instantiated in face_coeff_op.cpp.
using FaceCoeffOp = FaceCoeffOpT<double>;
using FaceCoeffOp32 = FaceCoeffOpT<float>;

} // namespace blockamr::solvers
