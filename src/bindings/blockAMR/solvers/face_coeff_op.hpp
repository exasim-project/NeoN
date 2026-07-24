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

#include "bc.hpp"
#include "linop_base.hpp"
#include "types.hpp"

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
class FaceCoeffOp : public AmrexLinOpBase<FaceCoeffOp>
{
public:

    explicit FaceCoeffOp(std::shared_ptr<const gko::Executor> exec);

    FaceCoeffOp(
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
        BcArray bc = {}
    );

protected:

    // Keeps the base's advanced apply_impl(alpha, b, beta, x) visible in this
    // scope (the declaration below would otherwise hide it).
    using AmrexLinOpBase<FaceCoeffOp>::apply_impl;

    void apply_impl(const gko::LinOp* b, gko::LinOp* x) const override;

private:

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
    std::shared_ptr<amrex::MultiFab> in_;
    std::shared_ptr<amrex::MultiFab> out_;
};

} // namespace blockamr::solvers
