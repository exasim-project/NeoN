// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <AMReX_Box.H>
#include <AMReX_Geometry.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_MultiFab.H>

#include <array>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "NeoN/core/executor/executor.hpp"

namespace blockamr::la
{

// Domain-boundary condition per side, order (xlo, xhi, ylo, yhi, zlo, zhi):
// 0 = periodic (FillBoundary's job), 1 = Dirichlet (u = 0 on the face),
// 2 = Neumann (du/dn = 0 on the face).
using BcArray = std::array<int, 6>;

// Ghost-layer fill spec for domain side s (0..5) of a valid box: the layer to
// write, the reflection sign (-1 Dirichlet reflect-odd, +1 Neumann reflect-even)
// and the offset from each ghost cell to its mirror interior cell. False when the
// side is periodic or the box does not touch that domain face.
struct BcGhostFill
{
    amrex::Box gbx;
    double sign;
    int di, dj, dk;
};

inline bool bcGhostFill(
    const amrex::Box& vbx, const amrex::Box& domain, const BcArray& bc, int s, BcGhostFill& f
)
{
    if (bc[static_cast<std::size_t>(s)] == 0)
    {
        return false;
    }
    const int dir = s / 2;
    const bool low = (s % 2) == 0;
    const bool touches =
        low ? vbx.smallEnd(dir) == domain.smallEnd(dir) : vbx.bigEnd(dir) == domain.bigEnd(dir);
    if (!touches)
    {
        return false;
    }
    const int gpos = low ? vbx.smallEnd(dir) - 1 : vbx.bigEnd(dir) + 1;
    f.gbx = vbx;
    f.gbx.setSmall(dir, gpos);
    f.gbx.setBig(dir, gpos);
    f.sign = (bc[static_cast<std::size_t>(s)] == 1) ? -1.0 : 1.0;
    const int shift = low ? 1 : -1;
    f.di = (dir == 0) ? shift : 0;
    f.dj = (dir == 1) ? shift : 0;
    f.dk = (dir == 2) ? shift : 0;
    return true;
}

// Fill the domain-boundary ghost layer of `mf` (1 ghost, component 0) so the
// face-coefficient stencil folds homogeneous BCs with the matrix untouched:
// Dirichlet -> ghost = -interior (2nd order at the dx/2 face distance),
// Neumann -> ghost = interior. Face ghost layers only — the 7-point stencil never
// reads edge/corner ghosts. Free function because nvcc forbids an extended
// __device__ lambda in a non-public member; templated on the FabArray to serve both
// the FP64 operator MultiFab and the FP32 GMG level fabs.
//
// Declaration-only: it is reached from four CUDA TUs that all land in the one
// _blockamr.so, so an inline AMREX_GPU_DEVICE lambda would be an extended lambda
// instantiated four times over — the nvcc trap. Definition and explicit
// instantiation live in core/deviceKernels.cpp.
template<class FA>
void fillDomainBcGhostsDevice(FA& mf, const amrex::Box& domain, const BcArray& bc);

// Host-loop twin of fillDomainBcGhostsDevice for the ReferenceExecutor path.
template<class FA>
void fillDomainBcGhostsHost(FA& mf, const amrex::Box& domain, const BcArray& bc)
{
    for (amrex::MFIter mfi(mf); mfi.isValid(); ++mfi)
    {
        const amrex::Box& vbx = mfi.validbox();
        const auto a = mf.array(mfi);
        for (int s = 0; s < 6; ++s)
        {
            BcGhostFill f;
            if (!bcGhostFill(vbx, domain, bc, s, f))
            {
                continue;
            }
            const auto lo = amrex::lbound(f.gbx);
            const auto hi = amrex::ubound(f.gbx);
            for (int k = lo.z; k <= hi.z; ++k)
            {
                for (int j = lo.y; j <= hi.y; ++j)
                {
                    for (int i = lo.x; i <= hi.x; ++i)
                    {
                        a(i, j, k) = f.sign * a(i + f.di, j + f.dj, k + f.dk);
                    }
                }
            }
        }
    }
}

// Inhomogeneous twin of the two fills above: ghost = sign*interior + scale*g, g
// read from the SAME ghost cell of `bcdata`, with
//   Dirichlet (sign -1): scale = 2        , g = u on the boundary FACE
//   Neumann   (sign +1): scale = dx[dir]  , g = du/dn, the OUTWARD normal derivative
// i.e. the homogeneous fill with the datum moved off zero, on the same dx/2 face
// placement — so the discretisation ORDER is unchanged, only the constant. The
// Neumann scale carries no side sign: on a low side the ghost offset and the
// outward normal both flip, and the two cancel.
//
// `bcdata` is cell-centred on `mf`'s BoxArray/DistributionMapping with >= 1 ghost
// and the datum living in the ghost layer — MLMG's setLevelBC contract, so one
// MultiFab drives both solvers (pinned by test_inhomogeneous_dirichlet_matches_mlmg).
// Only the domain-boundary ghost layer is read.
//
// For the OUTER residual only: the V-cycle and the Ginkgo operator keep the
// homogeneous fill, because both solve for a CORRECTION, whose boundary condition
// is homogeneous whatever the solution's is.
//
// Declaration-only for the same nvcc multi-TU reason as its twin above; defined in
// core/deviceKernels.cpp. It can take `exec` because both its callers hold one,
// unlike the twin, which is also reached from GMG headers that carry none.
void fillDomainBcGhostsInhomDevice(
    const NeoN::Executor& exec,
    amrex::MultiFab& mf,
    const amrex::MultiFab& bcdata,
    const amrex::Box& domain,
    const BcArray& bc,
    const amrex::Real* dx
);

// Host-loop twin of fillDomainBcGhostsInhomDevice for the ReferenceExecutor path.
inline void fillDomainBcGhostsInhomHost(
    amrex::MultiFab& mf,
    const amrex::MultiFab& bcdata,
    const amrex::Box& domain,
    const BcArray& bc,
    const amrex::Real* dx
)
{
    for (amrex::MFIter mfi(mf); mfi.isValid(); ++mfi)
    {
        const amrex::Box& vbx = mfi.validbox();
        const auto a = mf.array(mfi);
        const auto g = bcdata.const_array(mfi);
        for (int s = 0; s < 6; ++s)
        {
            BcGhostFill f;
            if (!bcGhostFill(vbx, domain, bc, s, f))
            {
                continue;
            }
            const double scale = (bc[static_cast<std::size_t>(s)] == 1) ? 2.0 : dx[s / 2];
            const auto lo = amrex::lbound(f.gbx);
            const auto hi = amrex::ubound(f.gbx);
            for (int k = lo.z; k <= hi.z; ++k)
            {
                for (int j = lo.y; j <= hi.y; ++j)
                {
                    for (int i = lo.x; i <= hi.x; ++i)
                    {
                        a(i, j, k) = f.sign * a(i + f.di, j + f.dj, k + f.dk) + scale * g(i, j, k);
                    }
                }
            }
        }
    }
}

// Host-accessible (pinned) copy of a MultiFab: the coefficient fields arrive in
// the default (device) arena, but the face-coeff stencil runs host-side on the
// ReferenceExecutor, so the solve-constant coefficients are staged once.
std::shared_ptr<amrex::MultiFab> pinnedCopy(const amrex::MultiFab& src);

BcArray
parseBc(const std::vector<std::string>& bc, const amrex::Geometry& geom, const std::string& who);

// Validate an inhomogeneous-BC carrier against the operator it is read alongside:
// same BoxArray and DistributionMapping as `like`, >= 1 ghost layer to hold the
// data, and at least one non-periodic side to read it on. Refused rather than
// ignored — a carrier nothing consults reads as a solver bug, not a configuration
// one.
void checkBcData(
    const amrex::MultiFab& bcdata,
    const amrex::MultiFab& like,
    const BcArray& bc,
    const std::string& who
);

// Scatter ONLY the ghost-adjacent shell (outer 1-cell layer of each valid box)
// from the flat Ginkgo vector into the MultiFab: that shell is all FillBoundary and
// the reflect domain-BC fill read to populate the face ghosts, while the interior
// valid cells are read straight from the flat vector by
// faceCoeffStencilFusedDevice. Flat index matches scatter_device (box-by-box, i
// fastest). Instantiated for V = double and float (bc.cpp); the MultiFab is always
// amrex::Real, only the flat Krylov vector changes width. `exec` is available here
// because this kernel's one caller, FaceCoeffOpT::applyWith, holds one.
template<class V>
void scatterShellDevice(const NeoN::Executor& exec, const V* vec, amrex::MultiFab& mf);

} // namespace blockamr::la
