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

// Ghost-fill spec for domain side s (0..5) of a valid box: the layer to write, the
// reflection sign (-1 Dirichlet reflect-odd, +1 Neumann reflect-even) and the offset to
// the mirror interior cell. False when periodic or the box does not touch that face.
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

// Fill `mf`'s domain BC ghost layer (1 ghost, comp 0, face layers only): Dirichlet ghost
// = -interior, Neumann ghost = interior. Declaration-only for the nvcc extended-lambda
// trap -- four CUDA TUs in one .so; defined in core/deviceKernels.cpp.
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

// Inhomogeneous twin: ghost = sign*interior + scale*g, g from bcdata's SAME ghost cell
// (MLMG's setLevelBC contract); scale 2 Dirichlet (g = u ON the face), dx Neumann (g =
// du/dn outward; a low side's two flips cancel, so no side sign). For the OUTER residual
// only. Declaration-only, same nvcc trap; defined in core/deviceKernels.cpp.
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

// Host-accessible (pinned) copy: the coefficient fields arrive in the device arena, but
// the face-coeff stencil runs host-side on the ReferenceExecutor.
std::shared_ptr<amrex::MultiFab> pinnedCopy(const amrex::MultiFab& src);

BcArray
parseBc(const std::vector<std::string>& bc, const amrex::Geometry& geom, const std::string& who);

// Validate an inhomogeneous-BC carrier: same layout as `like`, >= 1 ghost layer, and at
// least one non-periodic side to read it on. Refused rather than ignored.
void checkBcData(
    const amrex::MultiFab& bcdata,
    const amrex::MultiFab& like,
    const BcArray& bc,
    const std::string& who
);

// Scatter ONLY the ghost-adjacent shell (outer 1-cell layer of each valid box): that is
// all FillBoundary and the reflect domain-BC fill read, while interior cells are read
// straight from the flat vector. Flat index matches scatter_device (box-by-box, i
// fastest). Instantiated for V = double and float in bc.cpp.
template<class V>
void scatterShellDevice(const NeoN::Executor& exec, const V* vec, amrex::MultiFab& mf);

} // namespace blockamr::la
