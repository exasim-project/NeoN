// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <AMReX_Box.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_MultiFab.H>

#include <array>
#include <cstddef>

// ---------------------------------------------------------------------------
// Homogeneous domain boundary conditions: which ghost layer a boundary side owns,
// what to reflect into it, and the AMReX fill that does it. Header-only and free of
// nanobind and Ginkgo.
//
// Split out of bc.hpp so both sides of the RDC fence can share ONE definition. bc.cpp
// compiles only when Ginkgo does and bc.hpp pulls in nanobind, while the Kokkos
// V-cycle in bench/ has to build the same ghost fill as a device plan -- and wants the
// AMReX fill as the reference to test that plan against. Copying twenty lines of index
// arithmetic into the bench would be the alternative, and the two copies drifting
// apart is exactly the bug this file prevents.
// ---------------------------------------------------------------------------

namespace blockamr::solvers
{

// Domain-boundary condition per side, order (xlo, xhi, ylo, yhi, zlo, zhi):
// 0 = periodic (handled by FillBoundary), 1 = homogeneous Dirichlet (u = 0 on
// the face), 2 = homogeneous Neumann (du/dn = 0 on the face).
using BcArray = std::array<int, 6>;

// Ghost-layer fill spec for domain side s (0..5) of a valid box: the
// one-cell-thick ghost layer to write, the reflection sign (-1 Dirichlet
// reflect-odd, +1 Neumann reflect-even) and the offset from each ghost cell to
// its mirror interior cell. Returns false when the side is periodic or the box
// does not touch that domain face.
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
// Dirichlet -> ghost = -interior (u = 0 at the face, 2nd order at the dx/2
// face distance), Neumann -> ghost = interior (du/dn = 0). Only face ghost
// layers are needed — the 7-point stencil never reads edge/corner ghosts.
// Free function: nvcc forbids an extended __device__ lambda inside a
// protected/private member.
// Templated on the FabArray type: serves the FP64 operator MultiFab and the
// FP32 GMG level fabs; `value_type` sizes the sign/reflection cast.
template<class FA>
void fillDomainBcGhostsDevice(FA& mf, const amrex::Box& domain, const BcArray& bc)
{
    using T = typename FA::value_type;
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
            const T sign = static_cast<T>(f.sign);
            const int di = f.di, dj = f.dj, dk = f.dk;
            amrex::ParallelFor(
                f.gbx,
                [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                { a(i, j, k) = sign * a(i + di, j + dj, k + dk); }
            );
        }
    }
}

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

} // namespace blockamr::solvers
