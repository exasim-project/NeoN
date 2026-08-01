// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <AMReX_GpuQualifiers.H>

// The benchmarked cell kernels, written ONCE and templated on the accessor so every
// launcher runs identical arithmetic. Accessors take GLOBAL (i, j, k); see launch.hpp.

namespace blockamr
{

// y = a*x + y. No stencil, no ghosts: the pure bandwidth floor (2 reads, 1 write).
template<class In, class Out>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
axpyCell(In const& x, Out const& y, int i, int j, int k, double a)
{
    y(i, j, k) = a * x(i, j, k) + y(i, j, k);
}

// Constant-coefficient 7-point Laplacian, 1 ghost -- the stencil shape the V-cycle's
// matrix-free apply has: bandwidth-bound, 8 reads and 1 write per cell before caching.
template<class In, class Out>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
laplacianCell(In const& in, Out const& out, int i, int j, int k, double cx, double cy, double cz)
{
    const double s0 = in(i, j, k);
    out(i, j, k) = cx * (in(i + 1, j, k) + in(i - 1, j, k) - 2.0 * s0)
                 + cy * (in(i, j + 1, k) + in(i, j - 1, k) - 2.0 * s0)
                 + cz * (in(i, j, k + 1) + in(i, j, k - 1) - 2.0 * s0);
}

// Harmonic-mean VanLeer correction, copied from stencilKernels.cpp so both stay comparable.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE double vanleerCorr(double d_up, double d_down)
{
    const double prod = d_up * d_down;
    return (prod > 0.0) ? 2.0 * prod / (d_up + d_down) : 0.0;
}

// VanLeer-limited upwind divergence, mirroring divVanLeerCell in stencilKernels.cpp. Two
// ghost cells and three face fields -- FLOP-heavy and branchy, unlike the two above.
template<class In, class Face, class Out>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void divVanLeerCell(
    In const& phi,
    Face const& fx,
    Face const& fy,
    Face const& fz,
    Out const& out,
    int i,
    int j,
    int k,
    double dx,
    double dy,
    double dz
)
{
    double total = 0.0;
    {
        const double fl = fx(i, j, k), fr = fx(i + 1, j, k);
        const double sm2 = phi(i - 2, j, k), sm1 = phi(i - 1, j, k), s0 = phi(i, j, k),
                     sp1 = phi(i + 1, j, k), sp2 = phi(i + 2, j, k);
        const double dl = s0 - sm1;
        const double pl = (fl >= 0.0) ? sm1 + 0.5 * vanleerCorr(sm1 - sm2, dl)
                                      : s0 - 0.5 * vanleerCorr(sp1 - s0, dl);
        const double dr = sp1 - s0;
        const double pr = (fr >= 0.0) ? s0 + 0.5 * vanleerCorr(s0 - sm1, dr)
                                      : sp1 - 0.5 * vanleerCorr(sp2 - sp1, dr);
        total += (fr * pr - fl * pl) / dx;
    }
    {
        const double fl = fy(i, j, k), fr = fy(i, j + 1, k);
        const double sm2 = phi(i, j - 2, k), sm1 = phi(i, j - 1, k), s0 = phi(i, j, k),
                     sp1 = phi(i, j + 1, k), sp2 = phi(i, j + 2, k);
        const double dl = s0 - sm1;
        const double pl = (fl >= 0.0) ? sm1 + 0.5 * vanleerCorr(sm1 - sm2, dl)
                                      : s0 - 0.5 * vanleerCorr(sp1 - s0, dl);
        const double dr = sp1 - s0;
        const double pr = (fr >= 0.0) ? s0 + 0.5 * vanleerCorr(s0 - sm1, dr)
                                      : sp1 - 0.5 * vanleerCorr(sp2 - sp1, dr);
        total += (fr * pr - fl * pl) / dy;
    }
    {
        const double fl = fz(i, j, k), fr = fz(i, j, k + 1);
        const double sm2 = phi(i, j, k - 2), sm1 = phi(i, j, k - 1), s0 = phi(i, j, k),
                     sp1 = phi(i, j, k + 1), sp2 = phi(i, j, k + 2);
        const double dl = s0 - sm1;
        const double pl = (fl >= 0.0) ? sm1 + 0.5 * vanleerCorr(sm1 - sm2, dl)
                                      : s0 - 0.5 * vanleerCorr(sp1 - s0, dl);
        const double dr = sp1 - s0;
        const double pr = (fr >= 0.0) ? s0 + 0.5 * vanleerCorr(s0 - sm1, dr)
                                      : sp1 - 0.5 * vanleerCorr(sp2 - sp1, dr);
        total += (fr * pr - fl * pl) / dz;
    }
    out(i, j, k) = total;
}

} // namespace blockamr
