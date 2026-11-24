// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

// C++ stencil kernels of the explicit backend, run directly on AMReX MultiFabs via ParallelFor.
// Throughout this file fx/fy/fz are face FLUXES of an explicit scheme -- NOT the matrix face
// coefficients of the implicit/la path.

#include <nanobind/nanobind.h>

#include <AMReX.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Geometry.H>

#include "bindings.hpp"

namespace nb = nanobind;

namespace
{

// Harmonic-mean VanLeer correction: ψ(r)*Δ with one division, no abs.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
vanleerCorr(amrex::Real d_up, amrex::Real d_down)
{
    amrex::Real prod = d_up * d_down;
    return (prod > 0.0) ? 2.0 * prod / (d_up + d_down) : 0.0;
}

// Comp-indexed cell-level scheme helpers (per-axis division, x->y->z accumulation order).

using Arr4c = amrex::Array4<const amrex::Real>;

AMREX_GPU_DEVICE AMREX_FORCE_INLINE amrex::Real divUpwindCell(
    Arr4c const& phi,
    Arr4c const& fx,
    Arr4c const& fy,
    Arr4c const& fz,
    int i,
    int j,
    int k,
    int n,
    amrex::Real dx,
    amrex::Real dy,
    amrex::Real dz
)
{
    amrex::Real s0 = phi(i, j, k, n);
    amrex::Real total = 0.0;
    {
        amrex::Real fl = fx(i, j, k), fr = fx(i + 1, j, k);
        amrex::Real Fl = fl * ((fl >= 0.0) ? phi(i - 1, j, k, n) : s0);
        amrex::Real Fr = fr * ((fr >= 0.0) ? s0 : phi(i + 1, j, k, n));
        total += (Fr - Fl) / dx;
    }
    {
        amrex::Real fl = fy(i, j, k), fr = fy(i, j + 1, k);
        amrex::Real Fl = fl * ((fl >= 0.0) ? phi(i, j - 1, k, n) : s0);
        amrex::Real Fr = fr * ((fr >= 0.0) ? s0 : phi(i, j + 1, k, n));
        total += (Fr - Fl) / dy;
    }
    {
        amrex::Real fl = fz(i, j, k), fr = fz(i, j, k + 1);
        amrex::Real Fl = fl * ((fl >= 0.0) ? phi(i, j, k - 1, n) : s0);
        amrex::Real Fr = fr * ((fr >= 0.0) ? s0 : phi(i, j, k + 1, n));
        total += (Fr - Fl) / dz;
    }
    return total;
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE amrex::Real divLinearCell(
    Arr4c const& phi,
    Arr4c const& fx,
    Arr4c const& fy,
    Arr4c const& fz,
    int i,
    int j,
    int k,
    int n,
    amrex::Real dx,
    amrex::Real dy,
    amrex::Real dz
)
{
    amrex::Real s0 = phi(i, j, k, n);
    amrex::Real total = 0.0;
    {
        amrex::Real fl = fx(i, j, k), fr = fx(i + 1, j, k);
        amrex::Real Fl = fl * 0.5 * (phi(i - 1, j, k, n) + s0);
        amrex::Real Fr = fr * 0.5 * (s0 + phi(i + 1, j, k, n));
        total += (Fr - Fl) / dx;
    }
    {
        amrex::Real fl = fy(i, j, k), fr = fy(i, j + 1, k);
        amrex::Real Fl = fl * 0.5 * (phi(i, j - 1, k, n) + s0);
        amrex::Real Fr = fr * 0.5 * (s0 + phi(i, j + 1, k, n));
        total += (Fr - Fl) / dy;
    }
    {
        amrex::Real fl = fz(i, j, k), fr = fz(i, j, k + 1);
        amrex::Real Fl = fl * 0.5 * (phi(i, j, k - 1, n) + s0);
        amrex::Real Fr = fr * 0.5 * (s0 + phi(i, j, k + 1, n));
        total += (Fr - Fl) / dz;
    }
    return total;
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE amrex::Real divVanLeerCell(
    Arr4c const& phi,
    Arr4c const& fx,
    Arr4c const& fy,
    Arr4c const& fz,
    int i,
    int j,
    int k,
    int n,
    amrex::Real dx,
    amrex::Real dy,
    amrex::Real dz
)
{
    amrex::Real total = 0.0;
    {
        amrex::Real fl = fx(i, j, k), fr = fx(i + 1, j, k);
        amrex::Real sm2 = phi(i - 2, j, k, n), sm1 = phi(i - 1, j, k, n), s0 = phi(i, j, k, n),
                    sp1 = phi(i + 1, j, k, n), sp2 = phi(i + 2, j, k, n);
        amrex::Real dl = s0 - sm1;
        amrex::Real pl = (fl >= 0.0) ? sm1 + 0.5 * vanleerCorr(sm1 - sm2, dl)
                                     : s0 - 0.5 * vanleerCorr(sp1 - s0, dl);
        amrex::Real dr = sp1 - s0;
        amrex::Real pr = (fr >= 0.0) ? s0 + 0.5 * vanleerCorr(s0 - sm1, dr)
                                     : sp1 - 0.5 * vanleerCorr(sp2 - sp1, dr);
        total += (fr * pr - fl * pl) / dx;
    }
    {
        amrex::Real fl = fy(i, j, k), fr = fy(i, j + 1, k);
        amrex::Real sm2 = phi(i, j - 2, k, n), sm1 = phi(i, j - 1, k, n), s0 = phi(i, j, k, n),
                    sp1 = phi(i, j + 1, k, n), sp2 = phi(i, j + 2, k, n);
        amrex::Real dl = s0 - sm1;
        amrex::Real pl = (fl >= 0.0) ? sm1 + 0.5 * vanleerCorr(sm1 - sm2, dl)
                                     : s0 - 0.5 * vanleerCorr(sp1 - s0, dl);
        amrex::Real dr = sp1 - s0;
        amrex::Real pr = (fr >= 0.0) ? s0 + 0.5 * vanleerCorr(s0 - sm1, dr)
                                     : sp1 - 0.5 * vanleerCorr(sp2 - sp1, dr);
        total += (fr * pr - fl * pl) / dy;
    }
    {
        amrex::Real fl = fz(i, j, k), fr = fz(i, j, k + 1);
        amrex::Real sm2 = phi(i, j, k - 2, n), sm1 = phi(i, j, k - 1, n), s0 = phi(i, j, k, n),
                    sp1 = phi(i, j, k + 1, n), sp2 = phi(i, j, k + 2, n);
        amrex::Real dl = s0 - sm1;
        amrex::Real pl = (fl >= 0.0) ? sm1 + 0.5 * vanleerCorr(sm1 - sm2, dl)
                                     : s0 - 0.5 * vanleerCorr(sp1 - s0, dl);
        amrex::Real dr = sp1 - s0;
        amrex::Real pr = (fr >= 0.0) ? s0 + 0.5 * vanleerCorr(s0 - sm1, dr)
                                     : sp1 - 0.5 * vanleerCorr(sp2 - sp1, dr);
        total += (fr * pr - fl * pl) / dz;
    }
    return total;
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE amrex::Real divQuickCell(
    Arr4c const& phi,
    Arr4c const& fx,
    Arr4c const& fy,
    Arr4c const& fz,
    int i,
    int j,
    int k,
    int n,
    amrex::Real dx,
    amrex::Real dy,
    amrex::Real dz
)
{
    amrex::Real total = 0.0;
    {
        amrex::Real fl = fx(i, j, k), fr = fx(i + 1, j, k);
        amrex::Real sm2 = phi(i - 2, j, k, n), sm1 = phi(i - 1, j, k, n), s0 = phi(i, j, k, n),
                    sp1 = phi(i + 1, j, k, n), sp2 = phi(i + 2, j, k, n);
        amrex::Real pl = (fl >= 0.0) ? 0.375 * s0 + 0.75 * sm1 - 0.125 * sm2
                                     : 0.375 * sm1 + 0.75 * s0 - 0.125 * sp1;
        amrex::Real pr = (fr >= 0.0) ? 0.375 * sp1 + 0.75 * s0 - 0.125 * sm1
                                     : 0.375 * s0 + 0.75 * sp1 - 0.125 * sp2;
        total += (fr * pr - fl * pl) / dx;
    }
    {
        amrex::Real fl = fy(i, j, k), fr = fy(i, j + 1, k);
        amrex::Real sm2 = phi(i, j - 2, k, n), sm1 = phi(i, j - 1, k, n), s0 = phi(i, j, k, n),
                    sp1 = phi(i, j + 1, k, n), sp2 = phi(i, j + 2, k, n);
        amrex::Real pl = (fl >= 0.0) ? 0.375 * s0 + 0.75 * sm1 - 0.125 * sm2
                                     : 0.375 * sm1 + 0.75 * s0 - 0.125 * sp1;
        amrex::Real pr = (fr >= 0.0) ? 0.375 * sp1 + 0.75 * s0 - 0.125 * sm1
                                     : 0.375 * s0 + 0.75 * sp1 - 0.125 * sp2;
        total += (fr * pr - fl * pl) / dy;
    }
    {
        amrex::Real fl = fz(i, j, k), fr = fz(i, j, k + 1);
        amrex::Real sm2 = phi(i, j, k - 2, n), sm1 = phi(i, j, k - 1, n), s0 = phi(i, j, k, n),
                    sp1 = phi(i, j, k + 1, n), sp2 = phi(i, j, k + 2, n);
        amrex::Real pl = (fl >= 0.0) ? 0.375 * s0 + 0.75 * sm1 - 0.125 * sm2
                                     : 0.375 * sm1 + 0.75 * s0 - 0.125 * sp1;
        amrex::Real pr = (fr >= 0.0) ? 0.375 * sp1 + 0.75 * s0 - 0.125 * sm1
                                     : 0.375 * s0 + 0.75 * sp1 - 0.125 * sp2;
        total += (fr * pr - fl * pl) / dz;
    }
    return total;
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE amrex::Real laplacianCell(
    Arr4c const& phi, int i, int j, int k, int n, amrex::Real dx, amrex::Real dy, amrex::Real dz
)
{
    amrex::Real c = phi(i, j, k, n);
    return (phi(i + 1, j, k, n) - 2.0 * c + phi(i - 1, j, k, n)) / (dx * dx)
         + (phi(i, j + 1, k, n) - 2.0 * c + phi(i, j - 1, k, n)) / (dy * dy)
         + (phi(i, j, k + 1, n) - 2.0 * c + phi(i, j, k - 1, n)) / (dz * dz);
}

} // namespace


// Composable accumulate kernels (out +=): the cpp backend composes these into a scratch source.

// Accumulate coeff * div_scheme(phi) into out, one term per launch, ncomp-general.
static void divUpwindAcc(
    amrex::MultiFab& out_mf,
    const amrex::MultiFab& phi_mf,
    const amrex::MultiFab& fx_mf,
    const amrex::MultiFab& fy_mf,
    const amrex::MultiFab& fz_mf,
    const amrex::Geometry& geom,
    amrex::Real coeff,
    int ncomp
)
{
    const auto dx = geom.CellSizeArray();
    const amrex::Real dhx = dx[0], dhy = dx[1], dhz = dx[2];
    for (amrex::MFIter mfi(phi_mf); mfi.isValid(); ++mfi)
    {
        const amrex::Box& bx = mfi.validbox();
        auto const& phi = phi_mf.const_array(mfi);
        auto const& out = out_mf.array(mfi);
        auto const& fx = fx_mf.const_array(mfi);
        auto const& fy = fy_mf.const_array(mfi);
        auto const& fz = fz_mf.const_array(mfi);
        amrex::ParallelFor(
            bx,
            ncomp,
            [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) {
                out(i, j, k, n) +=
                    coeff * divUpwindCell(phi, fx, fy, fz, i, j, k, n, dhx, dhy, dhz);
            }
        );
    }
}

static void divLinearAcc(
    amrex::MultiFab& out_mf,
    const amrex::MultiFab& phi_mf,
    const amrex::MultiFab& fx_mf,
    const amrex::MultiFab& fy_mf,
    const amrex::MultiFab& fz_mf,
    const amrex::Geometry& geom,
    amrex::Real coeff,
    int ncomp
)
{
    const auto dx = geom.CellSizeArray();
    const amrex::Real dhx = dx[0], dhy = dx[1], dhz = dx[2];
    for (amrex::MFIter mfi(phi_mf); mfi.isValid(); ++mfi)
    {
        const amrex::Box& bx = mfi.validbox();
        auto const& phi = phi_mf.const_array(mfi);
        auto const& out = out_mf.array(mfi);
        auto const& fx = fx_mf.const_array(mfi);
        auto const& fy = fy_mf.const_array(mfi);
        auto const& fz = fz_mf.const_array(mfi);
        amrex::ParallelFor(
            bx,
            ncomp,
            [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) {
                out(i, j, k, n) +=
                    coeff * divLinearCell(phi, fx, fy, fz, i, j, k, n, dhx, dhy, dhz);
            }
        );
    }
}

static void divVanLeerAcc(
    amrex::MultiFab& out_mf,
    const amrex::MultiFab& phi_mf,
    const amrex::MultiFab& fx_mf,
    const amrex::MultiFab& fy_mf,
    const amrex::MultiFab& fz_mf,
    const amrex::Geometry& geom,
    amrex::Real coeff,
    int ncomp
)
{
    const auto dx = geom.CellSizeArray();
    const amrex::Real dhx = dx[0], dhy = dx[1], dhz = dx[2];
    for (amrex::MFIter mfi(phi_mf); mfi.isValid(); ++mfi)
    {
        const amrex::Box& bx = mfi.validbox();
        auto const& phi = phi_mf.const_array(mfi);
        auto const& out = out_mf.array(mfi);
        auto const& fx = fx_mf.const_array(mfi);
        auto const& fy = fy_mf.const_array(mfi);
        auto const& fz = fz_mf.const_array(mfi);
        amrex::ParallelFor(
            bx,
            ncomp,
            [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) {
                out(i, j, k, n) +=
                    coeff * divVanLeerCell(phi, fx, fy, fz, i, j, k, n, dhx, dhy, dhz);
            }
        );
    }
}

static void divQuickAcc(
    amrex::MultiFab& out_mf,
    const amrex::MultiFab& phi_mf,
    const amrex::MultiFab& fx_mf,
    const amrex::MultiFab& fy_mf,
    const amrex::MultiFab& fz_mf,
    const amrex::Geometry& geom,
    amrex::Real coeff,
    int ncomp
)
{
    const auto dx = geom.CellSizeArray();
    const amrex::Real dhx = dx[0], dhy = dx[1], dhz = dx[2];
    for (amrex::MFIter mfi(phi_mf); mfi.isValid(); ++mfi)
    {
        const amrex::Box& bx = mfi.validbox();
        auto const& phi = phi_mf.const_array(mfi);
        auto const& out = out_mf.array(mfi);
        auto const& fx = fx_mf.const_array(mfi);
        auto const& fy = fy_mf.const_array(mfi);
        auto const& fz = fz_mf.const_array(mfi);
        amrex::ParallelFor(
            bx,
            ncomp,
            [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
            { out(i, j, k, n) += coeff * divQuickCell(phi, fx, fy, fz, i, j, k, n, dhx, dhy, dhz); }
        );
    }
}

// Accumulate coeff * laplacian(phi) into out (constant gamma folded into coeff).
static void laplacianAcc(
    amrex::MultiFab& out_mf,
    const amrex::MultiFab& phi_mf,
    const amrex::Geometry& geom,
    amrex::Real coeff,
    int ncomp
)
{
    const auto dx = geom.CellSizeArray();
    const amrex::Real dhx = dx[0], dhy = dx[1], dhz = dx[2];
    for (amrex::MFIter mfi(phi_mf); mfi.isValid(); ++mfi)
    {
        const amrex::Box& bx = mfi.validbox();
        auto const& phi = phi_mf.const_array(mfi);
        auto const& out = out_mf.array(mfi);
        amrex::ParallelFor(
            bx,
            ncomp,
            [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
            { out(i, j, k, n) += coeff * laplacianCell(phi, i, j, k, n, dhx, dhy, dhz); }
        );
    }
}

// Accumulate coeff * grad(phi) into out: scalar phi (comp 0) → 3-component vector,
// central difference (phi_r - phi_l)/(2 dh) per spatial direction (cf. grad.py).
static void gradAcc(
    amrex::MultiFab& out_mf,
    const amrex::MultiFab& phi_mf,
    const amrex::Geometry& geom,
    amrex::Real coeff
)
{
    const auto dx = geom.CellSizeArray();
    const amrex::Real dhx = dx[0], dhy = dx[1], dhz = dx[2];
    for (amrex::MFIter mfi(phi_mf); mfi.isValid(); ++mfi)
    {
        const amrex::Box& bx = mfi.validbox();
        auto const& phi = phi_mf.const_array(mfi);
        auto const& out = out_mf.array(mfi);
        amrex::ParallelFor(
            bx,
            [=] AMREX_GPU_DEVICE(int i, int j, int k)
            {
                out(i, j, k, 0) +=
                    coeff * (phi(i + 1, j, k, 0) - phi(i - 1, j, k, 0)) / (2.0 * dhx);
                out(i, j, k, 1) +=
                    coeff * (phi(i, j + 1, k, 0) - phi(i, j - 1, k, 0)) / (2.0 * dhy);
                out(i, j, k, 2) +=
                    coeff * (phi(i, j, k + 1, 0) - phi(i, j, k - 1, 0)) / (2.0 * dhz);
            }
        );
    }
}

// Accumulate coeff * phi into out (pointwise source term).
static void
sourceAcc(amrex::MultiFab& out_mf, const amrex::MultiFab& phi_mf, amrex::Real coeff, int ncomp)
{
    for (amrex::MFIter mfi(phi_mf); mfi.isValid(); ++mfi)
    {
        const amrex::Box& bx = mfi.validbox();
        auto const& phi = phi_mf.const_array(mfi);
        auto const& out = out_mf.array(mfi);
        amrex::ParallelFor(
            bx,
            ncomp,
            [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
            { out(i, j, k, n) += coeff * phi(i, j, k, n); }
        );
    }
}

// Generic forward-Euler axpy: phi -= dt_over_coeff * src.
static void eulerUpdate(
    amrex::MultiFab& phi_mf, const amrex::MultiFab& src_mf, amrex::Real dt_over_coeff, int ncomp
)
{
    for (amrex::MFIter mfi(phi_mf); mfi.isValid(); ++mfi)
    {
        const amrex::Box& bx = mfi.validbox();
        auto const& phi = phi_mf.array(mfi);
        auto const& src = src_mf.const_array(mfi);
        amrex::ParallelFor(
            bx,
            ncomp,
            [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
            { phi(i, j, k, n) -= dt_over_coeff * src(i, j, k, n); }
        );
    }
}


// Divergence-only source kernels: no time step, written to a separate output.

// Upwind divergence source term.
static void divUpwind(
    amrex::MultiFab& out_mf,
    const amrex::MultiFab& phi_mf,
    const amrex::MultiFab& fx_mf,
    const amrex::MultiFab& fy_mf,
    const amrex::MultiFab& fz_mf,
    const amrex::Geometry& geom
)
{
    const auto dx = geom.CellSizeArray();
    const amrex::Real idx = 1.0 / dx[0];
    const amrex::Real idy = 1.0 / dx[1];
    const amrex::Real idz = 1.0 / dx[2];

    for (amrex::MFIter mfi(phi_mf); mfi.isValid(); ++mfi)
    {
        const amrex::Box& bx = mfi.validbox();
        auto const& phi = phi_mf.const_array(mfi);
        auto const& out = out_mf.array(mfi);
        auto const& fx = fx_mf.const_array(mfi);
        auto const& fy = fy_mf.const_array(mfi);
        auto const& fz = fz_mf.const_array(mfi);

        amrex::ParallelFor(
            bx,
            [=] AMREX_GPU_DEVICE(int i, int j, int k)
            {
                amrex::Real s0 = phi(i, j, k);
                amrex::Real fl_x = fx(i, j, k), fr_x = fx(i + 1, j, k);
                amrex::Real fl_y = fy(i, j, k), fr_y = fy(i, j + 1, k);
                amrex::Real fl_z = fz(i, j, k), fr_z = fz(i, j, k + 1);
                out(i, j, k) = (fr_x * ((fr_x >= 0) ? s0 : phi(i + 1, j, k))
                                - fl_x * ((fl_x >= 0) ? phi(i - 1, j, k) : s0))
                                 * idx
                             + (fr_y * ((fr_y >= 0) ? s0 : phi(i, j + 1, k))
                                - fl_y * ((fl_y >= 0) ? phi(i, j - 1, k) : s0))
                                   * idy
                             + (fr_z * ((fr_z >= 0) ? s0 : phi(i, j, k + 1))
                                - fl_z * ((fl_z >= 0) ? phi(i, j, k - 1) : s0))
                                   * idz;
            }
        );
    }
}

// Linear divergence source term.
static void divLinear(
    amrex::MultiFab& out_mf,
    const amrex::MultiFab& phi_mf,
    const amrex::MultiFab& fx_mf,
    const amrex::MultiFab& fy_mf,
    const amrex::MultiFab& fz_mf,
    const amrex::Geometry& geom
)
{
    const auto dx = geom.CellSizeArray();
    const amrex::Real idx = 1.0 / dx[0];
    const amrex::Real idy = 1.0 / dx[1];
    const amrex::Real idz = 1.0 / dx[2];

    for (amrex::MFIter mfi(phi_mf); mfi.isValid(); ++mfi)
    {
        const amrex::Box& bx = mfi.validbox();
        auto const& phi = phi_mf.const_array(mfi);
        auto const& out = out_mf.array(mfi);
        auto const& fx = fx_mf.const_array(mfi);
        auto const& fy = fy_mf.const_array(mfi);
        auto const& fz = fz_mf.const_array(mfi);

        amrex::ParallelFor(
            bx,
            [=] AMREX_GPU_DEVICE(int i, int j, int k)
            {
                amrex::Real s0 = phi(i, j, k);
                out(i, j, k) = (fx(i + 1, j, k) * 0.5 * (s0 + phi(i + 1, j, k))
                                - fx(i, j, k) * 0.5 * (phi(i - 1, j, k) + s0))
                                 * idx
                             + (fy(i, j + 1, k) * 0.5 * (s0 + phi(i, j + 1, k))
                                - fy(i, j, k) * 0.5 * (phi(i, j - 1, k) + s0))
                                   * idy
                             + (fz(i, j, k + 1) * 0.5 * (s0 + phi(i, j, k + 1))
                                - fz(i, j, k) * 0.5 * (phi(i, j, k - 1) + s0))
                                   * idz;
            }
        );
    }
}

// VanLeer divergence source term.
static void divVanLeer(
    amrex::MultiFab& out_mf,
    const amrex::MultiFab& phi_mf,
    const amrex::MultiFab& fx_mf,
    const amrex::MultiFab& fy_mf,
    const amrex::MultiFab& fz_mf,
    const amrex::Geometry& geom
)
{
    const auto dx = geom.CellSizeArray();
    const amrex::Real idx = 1.0 / dx[0];
    const amrex::Real idy = 1.0 / dx[1];
    const amrex::Real idz = 1.0 / dx[2];

    for (amrex::MFIter mfi(phi_mf); mfi.isValid(); ++mfi)
    {
        const amrex::Box& bx = mfi.validbox();
        auto const& phi = phi_mf.const_array(mfi);
        auto const& out = out_mf.array(mfi);
        auto const& fx = fx_mf.const_array(mfi);
        auto const& fy = fy_mf.const_array(mfi);
        auto const& fz = fz_mf.const_array(mfi);

        amrex::ParallelFor(
            bx,
            [=] AMREX_GPU_DEVICE(int i, int j, int k)
            {
                amrex::Real divF = 0.0;
                // x
                {
                    amrex::Real fl = fx(i, j, k), fr = fx(i + 1, j, k);
                    amrex::Real sm1 = phi(i - 1, j, k), s0 = phi(i, j, k), sp1 = phi(i + 1, j, k);
                    amrex::Real sm2 = phi(i - 2, j, k), sp2 = phi(i + 2, j, k);
                    amrex::Real dl = s0 - sm1;
                    amrex::Real pl = (fl >= 0) ? sm1 + 0.5 * vanleerCorr(sm1 - sm2, dl)
                                               : s0 - 0.5 * vanleerCorr(sp1 - s0, dl);
                    amrex::Real dr = sp1 - s0;
                    amrex::Real pr = (fr >= 0) ? s0 + 0.5 * vanleerCorr(s0 - sm1, dr)
                                               : sp1 - 0.5 * vanleerCorr(sp2 - sp1, dr);
                    divF += (fr * pr - fl * pl) * idx;
                }
                // y
                {
                    amrex::Real fl = fy(i, j, k), fr = fy(i, j + 1, k);
                    amrex::Real sm1 = phi(i, j - 1, k), s0 = phi(i, j, k), sp1 = phi(i, j + 1, k);
                    amrex::Real sm2 = phi(i, j - 2, k), sp2 = phi(i, j + 2, k);
                    amrex::Real dl = s0 - sm1;
                    amrex::Real pl = (fl >= 0) ? sm1 + 0.5 * vanleerCorr(sm1 - sm2, dl)
                                               : s0 - 0.5 * vanleerCorr(sp1 - s0, dl);
                    amrex::Real dr = sp1 - s0;
                    amrex::Real pr = (fr >= 0) ? s0 + 0.5 * vanleerCorr(s0 - sm1, dr)
                                               : sp1 - 0.5 * vanleerCorr(sp2 - sp1, dr);
                    divF += (fr * pr - fl * pl) * idy;
                }
                // z
                {
                    amrex::Real fl = fz(i, j, k), fr = fz(i, j, k + 1);
                    amrex::Real sm1 = phi(i, j, k - 1), s0 = phi(i, j, k), sp1 = phi(i, j, k + 1);
                    amrex::Real sm2 = phi(i, j, k - 2), sp2 = phi(i, j, k + 2);
                    amrex::Real dl = s0 - sm1;
                    amrex::Real pl = (fl >= 0) ? sm1 + 0.5 * vanleerCorr(sm1 - sm2, dl)
                                               : s0 - 0.5 * vanleerCorr(sp1 - s0, dl);
                    amrex::Real dr = sp1 - s0;
                    amrex::Real pr = (fr >= 0) ? s0 + 0.5 * vanleerCorr(s0 - sm1, dr)
                                               : sp1 - 0.5 * vanleerCorr(sp2 - sp1, dr);
                    divF += (fr * pr - fl * pl) * idz;
                }
                out(i, j, k) = divF;
            }
        );
    }
}

// QUICK divergence source term.
static void divQuick(
    amrex::MultiFab& out_mf,
    const amrex::MultiFab& phi_mf,
    const amrex::MultiFab& fx_mf,
    const amrex::MultiFab& fy_mf,
    const amrex::MultiFab& fz_mf,
    const amrex::Geometry& geom
)
{
    const auto dx = geom.CellSizeArray();
    const amrex::Real idx = 1.0 / dx[0];
    const amrex::Real idy = 1.0 / dx[1];
    const amrex::Real idz = 1.0 / dx[2];

    for (amrex::MFIter mfi(phi_mf); mfi.isValid(); ++mfi)
    {
        const amrex::Box& bx = mfi.validbox();
        auto const& phi = phi_mf.const_array(mfi);
        auto const& out = out_mf.array(mfi);
        auto const& fx = fx_mf.const_array(mfi);
        auto const& fy = fy_mf.const_array(mfi);
        auto const& fz = fz_mf.const_array(mfi);

        amrex::ParallelFor(
            bx,
            [=] AMREX_GPU_DEVICE(int i, int j, int k)
            {
                amrex::Real divF = 0.0;
                // x
                {
                    amrex::Real fl = fx(i, j, k), fr = fx(i + 1, j, k);
                    amrex::Real sm2 = phi(i - 2, j, k), sm1 = phi(i - 1, j, k), s0 = phi(i, j, k);
                    amrex::Real sp1 = phi(i + 1, j, k), sp2 = phi(i + 2, j, k);
                    amrex::Real pl = (fl >= 0) ? 0.375 * s0 + 0.75 * sm1 - 0.125 * sm2
                                               : 0.375 * sm1 + 0.75 * s0 - 0.125 * sp1;
                    amrex::Real pr = (fr >= 0) ? 0.375 * sp1 + 0.75 * s0 - 0.125 * sm1
                                               : 0.375 * s0 + 0.75 * sp1 - 0.125 * sp2;
                    divF += (fr * pr - fl * pl) * idx;
                }
                // y
                {
                    amrex::Real fl = fy(i, j, k), fr = fy(i, j + 1, k);
                    amrex::Real sm2 = phi(i, j - 2, k), sm1 = phi(i, j - 1, k), s0 = phi(i, j, k);
                    amrex::Real sp1 = phi(i, j + 1, k), sp2 = phi(i, j + 2, k);
                    amrex::Real pl = (fl >= 0) ? 0.375 * s0 + 0.75 * sm1 - 0.125 * sm2
                                               : 0.375 * sm1 + 0.75 * s0 - 0.125 * sp1;
                    amrex::Real pr = (fr >= 0) ? 0.375 * sp1 + 0.75 * s0 - 0.125 * sm1
                                               : 0.375 * s0 + 0.75 * sp1 - 0.125 * sp2;
                    divF += (fr * pr - fl * pl) * idy;
                }
                // z
                {
                    amrex::Real fl = fz(i, j, k), fr = fz(i, j, k + 1);
                    amrex::Real sm2 = phi(i, j, k - 2), sm1 = phi(i, j, k - 1), s0 = phi(i, j, k);
                    amrex::Real sp1 = phi(i, j, k + 1), sp2 = phi(i, j, k + 2);
                    amrex::Real pl = (fl >= 0) ? 0.375 * s0 + 0.75 * sm1 - 0.125 * sm2
                                               : 0.375 * sm1 + 0.75 * s0 - 0.125 * sp1;
                    amrex::Real pr = (fr >= 0) ? 0.375 * sp1 + 0.75 * s0 - 0.125 * sm1
                                               : 0.375 * s0 + 0.75 * sp1 - 0.125 * sp2;
                    divF += (fr * pr - fl * pl) * idz;
                }
                out(i, j, k) = divF;
            }
        );
    }
}


void registerStencilKernels(nb::module_& m)
{
    m.def(
        "div_upwind",
        [](amrex::MultiFab& out,
           const amrex::MultiFab& phi,
           const amrex::MultiFab& fx,
           const amrex::MultiFab& fy,
           const amrex::MultiFab& fz,
           const amrex::Geometry& geom) { divUpwind(out, phi, fx, fy, fz, geom); },
        nb::arg("out"),
        nb::arg("phi"),
        nb::arg("fx"),
        nb::arg("fy"),
        nb::arg("fz"),
        nb::arg("geom"),
        "Upwind divergence source term (C++ baseline)."
    );

    m.def(
        "div_linear",
        [](amrex::MultiFab& out,
           const amrex::MultiFab& phi,
           const amrex::MultiFab& fx,
           const amrex::MultiFab& fy,
           const amrex::MultiFab& fz,
           const amrex::Geometry& geom) { divLinear(out, phi, fx, fy, fz, geom); },
        nb::arg("out"),
        nb::arg("phi"),
        nb::arg("fx"),
        nb::arg("fy"),
        nb::arg("fz"),
        nb::arg("geom"),
        "Linear divergence source term (C++ baseline)."
    );

    m.def(
        "div_vanleer",
        [](amrex::MultiFab& out,
           const amrex::MultiFab& phi,
           const amrex::MultiFab& fx,
           const amrex::MultiFab& fy,
           const amrex::MultiFab& fz,
           const amrex::Geometry& geom) { divVanLeer(out, phi, fx, fy, fz, geom); },
        nb::arg("out"),
        nb::arg("phi"),
        nb::arg("fx"),
        nb::arg("fy"),
        nb::arg("fz"),
        nb::arg("geom"),
        "VanLeer divergence source term (C++ baseline)."
    );

    m.def(
        "div_quick",
        [](amrex::MultiFab& out,
           const amrex::MultiFab& phi,
           const amrex::MultiFab& fx,
           const amrex::MultiFab& fy,
           const amrex::MultiFab& fz,
           const amrex::Geometry& geom) { divQuick(out, phi, fx, fy, fz, geom); },
        nb::arg("out"),
        nb::arg("phi"),
        nb::arg("fx"),
        nb::arg("fy"),
        nb::arg("fz"),
        nb::arg("geom"),
        "QUICK divergence source term (C++ baseline)."
    );

    m.def(
        "div_upwind_acc",
        [](amrex::MultiFab& out,
           const amrex::MultiFab& phi,
           const amrex::MultiFab& fx,
           const amrex::MultiFab& fy,
           const amrex::MultiFab& fz,
           const amrex::Geometry& geom,
           double coeff,
           int ncomp) { divUpwindAcc(out, phi, fx, fy, fz, geom, coeff, ncomp); },
        nb::arg("out"),
        nb::arg("phi"),
        nb::arg("fx"),
        nb::arg("fy"),
        nb::arg("fz"),
        nb::arg("geom"),
        nb::arg("coeff") = 1.0,
        nb::arg("ncomp") = 1,
        "Accumulate coeff*div_upwind(phi) into out (out +=), ncomp-general."
    );

    m.def(
        "div_linear_acc",
        [](amrex::MultiFab& out,
           const amrex::MultiFab& phi,
           const amrex::MultiFab& fx,
           const amrex::MultiFab& fy,
           const amrex::MultiFab& fz,
           const amrex::Geometry& geom,
           double coeff,
           int ncomp) { divLinearAcc(out, phi, fx, fy, fz, geom, coeff, ncomp); },
        nb::arg("out"),
        nb::arg("phi"),
        nb::arg("fx"),
        nb::arg("fy"),
        nb::arg("fz"),
        nb::arg("geom"),
        nb::arg("coeff") = 1.0,
        nb::arg("ncomp") = 1,
        "Accumulate coeff*div_linear(phi) into out (out +=), ncomp-general."
    );

    m.def(
        "div_vanleer_acc",
        [](amrex::MultiFab& out,
           const amrex::MultiFab& phi,
           const amrex::MultiFab& fx,
           const amrex::MultiFab& fy,
           const amrex::MultiFab& fz,
           const amrex::Geometry& geom,
           double coeff,
           int ncomp) { divVanLeerAcc(out, phi, fx, fy, fz, geom, coeff, ncomp); },
        nb::arg("out"),
        nb::arg("phi"),
        nb::arg("fx"),
        nb::arg("fy"),
        nb::arg("fz"),
        nb::arg("geom"),
        nb::arg("coeff") = 1.0,
        nb::arg("ncomp") = 1,
        "Accumulate coeff*div_vanleer(phi) into out (out +=), ncomp-general."
    );

    m.def(
        "div_quick_acc",
        [](amrex::MultiFab& out,
           const amrex::MultiFab& phi,
           const amrex::MultiFab& fx,
           const amrex::MultiFab& fy,
           const amrex::MultiFab& fz,
           const amrex::Geometry& geom,
           double coeff,
           int ncomp) { divQuickAcc(out, phi, fx, fy, fz, geom, coeff, ncomp); },
        nb::arg("out"),
        nb::arg("phi"),
        nb::arg("fx"),
        nb::arg("fy"),
        nb::arg("fz"),
        nb::arg("geom"),
        nb::arg("coeff") = 1.0,
        nb::arg("ncomp") = 1,
        "Accumulate coeff*div_quick(phi) into out (out +=), ncomp-general."
    );

    m.def(
        "laplacian_acc",
        [](amrex::MultiFab& out,
           const amrex::MultiFab& phi,
           const amrex::Geometry& geom,
           double coeff,
           int ncomp) { laplacianAcc(out, phi, geom, coeff, ncomp); },
        nb::arg("out"),
        nb::arg("phi"),
        nb::arg("geom"),
        nb::arg("coeff") = 1.0,
        nb::arg("ncomp") = 1,
        "Accumulate coeff*laplacian(phi) into out (constant gamma folded into coeff)."
    );

    m.def(
        "grad_acc",
        [](amrex::MultiFab& out,
           const amrex::MultiFab& phi,
           const amrex::Geometry& geom,
           double coeff) { gradAcc(out, phi, geom, coeff); },
        nb::arg("out"),
        nb::arg("phi"),
        nb::arg("geom"),
        nb::arg("coeff") = 1.0,
        "Accumulate coeff*grad(phi) (scalar->3-vector, central difference) into out."
    );

    m.def(
        "source_acc",
        [](amrex::MultiFab& out, const amrex::MultiFab& phi, double coeff, int ncomp)
        { sourceAcc(out, phi, coeff, ncomp); },
        nb::arg("out"),
        nb::arg("phi"),
        nb::arg("coeff") = 1.0,
        nb::arg("ncomp") = 1,
        "Accumulate coeff*phi into out (pointwise source term)."
    );

    m.def(
        "euler_update",
        [](amrex::MultiFab& phi, const amrex::MultiFab& src, double dt_over_coeff, int ncomp)
        { eulerUpdate(phi, src, dt_over_coeff, ncomp); },
        nb::arg("phi"),
        nb::arg("src"),
        nb::arg("dt_over_coeff"),
        nb::arg("ncomp") = 1,
        "Generic forward-Euler axpy: phi -= dt_over_coeff*src."
    );
}
