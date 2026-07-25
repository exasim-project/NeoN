// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <AMReX_GpuLaunch.H>
#include <AMReX_Math.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Reduce.H>

#include <algorithm>
#include <cmath>

#include "profiling.hpp"

namespace blockamr::solvers
{

// ---------------------------------------------------------------------------
// Native geometric-multigrid V-cycle preconditioner (GmgPrecond) kernels.
// Built from AMReX primitives on the face-coefficient operator only — no
// MLLinOp/MLMG anywhere in this path. Device kernels are namespace-scope free
// functions (nvcc: no extended __device__ lambdas in private/protected
// members) with host-loop twins for the ReferenceExecutor path.
//
// Every kernel is templated on the level value type T (double for the default
// FP64 hierarchy, float for the M5 gmg_precision="fp32" hierarchy): the whole
// V-cycle — level coefficients, sol/rhs work fields, smoother, residual /
// restriction / prolongation, ghost fills and the λmax power iteration — runs in
// T while the outer CG/operator stays FP64. GmgFab<T> is the level fab type.
// ---------------------------------------------------------------------------

template<class T>
using GmgFab = amrex::FabArray<amrex::BaseFab<T>>;

// Tiny |diagonal| floor guarding the RB-GS in-place division (skip rather than
// divide by ~0). Per value type so the double path keeps its 1e-300 floor
// exactly while the float path uses a representable one (1e-300 is not a valid
// float literal).
template<class T>
AMREX_GPU_HOST_DEVICE constexpr T gmgDiagFloor();
template<>
AMREX_GPU_HOST_DEVICE constexpr double gmgDiagFloor<double>()
{
    return 1e-300;
}
template<>
AMREX_GPU_HOST_DEVICE constexpr float gmgDiagFloor<float>()
{
    return 1e-30f;
}

// Copy src (any FabArray, e.g. the caller's FP64 MultiFab or a same-type level
// fab) into the T-valued dst, converting per cell over dst's valid box. Replaces
// MultiFab::Copy on the FP32 path (which requires matching value types); for
// T=double it is an exact copy, so the FP64 path is numerically unchanged.
template<class T, class SRC>
void gmgConvertCopyDevice(GmgFab<T>& dst, const SRC& src)
{
    for (amrex::MFIter mfi(dst); mfi.isValid(); ++mfi)
    {
        const amrex::Box& vbx = mfi.validbox();
        const auto d = dst.array(mfi);
        const auto s = src.const_array(mfi);
        amrex::ParallelFor(
            vbx,
            [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
            { d(i, j, k) = static_cast<T>(s(i, j, k)); }
        );
    }
}

template<class T, class SRC>
void gmgConvertCopyHost(GmgFab<T>& dst, const SRC& src)
{
    for (amrex::MFIter mfi(dst); mfi.isValid(); ++mfi)
    {
        const amrex::Box& vbx = mfi.validbox();
        const auto d = dst.array(mfi);
        const auto s = src.const_array(mfi);
        const auto lo = amrex::lbound(vbx);
        const auto hi = amrex::ubound(vbx);
        for (int k = lo.z; k <= hi.z; ++k)
        {
            for (int j = lo.y; j <= hi.y; ++j)
            {
                for (int i = lo.x; i <= hi.x; ++i)
                {
                    d(i, j, k) = static_cast<T>(s(i, j, k));
                }
            }
        }
    }
}

// dst += src, per cell over dst's valid box, converting through dst's value_type.
// dst is any FabArray (the caller's FP64 MultiFab); src is a T-valued level fab.
// The native stationary solver adds the (possibly FP32) V-cycle correction back
// onto the FP64 solution; for both double it is a plain in-place add.
template<class DST, class T>
void gmgConvertAddDevice(DST& dst, const GmgFab<T>& src)
{
    using DT = typename DST::value_type;
    for (amrex::MFIter mfi(dst); mfi.isValid(); ++mfi)
    {
        const amrex::Box& vbx = mfi.validbox();
        const auto d = dst.array(mfi);
        const auto s = src.const_array(mfi);
        amrex::ParallelFor(
            vbx,
            [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
            { d(i, j, k) += static_cast<DT>(s(i, j, k)); }
        );
    }
}

template<class DST, class T>
void gmgConvertAddHost(DST& dst, const GmgFab<T>& src)
{
    using DT = typename DST::value_type;
    for (amrex::MFIter mfi(dst); mfi.isValid(); ++mfi)
    {
        const amrex::Box& vbx = mfi.validbox();
        const auto d = dst.array(mfi);
        const auto s = src.const_array(mfi);
        const auto lo = amrex::lbound(vbx);
        const auto hi = amrex::ubound(vbx);
        for (int k = lo.z; k <= hi.z; ++k)
        {
            for (int j = lo.y; j <= hi.y; ++j)
            {
                for (int i = lo.x; i <= hi.x; ++i)
                {
                    d(i, j, k) += static_cast<DT>(s(i, j, k));
                }
            }
        }
    }
}

// Fused residual + convert-scatter + norm for the native GMG stationary solver
// (M3 target 3). Computes r = rhs - A*sol - shift in DOUBLE (shift is the
// nullspace-projection constant, 0 when not projecting) and stores it (cast to T)
// straight into the L0 rhs fab `out` — no separate FP64 residual MultiFab and no
// convert-scatter pass (M3 3a). The norm is a SECOND, light kernel reducing `out`.
//
// Why two kernels, not one: folding the sum(r^2) reduction INTO the heavy stencil
// kernel (10 coefficient/field Array4 + double arithmetic) was measured to cost
// ~1.0 ms/iter at 256^3 — the reduction machinery spills the register-bound
// kernel and slows the whole pass, exceeding the 0.34+0.54 ms it saves. A separate
// reduction over the freshly-written `out` is only ~0.20 ms/iter (light kernel,
// stays at the bandwidth roofline) and reuses the just-cached data.
//
// Precision of the norm: in the DEFAULT fp64 hierarchy (T=double) `out` holds the
// exact double residual, so the reduced norm is bit-exact FP64 — the convergence
// authority is unchanged. In the fp32 hierarchy (T=float) `out` holds the residual
// rounded to float; the reduced norm therefore carries ~6e-8 relative rounding,
// far below the ~10x per-cycle residual drop, so the stopping cycle is unchanged
// (verified: iters and converged answer identical to the FP64-norm path).
//
// BOTH norms come out of the one reduction: the 2-norm's sum of squares and the
// inf-norm's max|r|, so the native stationary solver can stop in either
// (norm="l2" | "linf", the latter MLMG's — see stop_norm_inf.hpp) without a
// second pass over the residual. The extra ReduceOpMax is register-only work in
// an already bandwidth-bound kernel. Device + host twins.
struct ResidNorms
{
    double sumsq;  // sum r_i^2 — caller takes the sqrt for ||r||_2
    double maxabs; // max |r_i| — ||r||_inf
};

template<class T>
ResidNorms faceCoeffResidScatterNormDevice(
    const amrex::MultiFab& sol,
    const amrex::MultiFab& rhs,
    const amrex::MultiFab& ux,
    const amrex::MultiFab& lx,
    const amrex::MultiFab& uy,
    const amrex::MultiFab& ly,
    const amrex::MultiFab& uz,
    const amrex::MultiFab& lz,
    const amrex::MultiFab& alpha,
    double shift,
    GmgFab<T>& out
)
{
    ResidNorms res {};
    {
        prof::Timer t("gmg.solve.residkern");
        for (amrex::MFIter mfi(out); mfi.isValid(); ++mfi)
        {
            const amrex::Box& vbx = mfi.validbox();
            const auto psi = sol.const_array(mfi);
            const auto bb = rhs.const_array(mfi);
            const auto o = out.array(mfi);
            const auto ax = ux.const_array(mfi);
            const auto lxa = lx.const_array(mfi);
            const auto ay = uy.const_array(mfi);
            const auto lya = ly.const_array(mfi);
            const auto az = uz.const_array(mfi);
            const auto lza = lz.const_array(mfi);
            const auto al = alpha.const_array(mfi);
            amrex::ParallelFor(
                vbx,
                [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                {
                    const double aE = ax(i + 1, j, k);
                    const double aW = lxa(i, j, k);
                    const double aN = ay(i, j + 1, k);
                    const double aS = lya(i, j, k);
                    const double aT = az(i, j, k + 1);
                    const double aB = lza(i, j, k);
                    const double offd = aE * psi(i + 1, j, k) + aW * psi(i - 1, j, k)
                                      + aN * psi(i, j + 1, k) + aS * psi(i, j - 1, k)
                                      + aT * psi(i, j, k + 1) + aB * psi(i, j, k - 1);
                    const double diag = al(i, j, k) - (aE + aW + aN + aS + aT + aB);
                    const double r = bb(i, j, k) - (diag * psi(i, j, k) + offd) - shift;
                    o(i, j, k) = static_cast<T>(r);
                }
            );
        }
    }
    {
        prof::Timer t("gmg.solve.normkern");
        const auto o_ma = out.const_arrays();
        const auto both = amrex::ParReduce(
            amrex::TypeList<amrex::ReduceOpSum, amrex::ReduceOpMax> {},
            amrex::TypeList<double, double> {},
            out,
            amrex::IntVect(0),
            [=] AMREX_GPU_DEVICE(int box, int i, int j, int k) -> amrex::GpuTuple<double, double>
            {
                const double v = static_cast<double>(o_ma[box](i, j, k));
                return {v * v, amrex::Math::abs(v)};
            }
        );
        res.sumsq = amrex::get<0>(both);
        res.maxabs = amrex::get<1>(both);
    }
    return res;
}

template<class T>
ResidNorms faceCoeffResidScatterNormHost(
    const amrex::MultiFab& sol,
    const amrex::MultiFab& rhs,
    const amrex::MultiFab& ux,
    const amrex::MultiFab& lx,
    const amrex::MultiFab& uy,
    const amrex::MultiFab& ly,
    const amrex::MultiFab& uz,
    const amrex::MultiFab& lz,
    const amrex::MultiFab& alpha,
    double shift,
    GmgFab<T>& out
)
{
    ResidNorms res {};
    for (amrex::MFIter mfi(out); mfi.isValid(); ++mfi)
    {
        const amrex::Box& vbx = mfi.validbox();
        const auto psi = sol.const_array(mfi);
        const auto bb = rhs.const_array(mfi);
        const auto o = out.array(mfi);
        const auto ax = ux.const_array(mfi);
        const auto lxa = lx.const_array(mfi);
        const auto ay = uy.const_array(mfi);
        const auto lya = ly.const_array(mfi);
        const auto az = uz.const_array(mfi);
        const auto lza = lz.const_array(mfi);
        const auto al = alpha.const_array(mfi);
        const auto lo = amrex::lbound(vbx);
        const auto hi = amrex::ubound(vbx);
        for (int k = lo.z; k <= hi.z; ++k)
        {
            for (int j = lo.y; j <= hi.y; ++j)
            {
                for (int i = lo.x; i <= hi.x; ++i)
                {
                    const double aE = ax(i + 1, j, k);
                    const double aW = lxa(i, j, k);
                    const double aN = ay(i, j + 1, k);
                    const double aS = lya(i, j, k);
                    const double aT = az(i, j, k + 1);
                    const double aB = lza(i, j, k);
                    const double offd = aE * psi(i + 1, j, k) + aW * psi(i - 1, j, k)
                                      + aN * psi(i, j + 1, k) + aS * psi(i, j - 1, k)
                                      + aT * psi(i, j, k + 1) + aB * psi(i, j, k - 1);
                    const double diag = al(i, j, k) - (aE + aW + aN + aS + aT + aB);
                    const double r = bb(i, j, k) - (diag * psi(i, j, k) + offd) - shift;
                    o(i, j, k) = static_cast<T>(r);
                    // Reduce the STORED value (like the device twin's separate
                    // ParReduce over `out`) so reference and cuda give an identical
                    // norm: exact FP64 for T=double, fp32-rounded for T=float.
                    const double v = static_cast<double>(o(i, j, k));
                    res.sumsq += v * v;
                    res.maxabs = std::max(res.maxabs, std::abs(v));
                }
            }
        }
    }
    return res;
}

// ||mf||_2 over the valid region (0 ghost), accumulated in the fab's value_type
// (single-box/single-rank hierarchy; used only by the setup power iteration).
template<class T>
double gmgNorm2(const GmgFab<T>& mf)
{
    const T sq = amrex::ReduceSum(
        mf,
        amrex::IntVect(0),
        [=] AMREX_GPU_HOST_DEVICE(const amrex::Box& bx, const amrex::Array4<const T>& a) -> T
        {
            T s = 0;
            const auto lo = amrex::lbound(bx);
            const auto hi = amrex::ubound(bx);
            for (int k = lo.z; k <= hi.z; ++k)
            {
                for (int j = lo.y; j <= hi.y; ++j)
                {
                    for (int i = lo.x; i <= hi.x; ++i)
                    {
                        s += a(i, j, k) * a(i, j, k);
                    }
                }
            }
            return s;
        }
    );
    return std::sqrt(static_cast<double>(sq));
}

// One red-black successive-over-relaxation colour pass: cells with (i+j+k)
// parity `parity` are updated in place towards the exact Gauss-Seidel value
// gs = (rhs - off) / D, with D = alpha - sum(face coeffs) recomputed on the fly
// (tiny |D| guarded to no update). The update is
//   sol <- sol + omega * (gs - sol),
// so omega = 1 is plain Gauss-Seidel (the previous behaviour, bit-for-bit) and
// omega > 1 over-relaxes — MLMG's abec_gsrb uses omega = 1.15. The 7-point
// stencil only couples opposite colours, so the in-place update is race-free.
// sol's ghosts must be refreshed before EACH colour pass.
template<class T>
void gmgGsColorDevice(
    GmgFab<T>& sol,
    const GmgFab<T>& rhs,
    const GmgFab<T>& ux,
    const GmgFab<T>& lx,
    const GmgFab<T>& uy,
    const GmgFab<T>& ly,
    const GmgFab<T>& uz,
    const GmgFab<T>& lz,
    const GmgFab<T>& alpha,
    int parity,
    double omega
)
{
    const T om = static_cast<T>(omega);
    for (amrex::MFIter mfi(rhs); mfi.isValid(); ++mfi)
    {
        const amrex::Box& vbx = mfi.validbox();
        const auto psi = sol.array(mfi);
        const auto b = rhs.const_array(mfi);
        const auto ax = ux.const_array(mfi);
        const auto lxa = lx.const_array(mfi);
        const auto ay = uy.const_array(mfi);
        const auto lya = ly.const_array(mfi);
        const auto az = uz.const_array(mfi);
        const auto lza = lz.const_array(mfi);
        const auto al = alpha.const_array(mfi);
        amrex::ParallelFor(
            vbx,
            [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
            {
                if (((i + j + k) & 1) != parity)
                {
                    return;
                }
                const T aE = ax(i + 1, j, k);
                const T aW = lxa(i, j, k);
                const T aN = ay(i, j + 1, k);
                const T aS = lya(i, j, k);
                const T aT = az(i, j, k + 1);
                const T aB = lza(i, j, k);
                const T off = aE * psi(i + 1, j, k) + aW * psi(i - 1, j, k) + aN * psi(i, j + 1, k)
                            + aS * psi(i, j - 1, k) + aT * psi(i, j, k + 1) + aB * psi(i, j, k - 1);
                const T diag = al(i, j, k) - (aE + aW + aN + aS + aT + aB);
                if (amrex::Math::abs(diag) > gmgDiagFloor<T>())
                {
                    const T gs = (b(i, j, k) - off) / diag;
                    psi(i, j, k) += om * (gs - psi(i, j, k));
                }
            }
        );
    }
}

template<class T>
void gmgGsColorHost(
    GmgFab<T>& sol,
    const GmgFab<T>& rhs,
    const GmgFab<T>& ux,
    const GmgFab<T>& lx,
    const GmgFab<T>& uy,
    const GmgFab<T>& ly,
    const GmgFab<T>& uz,
    const GmgFab<T>& lz,
    const GmgFab<T>& alpha,
    int parity,
    double omega
)
{
    const T om = static_cast<T>(omega);
    for (amrex::MFIter mfi(rhs); mfi.isValid(); ++mfi)
    {
        const amrex::Box& vbx = mfi.validbox();
        const auto psi = sol.array(mfi);
        const auto b = rhs.const_array(mfi);
        const auto ax = ux.const_array(mfi);
        const auto lxa = lx.const_array(mfi);
        const auto ay = uy.const_array(mfi);
        const auto lya = ly.const_array(mfi);
        const auto az = uz.const_array(mfi);
        const auto lza = lz.const_array(mfi);
        const auto al = alpha.const_array(mfi);
        const auto lo = amrex::lbound(vbx);
        const auto hi = amrex::ubound(vbx);
        for (int k = lo.z; k <= hi.z; ++k)
        {
            for (int j = lo.y; j <= hi.y; ++j)
            {
                for (int i = lo.x; i <= hi.x; ++i)
                {
                    if (((i + j + k) & 1) != parity)
                    {
                        continue;
                    }
                    const T aE = ax(i + 1, j, k);
                    const T aW = lxa(i, j, k);
                    const T aN = ay(i, j + 1, k);
                    const T aS = lya(i, j, k);
                    const T aT = az(i, j, k + 1);
                    const T aB = lza(i, j, k);
                    const T off = aE * psi(i + 1, j, k) + aW * psi(i - 1, j, k)
                                + aN * psi(i, j + 1, k) + aS * psi(i, j - 1, k)
                                + aT * psi(i, j, k + 1) + aB * psi(i, j, k - 1);
                    const T diag = al(i, j, k) - (aE + aW + aN + aS + aT + aB);
                    if (std::abs(diag) > gmgDiagFloor<T>())
                    {
                        const T gs = (b(i, j, k) - off) / diag;
                        psi(i, j, k) += om * (gs - psi(i, j, k));
                    }
                }
            }
        }
    }
}

// Volume-average (factor-2) restriction of a cell field: coarse = mean of the
// 8 fine children. Also used to coarsen alpha (a per-volume density). Iterates
// the coarse MF; the fine MF shares the DistributionMapping, so the same MFIter
// index addresses the matching fine box (its BoxArray is refine(coarse, 2)).
template<class T>
void gmgRestrictDevice(const GmgFab<T>& fine, GmgFab<T>& crse)
{
    for (amrex::MFIter mfi(crse); mfi.isValid(); ++mfi)
    {
        const amrex::Box& vbx = mfi.validbox();
        const auto f = fine.const_array(mfi);
        const auto c = crse.array(mfi);
        amrex::ParallelFor(
            vbx,
            [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
            {
                const int i2 = 2 * i, j2 = 2 * j, k2 = 2 * k;
                c(i, j, k) = static_cast<T>(0.125)
                           * (f(i2, j2, k2) + f(i2 + 1, j2, k2) + f(i2, j2 + 1, k2)
                              + f(i2 + 1, j2 + 1, k2) + f(i2, j2, k2 + 1) + f(i2 + 1, j2, k2 + 1)
                              + f(i2, j2 + 1, k2 + 1) + f(i2 + 1, j2 + 1, k2 + 1));
            }
        );
    }
}

template<class T>
void gmgRestrictHost(const GmgFab<T>& fine, GmgFab<T>& crse)
{
    for (amrex::MFIter mfi(crse); mfi.isValid(); ++mfi)
    {
        const amrex::Box& vbx = mfi.validbox();
        const auto f = fine.const_array(mfi);
        const auto c = crse.array(mfi);
        const auto lo = amrex::lbound(vbx);
        const auto hi = amrex::ubound(vbx);
        for (int k = lo.z; k <= hi.z; ++k)
        {
            for (int j = lo.y; j <= hi.y; ++j)
            {
                for (int i = lo.x; i <= hi.x; ++i)
                {
                    const int i2 = 2 * i, j2 = 2 * j, k2 = 2 * k;
                    c(i, j, k) =
                        static_cast<T>(0.125)
                        * (f(i2, j2, k2) + f(i2 + 1, j2, k2) + f(i2, j2 + 1, k2)
                           + f(i2 + 1, j2 + 1, k2) + f(i2, j2, k2 + 1) + f(i2 + 1, j2, k2 + 1)
                           + f(i2, j2 + 1, k2 + 1) + f(i2 + 1, j2 + 1, k2 + 1));
                }
            }
        }
    }
}

// Coarsen a face-coefficient field in direction `dir`: coarse face i_c covers
// fine face 2*i_c with the 2x2 transverse fine faces; a ~ -beta/dx^2, so the
// coarse coefficient is the arithmetic average of those 4 fine coefficients
// (beta averaged) divided by `scale` (dx doubled -> 4 for rediscretisation).
template<class T>
void gmgCoarsenFaceDevice(const GmgFab<T>& fine, GmgFab<T>& crse, int dir, double scale)
{
    int u[3] = {0, 0, 0}, v[3] = {0, 0, 0};
    // The two transverse (cell) directions of face-normal `dir`.
    if (dir == 0)
    {
        u[1] = 1;
        v[2] = 1;
    }
    else if (dir == 1)
    {
        u[0] = 1;
        v[2] = 1;
    }
    else
    {
        u[0] = 1;
        v[1] = 1;
    }
    const int u0 = u[0], u1 = u[1], u2 = u[2];
    const int v0 = v[0], v1 = v[1], v2 = v[2];
    const T w = static_cast<T>(0.25 / scale);
    for (amrex::MFIter mfi(crse); mfi.isValid(); ++mfi)
    {
        const amrex::Box& vbx = mfi.validbox();
        const auto f = fine.const_array(mfi);
        const auto c = crse.array(mfi);
        amrex::ParallelFor(
            vbx,
            [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
            {
                const int i2 = 2 * i, j2 = 2 * j, k2 = 2 * k;
                c(i, j, k) =
                    w
                    * (f(i2, j2, k2) + f(i2 + u0, j2 + u1, k2 + u2) + f(i2 + v0, j2 + v1, k2 + v2)
                       + f(i2 + u0 + v0, j2 + u1 + v1, k2 + u2 + v2));
            }
        );
    }
}

template<class T>
void gmgCoarsenFaceHost(const GmgFab<T>& fine, GmgFab<T>& crse, int dir, double scale)
{
    int u[3] = {0, 0, 0}, v[3] = {0, 0, 0};
    if (dir == 0)
    {
        u[1] = 1;
        v[2] = 1;
    }
    else if (dir == 1)
    {
        u[0] = 1;
        v[2] = 1;
    }
    else
    {
        u[0] = 1;
        v[1] = 1;
    }
    const T w = static_cast<T>(0.25 / scale);
    for (amrex::MFIter mfi(crse); mfi.isValid(); ++mfi)
    {
        const amrex::Box& vbx = mfi.validbox();
        const auto f = fine.const_array(mfi);
        const auto c = crse.array(mfi);
        const auto lo = amrex::lbound(vbx);
        const auto hi = amrex::ubound(vbx);
        for (int k = lo.z; k <= hi.z; ++k)
        {
            for (int j = lo.y; j <= hi.y; ++j)
            {
                for (int i = lo.x; i <= hi.x; ++i)
                {
                    const int i2 = 2 * i, j2 = 2 * j, k2 = 2 * k;
                    c(i, j, k) = w
                               * (f(i2, j2, k2) + f(i2 + u[0], j2 + u[1], k2 + u[2])
                                  + f(i2 + v[0], j2 + v[1], k2 + v[2])
                                  + f(i2 + u[0] + v[0], j2 + u[1] + v[1], k2 + u[2] + v[2]));
                }
            }
        }
    }
}

// Piecewise-constant prolongation + correction: fine cell += coarse parent
// value (the adjoint of the volume-average restriction, up to the 1/8 factor).
template<class T>
void gmgProlongAddDevice(const GmgFab<T>& crse, GmgFab<T>& fine)
{
    for (amrex::MFIter mfi(fine); mfi.isValid(); ++mfi)
    {
        const amrex::Box& vbx = mfi.validbox();
        const auto c = crse.const_array(mfi);
        const auto f = fine.array(mfi);
        amrex::ParallelFor(
            vbx,
            [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
            { f(i, j, k) += c(amrex::coarsen(i, 2), amrex::coarsen(j, 2), amrex::coarsen(k, 2)); }
        );
    }
}

template<class T>
void gmgProlongAddHost(const GmgFab<T>& crse, GmgFab<T>& fine)
{
    for (amrex::MFIter mfi(fine); mfi.isValid(); ++mfi)
    {
        const amrex::Box& vbx = mfi.validbox();
        const auto c = crse.const_array(mfi);
        const auto f = fine.array(mfi);
        const auto lo = amrex::lbound(vbx);
        const auto hi = amrex::ubound(vbx);
        for (int k = lo.z; k <= hi.z; ++k)
        {
            for (int j = lo.y; j <= hi.y; ++j)
            {
                for (int i = lo.x; i <= hi.x; ++i)
                {
                    f(i, j, k) +=
                        c(amrex::coarsen(i, 2), amrex::coarsen(j, 2), amrex::coarsen(k, 2));
                }
            }
        }
    }
}

// Fused residual + volume-average restriction: coarse rhs cell = mean of the 8
// fine residuals r = rhs - A sol, each computed on the fly. Iterates the coarse
// box (fine sol's ghosts must be filled). Saves the full fine-grid resid
// read+write of the separate residual + restriction passes (M4 item 3).
template<class T>
void gmgResidRestrictDevice(
    const GmgFab<T>& sol,
    const GmgFab<T>& rhs,
    GmgFab<T>& crhs,
    const GmgFab<T>& ux,
    const GmgFab<T>& lx,
    const GmgFab<T>& uy,
    const GmgFab<T>& ly,
    const GmgFab<T>& uz,
    const GmgFab<T>& lz,
    const GmgFab<T>& alpha
)
{
    for (amrex::MFIter mfi(crhs); mfi.isValid(); ++mfi)
    {
        const amrex::Box& vbx = mfi.validbox();
        const auto psi = sol.const_array(mfi);
        const auto b = rhs.const_array(mfi);
        const auto cr = crhs.array(mfi);
        const auto ax = ux.const_array(mfi);
        const auto lxa = lx.const_array(mfi);
        const auto ay = uy.const_array(mfi);
        const auto lya = ly.const_array(mfi);
        const auto az = uz.const_array(mfi);
        const auto lza = lz.const_array(mfi);
        const auto al = alpha.const_array(mfi);
        amrex::ParallelFor(
            vbx,
            [=] AMREX_GPU_DEVICE(int ic, int jc, int kc) noexcept
            {
                T acc = 0;
                for (int dk = 0; dk < 2; ++dk)
                {
                    for (int dj = 0; dj < 2; ++dj)
                    {
                        for (int di = 0; di < 2; ++di)
                        {
                            const int i = 2 * ic + di, j = 2 * jc + dj, k = 2 * kc + dk;
                            const T aE = ax(i + 1, j, k);
                            const T aW = lxa(i, j, k);
                            const T aN = ay(i, j + 1, k);
                            const T aS = lya(i, j, k);
                            const T aT = az(i, j, k + 1);
                            const T aB = lza(i, j, k);
                            const T off = aE * psi(i + 1, j, k) + aW * psi(i - 1, j, k)
                                        + aN * psi(i, j + 1, k) + aS * psi(i, j - 1, k)
                                        + aT * psi(i, j, k + 1) + aB * psi(i, j, k - 1);
                            const T diag = al(i, j, k) - (aE + aW + aN + aS + aT + aB);
                            acc += b(i, j, k) - (diag * psi(i, j, k) + off);
                        }
                    }
                }
                cr(ic, jc, kc) = static_cast<T>(0.125) * acc;
            }
        );
    }
}

template<class T>
void gmgResidRestrictHost(
    const GmgFab<T>& sol,
    const GmgFab<T>& rhs,
    GmgFab<T>& crhs,
    const GmgFab<T>& ux,
    const GmgFab<T>& lx,
    const GmgFab<T>& uy,
    const GmgFab<T>& ly,
    const GmgFab<T>& uz,
    const GmgFab<T>& lz,
    const GmgFab<T>& alpha
)
{
    for (amrex::MFIter mfi(crhs); mfi.isValid(); ++mfi)
    {
        const amrex::Box& vbx = mfi.validbox();
        const auto psi = sol.const_array(mfi);
        const auto b = rhs.const_array(mfi);
        const auto cr = crhs.array(mfi);
        const auto ax = ux.const_array(mfi);
        const auto lxa = lx.const_array(mfi);
        const auto ay = uy.const_array(mfi);
        const auto lya = ly.const_array(mfi);
        const auto az = uz.const_array(mfi);
        const auto lza = lz.const_array(mfi);
        const auto al = alpha.const_array(mfi);
        const auto lo = amrex::lbound(vbx);
        const auto hi = amrex::ubound(vbx);
        for (int kc = lo.z; kc <= hi.z; ++kc)
        {
            for (int jc = lo.y; jc <= hi.y; ++jc)
            {
                for (int ic = lo.x; ic <= hi.x; ++ic)
                {
                    T acc = 0;
                    for (int dk = 0; dk < 2; ++dk)
                    {
                        for (int dj = 0; dj < 2; ++dj)
                        {
                            for (int di = 0; di < 2; ++di)
                            {
                                const int i = 2 * ic + di, j = 2 * jc + dj, k = 2 * kc + dk;
                                const T aE = ax(i + 1, j, k);
                                const T aW = lxa(i, j, k);
                                const T aN = ay(i, j + 1, k);
                                const T aS = lya(i, j, k);
                                const T aT = az(i, j, k + 1);
                                const T aB = lza(i, j, k);
                                const T off = aE * psi(i + 1, j, k) + aW * psi(i - 1, j, k)
                                            + aN * psi(i, j + 1, k) + aS * psi(i, j - 1, k)
                                            + aT * psi(i, j, k + 1) + aB * psi(i, j, k - 1);
                                const T diag = al(i, j, k) - (aE + aW + aN + aS + aT + aB);
                                acc += b(i, j, k) - (diag * psi(i, j, k) + off);
                            }
                        }
                    }
                    cr(ic, jc, kc) = static_cast<T>(0.125) * acc;
                }
            }
        }
    }
}

// One fused Jacobi-Chebyshev degree step: computes r = rhs - A sol on the fly
// (sol's ghosts must be filled) and the polynomial increment
// d = cb * D^{-1} r + (readOld ? ca * d : 0), D = alpha - sum(face coeffs). sol
// is NOT written here (its neighbours are read for r) — the caller adds d to sol
// afterwards, so the whole step is Jacobi-like (race-free) and, being a fixed
// polynomial in the symmetric operator, a symmetric linear smoother (CG-safe).
template<class T>
void gmgChebComputeDDevice(
    const GmgFab<T>& sol,
    const GmgFab<T>& rhs,
    const GmgFab<T>& ux,
    const GmgFab<T>& lx,
    const GmgFab<T>& uy,
    const GmgFab<T>& ly,
    const GmgFab<T>& uz,
    const GmgFab<T>& lz,
    const GmgFab<T>& alpha,
    GmgFab<T>& d,
    T ca,
    T cb,
    bool readOld
)
{
    for (amrex::MFIter mfi(rhs); mfi.isValid(); ++mfi)
    {
        const amrex::Box& vbx = mfi.validbox();
        const auto psi = sol.const_array(mfi);
        const auto b = rhs.const_array(mfi);
        const auto dd = d.array(mfi);
        const auto ax = ux.const_array(mfi);
        const auto lxa = lx.const_array(mfi);
        const auto ay = uy.const_array(mfi);
        const auto lya = ly.const_array(mfi);
        const auto az = uz.const_array(mfi);
        const auto lza = lz.const_array(mfi);
        const auto al = alpha.const_array(mfi);
        amrex::ParallelFor(
            vbx,
            [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
            {
                const T aE = ax(i + 1, j, k);
                const T aW = lxa(i, j, k);
                const T aN = ay(i, j + 1, k);
                const T aS = lya(i, j, k);
                const T aT = az(i, j, k + 1);
                const T aB = lza(i, j, k);
                const T off = aE * psi(i + 1, j, k) + aW * psi(i - 1, j, k) + aN * psi(i, j + 1, k)
                            + aS * psi(i, j - 1, k) + aT * psi(i, j, k + 1) + aB * psi(i, j, k - 1);
                const T diag = al(i, j, k) - (aE + aW + aN + aS + aT + aB);
                const T r = b(i, j, k) - (diag * psi(i, j, k) + off);
                T dval = cb * (r / diag);
                if (readOld)
                {
                    dval += ca * dd(i, j, k);
                }
                dd(i, j, k) = dval;
            }
        );
    }
}

template<class T>
void gmgChebComputeDHost(
    const GmgFab<T>& sol,
    const GmgFab<T>& rhs,
    const GmgFab<T>& ux,
    const GmgFab<T>& lx,
    const GmgFab<T>& uy,
    const GmgFab<T>& ly,
    const GmgFab<T>& uz,
    const GmgFab<T>& lz,
    const GmgFab<T>& alpha,
    GmgFab<T>& d,
    T ca,
    T cb,
    bool readOld
)
{
    for (amrex::MFIter mfi(rhs); mfi.isValid(); ++mfi)
    {
        const amrex::Box& vbx = mfi.validbox();
        const auto psi = sol.const_array(mfi);
        const auto b = rhs.const_array(mfi);
        const auto dd = d.array(mfi);
        const auto ax = ux.const_array(mfi);
        const auto lxa = lx.const_array(mfi);
        const auto ay = uy.const_array(mfi);
        const auto lya = ly.const_array(mfi);
        const auto az = uz.const_array(mfi);
        const auto lza = lz.const_array(mfi);
        const auto al = alpha.const_array(mfi);
        const auto lo = amrex::lbound(vbx);
        const auto hi = amrex::ubound(vbx);
        for (int k = lo.z; k <= hi.z; ++k)
        {
            for (int j = lo.y; j <= hi.y; ++j)
            {
                for (int i = lo.x; i <= hi.x; ++i)
                {
                    const T aE = ax(i + 1, j, k);
                    const T aW = lxa(i, j, k);
                    const T aN = ay(i, j + 1, k);
                    const T aS = lya(i, j, k);
                    const T aT = az(i, j, k + 1);
                    const T aB = lza(i, j, k);
                    const T off = aE * psi(i + 1, j, k) + aW * psi(i - 1, j, k)
                                + aN * psi(i, j + 1, k) + aS * psi(i, j - 1, k)
                                + aT * psi(i, j, k + 1) + aB * psi(i, j, k - 1);
                    const T diag = al(i, j, k) - (aE + aW + aN + aS + aT + aB);
                    const T r = b(i, j, k) - (diag * psi(i, j, k) + off);
                    T dval = cb * (r / diag);
                    if (readOld)
                    {
                        dval += ca * dd(i, j, k);
                    }
                    dd(i, j, k) = dval;
                }
            }
        }
    }
}

// out = D^{-1} A v (v's ghosts filled), used by the setup power iteration that
// estimates lambda_max of D^{-1}A per level for the Chebyshev interval.
template<class T>
void gmgDinvApplyDevice(
    const GmgFab<T>& v,
    GmgFab<T>& out,
    const GmgFab<T>& ux,
    const GmgFab<T>& lx,
    const GmgFab<T>& uy,
    const GmgFab<T>& ly,
    const GmgFab<T>& uz,
    const GmgFab<T>& lz,
    const GmgFab<T>& alpha
)
{
    for (amrex::MFIter mfi(out); mfi.isValid(); ++mfi)
    {
        const amrex::Box& vbx = mfi.validbox();
        const auto psi = v.const_array(mfi);
        const auto o = out.array(mfi);
        const auto ax = ux.const_array(mfi);
        const auto lxa = lx.const_array(mfi);
        const auto ay = uy.const_array(mfi);
        const auto lya = ly.const_array(mfi);
        const auto az = uz.const_array(mfi);
        const auto lza = lz.const_array(mfi);
        const auto al = alpha.const_array(mfi);
        amrex::ParallelFor(
            vbx,
            [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
            {
                const T aE = ax(i + 1, j, k);
                const T aW = lxa(i, j, k);
                const T aN = ay(i, j + 1, k);
                const T aS = lya(i, j, k);
                const T aT = az(i, j, k + 1);
                const T aB = lza(i, j, k);
                const T off = aE * psi(i + 1, j, k) + aW * psi(i - 1, j, k) + aN * psi(i, j + 1, k)
                            + aS * psi(i, j - 1, k) + aT * psi(i, j, k + 1) + aB * psi(i, j, k - 1);
                const T diag = al(i, j, k) - (aE + aW + aN + aS + aT + aB);
                o(i, j, k) = (diag * psi(i, j, k) + off) / diag;
            }
        );
    }
}

template<class T>
void gmgDinvApplyHost(
    const GmgFab<T>& v,
    GmgFab<T>& out,
    const GmgFab<T>& ux,
    const GmgFab<T>& lx,
    const GmgFab<T>& uy,
    const GmgFab<T>& ly,
    const GmgFab<T>& uz,
    const GmgFab<T>& lz,
    const GmgFab<T>& alpha
)
{
    for (amrex::MFIter mfi(out); mfi.isValid(); ++mfi)
    {
        const amrex::Box& vbx = mfi.validbox();
        const auto psi = v.const_array(mfi);
        const auto o = out.array(mfi);
        const auto ax = ux.const_array(mfi);
        const auto lxa = lx.const_array(mfi);
        const auto ay = uy.const_array(mfi);
        const auto lya = ly.const_array(mfi);
        const auto az = uz.const_array(mfi);
        const auto lza = lz.const_array(mfi);
        const auto al = alpha.const_array(mfi);
        const auto lo = amrex::lbound(vbx);
        const auto hi = amrex::ubound(vbx);
        for (int k = lo.z; k <= hi.z; ++k)
        {
            for (int j = lo.y; j <= hi.y; ++j)
            {
                for (int i = lo.x; i <= hi.x; ++i)
                {
                    const T aE = ax(i + 1, j, k);
                    const T aW = lxa(i, j, k);
                    const T aN = ay(i, j + 1, k);
                    const T aS = lya(i, j, k);
                    const T aT = az(i, j, k + 1);
                    const T aB = lza(i, j, k);
                    const T off = aE * psi(i + 1, j, k) + aW * psi(i - 1, j, k)
                                + aN * psi(i, j + 1, k) + aS * psi(i, j - 1, k)
                                + aT * psi(i, j, k + 1) + aB * psi(i, j, k - 1);
                    const T diag = al(i, j, k) - (aE + aW + aN + aS + aT + aB);
                    o(i, j, k) = (diag * psi(i, j, k) + off) / diag;
                }
            }
        }
    }
}

// Checkerboard seed (+-1 by cell parity) for the power iteration — close to the
// top eigenvector of the 7-point operator, so few iterations suffice.
template<class T>
void gmgFillCheckerDevice(GmgFab<T>& v)
{
    for (amrex::MFIter mfi(v); mfi.isValid(); ++mfi)
    {
        const amrex::Box& vbx = mfi.validbox();
        const auto a = v.array(mfi);
        amrex::ParallelFor(
            vbx,
            [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
            { a(i, j, k) = (((i + j + k) & 1) == 0) ? T(1) : T(-1); }
        );
    }
}

template<class T>
void gmgFillCheckerHost(GmgFab<T>& v)
{
    for (amrex::MFIter mfi(v); mfi.isValid(); ++mfi)
    {
        const amrex::Box& vbx = mfi.validbox();
        const auto a = v.array(mfi);
        const auto lo = amrex::lbound(vbx);
        const auto hi = amrex::ubound(vbx);
        for (int k = lo.z; k <= hi.z; ++k)
        {
            for (int j = lo.y; j <= hi.y; ++j)
            {
                for (int i = lo.x; i <= hi.x; ++i)
                {
                    a(i, j, k) = (((i + j + k) & 1) == 0) ? T(1) : T(-1);
                }
            }
        }
    }
}

} // namespace blockamr::solvers
