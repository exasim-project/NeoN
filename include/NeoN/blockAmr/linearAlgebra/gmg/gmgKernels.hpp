// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <AMReX_GpuLaunch.H>
#include <AMReX_Math.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParallelContext.H>
#include <AMReX_ParallelReduce.H>
#include <AMReX_Reduce.H>

#include <algorithm>
#include <cmath>

#include "NeoN/blockAmr/core/profiling.hpp"
#include "NeoN/blockAmr/linearAlgebra/gmg/bf16.hpp"

namespace blockamr::la
{

// Kernels of the native GMG V-cycle preconditioner (GmgPrecond): AMReX
// primitives on the face-coefficient operator only, no MLLinOp/MLMG. Device
// kernels are namespace-scope free functions because nvcc rejects extended
// __device__ lambdas in private/protected members; each has a host-loop twin for
// the ReferenceExecutor path.
//
// T is what a level is STORED in (double, float, Bf16); the whole V-cycle runs in
// T while the outer CG/operator stays FP64. Arithmetic happens in
// la::GmgComputeT<T> (bf16.hpp) — = T except float for Bf16 — so a kernel mixing a
// stored value with a literal weight spells the weight GmgComputeT<T> and a bf16
// level does not round it to 3 digits.
//
// nvcc cross-TU rule: a kernel reached from more than one CUDA TU of the same
// binary must be declaration-only here and defined + explicitly instantiated in
// gmgKernels.cpp. A missed instantiation is a null device function pointer at
// RUNTIME, not a link error.

template<class T>
using GmgFab = amrex::FabArray<amrex::BaseFab<T>>;

// The 7 coefficient fields as one named struct, so a positional uy/ly-style swap
// cannot happen silently; alpha first to match every constructor rather than the
// kernels' historical "alpha last" order.
template<class T>
struct FaceCoeffs
{
    const GmgFab<T>* alpha;
    const GmgFab<T>* ux;
    const GmgFab<T>* lx;
    const GmgFab<T>* uy;
    const GmgFab<T>* ly;
    const GmgFab<T>* uz;
    const GmgFab<T>* lz;
};

// The 6 face-coefficient VALUES at one cell: inside the loop body a kernel holds
// per-MFIter Array4 views, not the FabArrays of FaceCoeffs<T>.
template<class T>
struct FaceCoeffVals
{
    T aE, aW, aN, aS, aT, aB;
};

template<class T>
AMREX_GPU_HOST_DEVICE FaceCoeffVals<T> loadFaceCoeffs(
    const amrex::Array4<const T>& ux,
    const amrex::Array4<const T>& lx,
    const amrex::Array4<const T>& uy,
    const amrex::Array4<const T>& ly,
    const amrex::Array4<const T>& uz,
    const amrex::Array4<const T>& lz,
    int i,
    int j,
    int k
) noexcept
{
    return {
        ux(i + 1, j, k), lx(i, j, k), uy(i, j + 1, k), ly(i, j, k), uz(i, j, k + 1), lz(i, j, k)
    };
}

// Summation order aE+aW+aN+aS+aT+aB is that of every existing site and must stay
// bit-for-bit identical, not merely algebraically equivalent.
template<class T>
AMREX_GPU_HOST_DEVICE T stencilDiag(T alpha, const FaceCoeffVals<T>& c) noexcept
{
    return alpha - (c.aE + c.aW + c.aN + c.aS + c.aT + c.aB);
}

template<class T>
AMREX_GPU_HOST_DEVICE T
stencilOffDiag(const FaceCoeffVals<T>& c, T pE, T pW, T pN, T pS, T pT, T pB) noexcept
{
    return c.aE * pE + c.aW * pW + c.aN * pN + c.aS * pS + c.aT * pT + c.aB * pB;
}

// |diagonal| floor guarding the RB-GS in-place division (skip rather than divide
// by ~0). Per value type because 1e-300 is not a valid float literal.
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

// Copy src (any FabArray) into the T-valued dst over dst's valid box, converting
// per cell — MultiFab::Copy cannot cross value types. Exact for T=double, so the
// FP64 path is numerically unchanged.
//
// Cross-TU: declaration-only here, defined + explicitly instantiated in
// gmgKernels.cpp; a missed instantiation is a runtime null device function pointer.
template<class T, class SRC>
void gmgConvertCopy(GmgFab<T>& dst, const SRC& src, bool onDevice);

// dst += src over dst's valid box, converting through dst's value_type: adds the
// (possibly FP32) V-cycle correction back onto the caller's FP64 solution.
template<class DST, class T>
void gmgConvertAdd(DST& dst, const GmgFab<T>& src, bool onDevice)
{
    using DT = typename DST::value_type;
    amrex::Gpu::LaunchSafeGuard lsg(onDevice);
    for (amrex::MFIter mfi(dst); mfi.isValid(); ++mfi)
    {
        const amrex::Box& vbx = mfi.validbox();
        const auto d = dst.array(mfi);
        const auto s = src.const_array(mfi);
        amrex::HostDeviceParallelFor(
            vbx,
            [=] AMREX_GPU_HOST_DEVICE(int i, int j, int k) noexcept
            { d(i, j, k) += static_cast<DT>(s(i, j, k)); }
        );
    }
}

// Fused residual + convert-scatter + norm: r = rhs - A*sol - shift computed in
// DOUBLE (shift = the nullspace-projection constant, 0 when not projecting) and
// stored cast to T straight into the L0 rhs `out`; the norm is a SECOND, light
// kernel reducing `out`.
//
// Not one kernel: folding the reduction into the heavy stencil kernel measured
// ~1.0 ms/iter at 256^3 (it spills the register-bound kernel) against the
// 0.34+0.54 ms it saves; the separate reduction over the just-written `out` costs
// ~0.20 ms/iter and reuses the cached data.
//
// Reducing the STORED `out` gives a bit-exact FP64 norm for T=double; for T=float
// it carries ~6e-8 relative rounding, far below the ~10x per-cycle residual drop,
// so the stopping cycle is unchanged (verified: identical iters and answer).
//
// One reduction yields BOTH norms (sum r^2 and max|r|), so the solver can stop in
// norm="l2" or "linf" (stopNormInf.hpp) without a second pass over the residual;
// the extra ReduceOpMax is register-only work here.
struct ResidNorms
{
    double sumsq;  // sum r_i^2 — caller takes the sqrt for ||r||_2
    double maxabs; // max |r_i| — ||r||_inf
};

// Combine the per-rank norms into the global ones. amrex::ParReduce/ReduceSum are
// rank-LOCAL, so this must happen here and on the SAME communicator as the
// ||rhs|| gmgSolve compares against (MultiFab::norm2/norminf, which do reduce);
// otherwise the residual is understated and the solve stops early and silently —
// measured at 2 ranks: ||r||_2 0.711x and ||r||_inf 0.871x of the true value.
// Two collectives because l2 needs a Sum and linf a Max: one latency each per
// V-cycle, not per kernel.
inline void reduceResidNorms(ResidNorms& r)
{
    const MPI_Comm comm = amrex::ParallelContext::CommunicatorSub();
    amrex::ParallelAllReduce::Sum(r.sumsq, comm);
    amrex::ParallelAllReduce::Max(r.maxabs, comm);
}

// The residual write honours Gpu::LaunchSafeGuard like every other twin, but the
// norm is an EXPLICIT if (onDevice) branch: amrex::ParReduce picks its
// ReduceOps::eval() at COMPILE time via #ifdef AMREX_USE_GPU with no runtime
// inLaunchRegion() check, so LaunchSafeGuard cannot send it to the host (unlike
// HostDeviceParallelFor or amrex::ReduceSum, gmgNorm2's pattern). Hence a real
// split: device = ParReduce, host = a sequential loop in the same MFIter/k/j/i and
// accumulation order.
template<class T>
ResidNorms faceCoeffResidScatterNorm(
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
    GmgFab<T>& out,
    bool onDevice
)
{
    ResidNorms res {};
    {
        prof::Timer t("gmg.solve.residkern");
        amrex::Gpu::LaunchSafeGuard lsg(onDevice);
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
            amrex::HostDeviceParallelFor(
                vbx,
                [=] AMREX_GPU_HOST_DEVICE(int i, int j, int k) noexcept
                {
                    const auto c = loadFaceCoeffs<double>(ax, lxa, ay, lya, az, lza, i, j, k);
                    const double offd = stencilOffDiag(
                        c,
                        psi(i + 1, j, k),
                        psi(i - 1, j, k),
                        psi(i, j + 1, k),
                        psi(i, j - 1, k),
                        psi(i, j, k + 1),
                        psi(i, j, k - 1)
                    );
                    const double diag = stencilDiag(al(i, j, k), c);
                    const double r = bb(i, j, k) - (diag * psi(i, j, k) + offd) - shift;
                    o(i, j, k) = static_cast<T>(r);
                }
            );
        }
    }
    if (onDevice)
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
    else
    {
        for (amrex::MFIter mfi(out); mfi.isValid(); ++mfi)
        {
            const amrex::Box& vbx = mfi.validbox();
            const auto o = out.const_array(mfi);
            const auto lo = amrex::lbound(vbx);
            const auto hi = amrex::ubound(vbx);
            for (int k = lo.z; k <= hi.z; ++k)
            {
                for (int j = lo.y; j <= hi.y; ++j)
                {
                    for (int i = lo.x; i <= hi.x; ++i)
                    {
                        // Reduce the STORED value, as the device ParReduce does,
                        // so reference and cuda give an identical norm.
                        const double v = static_cast<double>(o(i, j, k));
                        res.sumsq += v * v;
                        res.maxabs = std::max(res.maxabs, std::abs(v));
                    }
                }
            }
        }
    }
    reduceResidNorms(res);
    return res;
}

// ||mf||_2 over the valid region (0 ghost), accumulated in the fab's value_type and
// summed across ranks (setup power iteration only). The cross-rank sum is required:
// the power iteration renormalises by this norm, so a per-rank one would give each
// rank a DIFFERENT lambda_max and different Chebyshev coefficients — the smoother
// would stop being one global linear operator. Every level shares level 0's
// DistributionMapping, so all ranks reach this collective equally often.
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
    double sumsq = static_cast<double>(sq);
    amrex::ParallelAllReduce::Sum(sumsq, amrex::ParallelContext::CommunicatorSub());
    return std::sqrt(sumsq);
}

// One red-black SOR colour pass over the cells of (i+j+k) parity `parity`:
// sol <- sol + omega * (gs - sol) with gs = (rhs - off) / D and D = alpha -
// sum(face coeffs) recomputed on the fly (tiny |D| left unchanged). omega = 1 is
// plain Gauss-Seidel, bit-for-bit the previous behaviour; MLMG's abec_gsrb uses
// 1.15. The 7-point stencil couples only opposite colours, so the in-place update
// is race-free — but sol's ghosts must be refreshed before EACH colour pass.
//
// Cross-TU: declaration-only here, defined + explicitly instantiated in
// gmgKernels.cpp; a missed instantiation is a runtime null device function pointer.
template<class T>
void gmgGsColor(
    GmgFab<T>& sol,
    const GmgFab<T>& rhs,
    const FaceCoeffs<T>& fc,
    int parity,
    double omega,
    bool onDevice
);

// Factor-2 restriction of a cell field: a PLAIN 8-child volume average, no dx
// factor at all, which is correct ONLY for a dx-INDEPENDENT density — alpha
// qualifies. Contrast gmgCoarsenFace below, whose face coefficient a = -beta/dx^2
// DOES depend on dx and therefore carries an extra 1/scale; getting these two
// laws mixed up is a live source of bugs. Iterates the coarse MF; the fine MF
// shares the DistributionMapping, so the same MFIter index addresses the matching
// fine box (its BoxArray is refine(coarse, 2)).
//
// Cross-TU: declaration-only here, defined + explicitly instantiated in
// gmgKernels.cpp; a missed instantiation is a runtime null device function pointer.
template<class T>
void gmgRestrict(const GmgFab<T>& fine, GmgFab<T>& crse, bool onDevice);

// Coarsen a face coefficient in direction `dir`: coarse face i_c averages the 4
// covered fine faces at 2*i_c and divides by `scale`, i.e. weight w = 0.25/scale.
// Because a = -beta/dx^2 (negative, and dx-DEPENDENT), rediscretisation with dx
// doubled means scale = 4 — the dx factor gmgRestrict above must NOT have.
//
// Cross-TU: declaration-only here, defined + explicitly instantiated in
// gmgKernels.cpp; a missed instantiation is a runtime null device function pointer.
template<class T>
void gmgCoarsenFace(const GmgFab<T>& fine, GmgFab<T>& crse, int dir, double scale, bool onDevice);

// Piecewise-constant prolongation + correction: fine cell += coarse parent value —
// the adjoint of the volume-average restriction up to the 1/8 factor, which is what
// makes the V-cycle CG-safe.
//
// Cross-TU: declaration-only here, defined + explicitly instantiated in
// gmgKernels.cpp; a missed instantiation is a runtime null device function pointer.
template<class T>
void gmgProlongAdd(const GmgFab<T>& crse, GmgFab<T>& fine, bool onDevice);

// Fused residual + volume-average restriction: coarse rhs = mean of the 8 fine
// residuals r = rhs - A sol computed on the fly, saving the separate passes' full
// fine-grid read+write. Iterates the coarse box; fine sol's ghosts must be filled.
//
// Cross-TU: declaration-only here, defined + explicitly instantiated in
// gmgKernels.cpp; a missed instantiation is a runtime null device function pointer.
template<class T>
void gmgResidRestrict(
    const GmgFab<T>& sol,
    const GmgFab<T>& rhs,
    GmgFab<T>& crhs,
    const FaceCoeffs<T>& fc,
    bool onDevice
);

// One fused Jacobi-Chebyshev degree step: d = cb * D^{-1} r + (readOld ? ca*d : 0)
// with r = rhs - A sol on the fly (sol's ghosts must be filled). sol is NOT written
// here — its neighbours are read for r and the caller adds d afterwards, so the step
// is Jacobi-like (race-free) and, a fixed polynomial in a symmetric operator, a
// symmetric linear smoother (CG-safe).
template<class T>
void gmgChebComputeD(
    const GmgFab<T>& sol,
    const GmgFab<T>& rhs,
    const FaceCoeffs<T>& fc,
    GmgFab<T>& d,
    T ca,
    T cb,
    bool readOld,
    bool onDevice
)
{
    amrex::Gpu::LaunchSafeGuard lsg(onDevice);
    for (amrex::MFIter mfi(rhs); mfi.isValid(); ++mfi)
    {
        const amrex::Box& vbx = mfi.validbox();
        const auto psi = sol.const_array(mfi);
        const auto b = rhs.const_array(mfi);
        const auto dd = d.array(mfi);
        const auto ax = fc.ux->const_array(mfi);
        const auto lxa = fc.lx->const_array(mfi);
        const auto ay = fc.uy->const_array(mfi);
        const auto lya = fc.ly->const_array(mfi);
        const auto az = fc.uz->const_array(mfi);
        const auto lza = fc.lz->const_array(mfi);
        const auto al = fc.alpha->const_array(mfi);
        amrex::HostDeviceParallelFor(
            vbx,
            [=] AMREX_GPU_HOST_DEVICE(int i, int j, int k) noexcept
            {
                const auto c = loadFaceCoeffs<T>(ax, lxa, ay, lya, az, lza, i, j, k);
                const T off = stencilOffDiag(
                    c,
                    psi(i + 1, j, k),
                    psi(i - 1, j, k),
                    psi(i, j + 1, k),
                    psi(i, j - 1, k),
                    psi(i, j, k + 1),
                    psi(i, j, k - 1)
                );
                const T diag = stencilDiag(al(i, j, k), c);
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

// out = A v (v's ghosts filled) — the operator itself, which is what a Krylov
// bottom solve needs. The diagonal is recomputed from the face coefficients rather
// than stored, so an asymmetric operator (ux(i+1) != lx(i+1)) needs no special
// case: each direction reads its own upper and lower array.
template<class T>
void gmgApply(const GmgFab<T>& v, GmgFab<T>& out, const FaceCoeffs<T>& fc, bool onDevice)
{
    amrex::Gpu::LaunchSafeGuard lsg(onDevice);
    for (amrex::MFIter mfi(out); mfi.isValid(); ++mfi)
    {
        const amrex::Box& vbx = mfi.validbox();
        const auto psi = v.const_array(mfi);
        const auto o = out.array(mfi);
        const auto ax = fc.ux->const_array(mfi);
        const auto lxa = fc.lx->const_array(mfi);
        const auto ay = fc.uy->const_array(mfi);
        const auto lya = fc.ly->const_array(mfi);
        const auto az = fc.uz->const_array(mfi);
        const auto lza = fc.lz->const_array(mfi);
        const auto al = fc.alpha->const_array(mfi);
        amrex::HostDeviceParallelFor(
            vbx,
            [=] AMREX_GPU_HOST_DEVICE(int i, int j, int k) noexcept
            {
                const auto c = loadFaceCoeffs<T>(ax, lxa, ay, lya, az, lza, i, j, k);
                const T off = stencilOffDiag(
                    c,
                    psi(i + 1, j, k),
                    psi(i - 1, j, k),
                    psi(i, j + 1, k),
                    psi(i, j - 1, k),
                    psi(i, j, k + 1),
                    psi(i, j, k - 1)
                );
                const T diag = stencilDiag(al(i, j, k), c);
                o(i, j, k) = diag * psi(i, j, k) + off;
            }
        );
    }
}

// out = D^{-1} A v (v's ghosts filled), used by the setup power iteration that
// estimates lambda_max of D^{-1}A per level for the Chebyshev interval.
template<class T>
void gmgDinvApply(const GmgFab<T>& v, GmgFab<T>& out, const FaceCoeffs<T>& fc, bool onDevice)
{
    amrex::Gpu::LaunchSafeGuard lsg(onDevice);
    for (amrex::MFIter mfi(out); mfi.isValid(); ++mfi)
    {
        const amrex::Box& vbx = mfi.validbox();
        const auto psi = v.const_array(mfi);
        const auto o = out.array(mfi);
        const auto ax = fc.ux->const_array(mfi);
        const auto lxa = fc.lx->const_array(mfi);
        const auto ay = fc.uy->const_array(mfi);
        const auto lya = fc.ly->const_array(mfi);
        const auto az = fc.uz->const_array(mfi);
        const auto lza = fc.lz->const_array(mfi);
        const auto al = fc.alpha->const_array(mfi);
        amrex::HostDeviceParallelFor(
            vbx,
            [=] AMREX_GPU_HOST_DEVICE(int i, int j, int k) noexcept
            {
                const auto c = loadFaceCoeffs<T>(ax, lxa, ay, lya, az, lza, i, j, k);
                const T off = stencilOffDiag(
                    c,
                    psi(i + 1, j, k),
                    psi(i - 1, j, k),
                    psi(i, j + 1, k),
                    psi(i, j - 1, k),
                    psi(i, j, k + 1),
                    psi(i, j, k - 1)
                );
                const T diag = stencilDiag(al(i, j, k), c);
                o(i, j, k) = (diag * psi(i, j, k) + off) / diag;
            }
        );
    }
}

// Checkerboard seed (+-1 by cell parity) for the power iteration — close to the
// top eigenvector of the 7-point operator, so few iterations suffice.
template<class T>
void gmgFillChecker(GmgFab<T>& v, bool onDevice)
{
    amrex::Gpu::LaunchSafeGuard lsg(onDevice);
    for (amrex::MFIter mfi(v); mfi.isValid(); ++mfi)
    {
        const amrex::Box& vbx = mfi.validbox();
        const auto a = v.array(mfi);
        amrex::HostDeviceParallelFor(
            vbx,
            [=] AMREX_GPU_HOST_DEVICE(int i, int j, int k) noexcept
            { a(i, j, k) = (((i + j + k) & 1) == 0) ? T(1) : T(-1); }
        );
    }
}

} // namespace blockamr::la
