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
//
// T is what a level is STORED in, which for the bf16 hierarchy is not what the
// arithmetic happens in: solvers::GmgComputeT<T> (bf16.hpp) is the compute type,
// = T for double and float and float for Bf16. Kernels that mix a stored value
// with a literal weight spell the weight GmgComputeT<T> so a bf16 level does not
// round it to 3 digits; kernels reached only by the FP64/FP32 paths are written
// in T throughout, which is the same thing for those two.
// ---------------------------------------------------------------------------

template<class T>
using GmgFab = amrex::FabArray<amrex::BaseFab<T>>;

// Bundle of the 7 coefficient fields every stencil kernel takes positionally
// today (report: "alpha last" in kernels, "alpha first" in constructors — a
// silent uy/ly-swap-shaped hazard). A named struct removes the ordering
// question outright; alpha is listed first to match every constructor
// (gmgBottom.hpp:73, gmgPrecond.hpp:109) rather than the kernels' own
// historical order.
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

// The 6 face-coefficient VALUES at one cell, loaded from Array4s (not
// FaceCoeffs<T> directly -- kernels already hold per-MFIter Array4 views,
// not the FabArrays themselves, inside the ParallelFor/loop body).
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

// Same summation ORDER as every existing site (aE+aW+aN+aS+aT+aB) -- must
// stay bit-for-bit, not just algebraically equivalent.
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
//
// Cross-TU (Class B, see T9 report): reached from persistent.cpp (via
// gmgPrecond.hpp) AND gmgKokkos/vcycle.hpp, instantiated both by apply.cpp's
// production precond="gmg_kokkos" path and by bench/gmgVcycleBench.cpp's
// harness for every backend (both object libraries land in the same
// _blockamr.so) — an AMREX_GPU_HOST_DEVICE lambda here would be an extended
// lambda instantiated in two CUDA TUs of one binary, the exact nvcc trap T2
// already hit. So this stays declaration-only in the header; the single
// definition + explicit instantiation lives in gmgKernels.cpp.
template<class T, class SRC>
void gmgConvertCopy(GmgFab<T>& dst, const SRC& src, bool onDevice);

// dst += src, per cell over dst's valid box, converting through dst's value_type.
// dst is any FabArray (the caller's FP64 MultiFab); src is a T-valued level fab.
// The native stationary solver adds the (possibly FP32) V-cycle correction back
// onto the FP64 solution; for both double it is a plain in-place add.
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
// (norm="l2" | "linf", the latter MLMG's — see stopNormInf.hpp) without a
// second pass over the residual. The extra ReduceOpMax is register-only work in
// an already bandwidth-bound kernel. Device + host twins.
struct ResidNorms
{
    double sumsq;  // sum r_i^2 — caller takes the sqrt for ||r||_2
    double maxabs; // max |r_i| — ||r||_inf
};

// Combine the per-rank norms into the global ones.
//
// amrex::ParReduce and amrex::ReduceSum are LOCAL: they reduce over the boxes a
// rank owns and do no MPI at all. AMReX's own MultiFab::norm2/Dot call
// ParallelAllReduce on the result afterwards (AMReX_FabArrayUtility.H), and that
// is what has to happen here too, on the SAME communicator — gmgSolve compares
// this norm against a ||rhs|| taken with MultiFab::norm2/norminf, so a rank-local
// residual against a globally reduced baseline understates the residual and stops
// the solve early and silently. Measured at 2 ranks before this reduction: the
// reported ||r||_2 was 0.711x the true one (~1/sqrt(2)) and ||r||_inf 0.871x.
//
// Two collectives rather than one: the 2-norm needs a Sum and the inf-norm a Max.
// They cost one latency each per V-cycle, not per kernel.
inline void reduceResidNorms(ResidNorms& r)
{
    const MPI_Comm comm = amrex::ParallelContext::CommunicatorSub();
    amrex::ParallelAllReduce::Sum(r.sumsq, comm);
    amrex::ParallelAllReduce::Max(r.maxabs, comm);
}

// Two internal building blocks, not one HostDeviceParallelFor: (1) the
// residual write into `out` is a plain stencil kernel, collapsed the same way
// as every other twin -- HostDeviceParallelFor genuinely honors
// Gpu::LaunchSafeGuard here. (2) the norm is a SEPARATE reduction over the
// just-written `out`, kept as an EXPLICIT if (onDevice) branch rather than a
// LaunchSafeGuard-wrapped amrex::ParReduce: unlike HostDeviceParallelFor (and
// unlike amrex::ReduceSum, gmgNorm2's own pattern), amrex::ParReduce's
// ReduceOps::eval() for a FabArray is selected at COMPILE time by
// #ifdef AMREX_USE_GPU with no runtime Gpu::inLaunchRegion() check anywhere
// in its call chain -- AMReX's own LaunchSafeGuard doc says as much ("This
// will only switch from GPU to CPU for kernels launched with macros...
// should not be used for comparing GPU to non-GPU... behavior"). So the
// device branch below is the old device twin's amrex::ParReduce verbatim,
// and the host branch is the old host twin's sequential accumulation loop
// verbatim (same MFIter/k/j/i iteration order, same res.sumsq/res.maxabs
// accumulation order) -- a real host-vs-device split, not a cosmetic one.
// Reducing the STORED value of `out` (not the local double `r`) is preserved
// for both paths, so reference and cuda give an identical norm: exact FP64
// for T=double, fp32-rounded for T=float -- see the two-kernel-fusion
// rationale above.
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
                        // Reduce the STORED value (like the device branch's
                        // ParReduce over `out`) so reference and cuda give an
                        // identical norm: exact FP64 for T=double, fp32-rounded
                        // for T=float.
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

// ||mf||_2 over the valid region (0 ghost), accumulated in the fab's value_type
// and summed across ranks (used only by the setup power iteration).
//
// The cross-rank sum is not cosmetic: the power iteration divides by this norm to
// renormalise its iterate, so a per-rank norm would give each rank a DIFFERENT
// lambda_max and hence different Chebyshev coefficients — the smoother would stop
// being one global linear operator. Every level shares level 0's
// DistributionMapping, so all ranks reach this collective the same number of times.
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

// One red-black successive-over-relaxation colour pass: cells with (i+j+k)
// parity `parity` are updated in place towards the exact Gauss-Seidel value
// gs = (rhs - off) / D, with D = alpha - sum(face coeffs) recomputed on the fly
// (tiny |D| guarded to no update). The update is
//   sol <- sol + omega * (gs - sol),
// so omega = 1 is plain Gauss-Seidel (the previous behaviour, bit-for-bit) and
// omega > 1 over-relaxes — MLMG's abec_gsrb uses omega = 1.15. The 7-point
// stencil only couples opposite colours, so the in-place update is race-free.
// sol's ghosts must be refreshed before EACH colour pass.
//
// Cross-TU (Class B, see T9 report): reached from persistent.cpp (via
// gmgPrecond.hpp) AND bench/gmgVcycleBench.cpp's "amrex" baseline column
// (both object libraries land in the same _blockamr.so) — an
// AMREX_GPU_HOST_DEVICE lambda here would be an extended lambda instantiated
// in two CUDA TUs of one binary, the exact nvcc trap T2 already hit. So this
// stays declaration-only in the header; the single definition + explicit
// instantiation lives in gmgKernels.cpp.
template<class T>
void gmgGsColor(
    GmgFab<T>& sol,
    const GmgFab<T>& rhs,
    const FaceCoeffs<T>& fc,
    int parity,
    double omega,
    bool onDevice
);

// Volume-average (factor-2) restriction of a cell field: coarse = mean of the
// 8 fine children. Also used to coarsen alpha (a per-volume density). Iterates
// the coarse MF; the fine MF shares the DistributionMapping, so the same MFIter
// index addresses the matching fine box (its BoxArray is refine(coarse, 2)).
//
// Cross-TU (Class B, see T9 report): reached from persistent.cpp (via
// gmgPrecond.hpp) AND gmgKokkos/vcycle.hpp, instantiated both by apply.cpp's
// production precond="gmg_kokkos" path and by bench/gmgVcycleBench.cpp's
// harness for every backend (both object libraries land in the same
// _blockamr.so) — an AMREX_GPU_HOST_DEVICE lambda here would be an extended
// lambda instantiated in two CUDA TUs of one binary, the exact nvcc trap T2
// already hit. So this stays declaration-only in the header; the single
// definition + explicit instantiation lives in gmgKernels.cpp.
template<class T>
void gmgRestrict(const GmgFab<T>& fine, GmgFab<T>& crse, bool onDevice);

// Coarsen a face-coefficient field in direction `dir`: coarse face i_c covers
// fine face 2*i_c with the 2x2 transverse fine faces; a ~ -beta/dx^2, so the
// coarse coefficient is the arithmetic average of those 4 fine coefficients
// (beta averaged) divided by `scale` (dx doubled -> 4 for rediscretisation).
//
// Cross-TU (Class B, see T9 report): reached from persistent.cpp (via
// gmgPrecond.hpp) AND gmgKokkos/vcycle.hpp, instantiated both by apply.cpp's
// production precond="gmg_kokkos" path and by bench/gmgVcycleBench.cpp's
// harness for every backend (both object libraries land in the same
// _blockamr.so) — an AMREX_GPU_HOST_DEVICE lambda here would be an extended
// lambda instantiated in two CUDA TUs of one binary, the exact nvcc trap T2
// already hit. So this stays declaration-only in the header; the single
// definition + explicit instantiation lives in gmgKernels.cpp.
template<class T>
void gmgCoarsenFace(const GmgFab<T>& fine, GmgFab<T>& crse, int dir, double scale, bool onDevice);

// Piecewise-constant prolongation + correction: fine cell += coarse parent
// value (the adjoint of the volume-average restriction, up to the 1/8 factor).
//
// Cross-TU (Class B, see T9 report): reached from persistent.cpp (via
// gmgPrecond.hpp) AND bench/gmgVcycleBench.cpp's "amrex" baseline column
// (both object libraries land in the same _blockamr.so) — an
// AMREX_GPU_HOST_DEVICE lambda here would be an extended lambda instantiated
// in two CUDA TUs of one binary, the exact nvcc trap T2 already hit. So this
// stays declaration-only in the header; the single definition + explicit
// instantiation lives in gmgKernels.cpp.
template<class T>
void gmgProlongAdd(const GmgFab<T>& crse, GmgFab<T>& fine, bool onDevice);

// Fused residual + volume-average restriction: coarse rhs cell = mean of the 8
// fine residuals r = rhs - A sol, each computed on the fly. Iterates the coarse
// box (fine sol's ghosts must be filled). Saves the full fine-grid resid
// read+write of the separate residual + restriction passes (M4 item 3).
//
// Cross-TU (Class B, see T9 report): reached from persistent.cpp (via
// gmgPrecond.hpp) AND bench/gmgVcycleBench.cpp's "amrex" baseline column
// (both object libraries land in the same _blockamr.so) — an
// AMREX_GPU_HOST_DEVICE lambda here would be an extended lambda instantiated
// in two CUDA TUs of one binary, the exact nvcc trap T2 already hit. So this
// stays declaration-only in the header; the single definition + explicit
// instantiation lives in gmgKernels.cpp.
template<class T>
void gmgResidRestrict(
    const GmgFab<T>& sol,
    const GmgFab<T>& rhs,
    GmgFab<T>& crhs,
    const FaceCoeffs<T>& fc,
    bool onDevice
);

// One fused Jacobi-Chebyshev degree step: computes r = rhs - A sol on the fly
// (sol's ghosts must be filled) and the polynomial increment
// d = cb * D^{-1} r + (readOld ? ca * d : 0), D = alpha - sum(face coeffs). sol
// is NOT written here (its neighbours are read for r) — the caller adds d to sol
// afterwards, so the whole step is Jacobi-like (race-free) and, being a fixed
// polynomial in the symmetric operator, a symmetric linear smoother (CG-safe).
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

// out = A v (v's ghosts filled). The same seven-point stencil as
// gmgDinvApply without the diagonal scaling -- the operator itself, which
// is what a Krylov bottom solve needs. The diagonal is recomputed from the face
// coefficients rather than stored, exactly as every other kernel here does, so
// an asymmetric operator (ux(i+1) != lx(i+1)) is handled with no special case:
// each direction reads its own upper and lower array.
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

} // namespace blockamr::solvers
