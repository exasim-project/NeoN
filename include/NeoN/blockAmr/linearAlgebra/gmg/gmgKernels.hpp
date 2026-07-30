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
#include "NeoN/blockAmr/linearAlgebra/stencil.hpp"

namespace blockamr::la
{

// Kernels of the native GMG V-cycle (GmgPrecond) on the face-coefficient operator: device
// free functions (nvcc rejects extended __device__ lambdas in class members), each with a
// host-loop twin. T is a level's STORAGE type; arithmetic in la::GmgComputeT<T> (bf16.hpp).

template<class T>
using GmgFab = amrex::FabArray<amrex::BaseFab<T>>;

// The 7 coefficient fields as one named struct, so a positional uy/ly swap cannot happen.
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

// FaceCoeffVals / loadFaceCoeffs / stencilDiag / stencilOffDiag: linearAlgebra/stencil.hpp.

// |diagonal| floor guarding the RB-GS division; per type because 1e-300 is not a float.
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

// Copy src into the T-valued dst over dst's valid box, converting per cell; exact for double.
// Instantiated in gmgKernels.cpp; a miss is a null device fnptr at runtime, not a link error.
template<class T, class SRC>
void gmgConvertCopy(GmgFab<T>& dst, const SRC& src, bool onDevice);

// dst += src over dst's valid box, converting through dst's value_type.
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

// Fused residual + convert-scatter (r = rhs - A*sol - shift in DOUBLE, stored cast to T into
// the L0 rhs) plus a second light kernel giving BOTH norms. Why not one kernel, and the
// norm's precision: report/blockamr-gmg-notes.md#fusion (NeoFOAM repo).
struct ResidNorms
{
    double sumsq;  // sum r_i^2 — caller takes the sqrt for ||r||_2
    double maxabs; // max |r_i| — ||r||_inf
};

// Combine the per-rank norms: ParReduce is rank-LOCAL and must use the SAME communicator
// as the ||rhs|| gmgSolve compares against — report/blockamr-gmg-notes.md#rank-reduction.
inline void reduceResidNorms(ResidNorms& r)
{
    const MPI_Comm comm = amrex::ParallelContext::CommunicatorSub();
    amrex::ParallelAllReduce::Sum(r.sumsq, comm);
    amrex::ParallelAllReduce::Max(r.maxabs, comm);
}

// The norm is an explicit if (onDevice), not LaunchSafeGuard: ParReduce picks its backend at
// compile time, so the host branch is a sequential loop in the same accumulation order.
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
                        // Reduce the STORED value, as the device ParReduce does.
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

// ||mf||_2 over the valid region, accumulated in the fab's value_type and summed across ranks
// (setup power iteration only). Why the cross-rank sum is required:
// report/blockamr-gmg-notes.md#rank-reduction.
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

// One red-black SOR colour pass over (i+j+k) parity `parity`: sol += omega*(gs - sol), gs =
// (rhs - off)/D on the fly; omega=1 is plain Gauss-Seidel; ghosts refresh before EACH pass.
// Instantiated in gmgKernels.cpp; a miss is a null device fnptr at runtime, not a link error.
template<class T>
void gmgGsColor(
    GmgFab<T>& sol,
    const GmgFab<T>& rhs,
    const FaceCoeffs<T>& fc,
    int parity,
    double omega,
    bool onDevice
);

// Factor-2 restriction of a cell field: a PLAIN 8-child average with no dx factor, valid
// ONLY for a dx-INDEPENDENT density (alpha). Confusing it with gmgCoarsenFace's
// w = 0.25/scale caused a real bug.
// Instantiated in gmgKernels.cpp; a miss is a null device fnptr at runtime, not a link error.
template<class T>
void gmgRestrict(const GmgFab<T>& fine, GmgFab<T>& crse, bool onDevice);

// Coarsen a face coefficient in `dir`: the 4 covered fine faces averaged, divided by `scale`,
// i.e. w = 0.25/scale. a = -beta/dx^2 is NEGATIVE and dx-DEPENDENT, so a doubled dx means
// scale = 4 — the factor gmgRestrict must NOT have.
// Instantiated in gmgKernels.cpp; a miss is a null device fnptr at runtime, not a link error.
template<class T>
void gmgCoarsenFace(const GmgFab<T>& fine, GmgFab<T>& crse, int dir, double scale, bool onDevice);

// Piecewise-constant prolongation + correction: fine cell += coarse parent, the adjoint of the
// volume-average restriction up to 1/8, which is what makes the V-cycle CG-safe.
// Instantiated in gmgKernels.cpp; a miss is a null device fnptr at runtime, not a link error.
template<class T>
void gmgProlongAdd(const GmgFab<T>& crse, GmgFab<T>& fine, bool onDevice);

// Fused residual + volume-average restriction: coarse rhs = mean of the 8 fine residuals on
// the fly. Iterates the coarse box; fine sol's ghosts must be filled.
// Instantiated in gmgKernels.cpp; a miss is a null device fnptr at runtime, not a link error.
template<class T>
void gmgResidRestrict(
    const GmgFab<T>& sol,
    const GmgFab<T>& rhs,
    GmgFab<T>& crhs,
    const FaceCoeffs<T>& fc,
    bool onDevice
);

// One fused Jacobi-Chebyshev degree step: d = cb * D^{-1} r + (readOld ? ca*d : 0), r on the
// fly (sol's ghosts filled). sol is NOT written here, so the step is race-free and CG-safe.
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

// out = A v (v's ghosts filled) — what a Krylov bottom solve needs. The diagonal is recomputed
// from the face coefficients, so an asymmetric operator needs no special case.
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

// out = D^{-1} A v (v's ghosts filled), for the setup power iteration estimating lambda_max.
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

// Checkerboard seed (+-1 by cell parity) for the power iteration: near the top eigenvector.
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
