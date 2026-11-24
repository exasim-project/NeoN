// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

// The gmgKernels.hpp kernels reached from more than one CUDA TU: declaration-only in the header,
// defined here, and a missed instantiation is a null device function pointer at runtime, not a
// link error. Why: report/blockamr-linear-algebra-notes.md#the-nvcc-multi-tu-trap

#include "NeoN/blockAmr/linearAlgebra/gmg/gmgKernels.hpp"

namespace blockamr::la
{

// Copy any FabArray into the T-valued dst, converting per cell over dst's valid box: MultiFab::Copy
// needs matching value types. For T=double it is exact, so the FP64 path is unchanged.
template<class T, class SRC>
void gmgConvertCopy(GmgFab<T>& dst, const SRC& src, bool onDevice)
{
    amrex::Gpu::LaunchSafeGuard lsg(onDevice);
    for (amrex::MFIter mfi(dst); mfi.isValid(); ++mfi)
    {
        const amrex::Box& vbx = mfi.validbox();
        const auto d = dst.array(mfi);
        const auto s = src.const_array(mfi);
        amrex::HostDeviceParallelFor(
            vbx,
            [=] AMREX_GPU_HOST_DEVICE(int i, int j, int k) noexcept
            { d(i, j, k) = static_cast<T>(s(i, j, k)); }
        );
    }
}

template void
gmgConvertCopy<double, amrex::MultiFab>(GmgFab<double>&, const amrex::MultiFab&, bool);
template void gmgConvertCopy<float, amrex::MultiFab>(GmgFab<float>&, const amrex::MultiFab&, bool);
template void gmgConvertCopy<Bf16, amrex::MultiFab>(GmgFab<Bf16>&, const amrex::MultiFab&, bool);
template void gmgConvertCopy<double, GmgFab<double>>(GmgFab<double>&, const GmgFab<double>&, bool);
template void gmgConvertCopy<float, GmgFab<float>>(GmgFab<float>&, const GmgFab<float>&, bool);

template<class T>
void gmgGsColor(
    GmgFab<T>& sol, const GmgFab<T>& rhs, const FaceCoeffs<T>& fc, GsSweep sweep, bool onDevice
)
{
    amrex::Gpu::LaunchSafeGuard lsg(onDevice);
    const int parity = sweep.parity;
    const T om = static_cast<T>(sweep.omega);
    for (amrex::MFIter mfi(rhs); mfi.isValid(); ++mfi)
    {
        const amrex::Box& vbx = mfi.validbox();
        const auto psi = sol.array(mfi);
        const auto b = rhs.const_array(mfi);
        const auto al = fc.alpha->const_array(mfi);
        const FaceCoeffArrays fca {
            fc.ux->const_array(mfi),
            fc.lx->const_array(mfi),
            fc.uy->const_array(mfi),
            fc.ly->const_array(mfi),
            fc.uz->const_array(mfi),
            fc.lz->const_array(mfi)
        };
        amrex::HostDeviceParallelFor(
            vbx,
            [=] AMREX_GPU_HOST_DEVICE(int i, int j, int k) noexcept
            {
                if (((i + j + k) & 1) != parity)
                {
                    return;
                }
                const auto c = loadFaceCoeffs<T>(fca, i, j, k);
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
                if (amrex::Math::abs(diag) > gmgDiagFloor<T>())
                {
                    const T gs = (b(i, j, k) - off) / diag;
                    psi(i, j, k) += om * (gs - psi(i, j, k));
                }
            }
        );
    }
}

template void gmgGsColor<double>(
    GmgFab<double>&, const GmgFab<double>&, const FaceCoeffs<double>&, GsSweep, bool
);
template void
gmgGsColor<float>(GmgFab<float>&, const GmgFab<float>&, const FaceCoeffs<float>&, GsSweep, bool);

// Plain volume average: coarse = mean of the 8 fine children. Valid ONLY for a dx-INDEPENDENT
// density such as alpha -- a dx-dependent coefficient needs gmgCoarsenFace's 1/scale instead.
// Iterates the coarse MF; the fine MF shares its DistributionMapping (BoxArray refine(coarse, 2)).
template<class T>
void gmgRestrict(const GmgFab<T>& fine, GmgFab<T>& crse, bool onDevice)
{
    amrex::Gpu::LaunchSafeGuard lsg(onDevice);
    for (amrex::MFIter mfi(crse); mfi.isValid(); ++mfi)
    {
        const amrex::Box& vbx = mfi.validbox();
        const auto f = fine.const_array(mfi);
        const auto c = crse.array(mfi);
        amrex::HostDeviceParallelFor(
            vbx,
            [=] AMREX_GPU_HOST_DEVICE(int i, int j, int k) noexcept
            {
                const int i2 = 2 * i, j2 = 2 * j, k2 = 2 * k;
                c(i, j, k) = static_cast<GmgComputeT<T>>(0.125)
                           * (f(i2, j2, k2) + f(i2 + 1, j2, k2) + f(i2, j2 + 1, k2)
                              + f(i2 + 1, j2 + 1, k2) + f(i2, j2, k2 + 1) + f(i2 + 1, j2, k2 + 1)
                              + f(i2, j2 + 1, k2 + 1) + f(i2 + 1, j2 + 1, k2 + 1));
            }
        );
    }
}

template void gmgRestrict<double>(const GmgFab<double>&, GmgFab<double>&, bool);
template void gmgRestrict<float>(const GmgFab<float>&, GmgFab<float>&, bool);
template void gmgRestrict<Bf16>(const GmgFab<Bf16>&, GmgFab<Bf16>&, bool);

// Coarsen a face coefficient in direction `dir`: coarse face i_c averages the 4 fine faces at
// 2*i_c, times 1/scale -- a = -beta/dx^2 (negative) is dx-DEPENDENT, so w = 0.25/scale, unlike
// gmgRestrict's plain average. Confusing the two laws caused a real bug.
template<class T>
void gmgCoarsenFace(const GmgFab<T>& fine, GmgFab<T>& crse, int dir, double scale, bool onDevice)
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
    const GmgComputeT<T> w = static_cast<GmgComputeT<T>>(0.25 / scale);
    amrex::Gpu::LaunchSafeGuard lsg(onDevice);
    for (amrex::MFIter mfi(crse); mfi.isValid(); ++mfi)
    {
        const amrex::Box& vbx = mfi.validbox();
        const auto f = fine.const_array(mfi);
        const auto c = crse.array(mfi);
        amrex::HostDeviceParallelFor(
            vbx,
            [=] AMREX_GPU_HOST_DEVICE(int i, int j, int k) noexcept
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

template void gmgCoarsenFace<double>(const GmgFab<double>&, GmgFab<double>&, int, double, bool);
template void gmgCoarsenFace<float>(const GmgFab<float>&, GmgFab<float>&, int, double, bool);
template void gmgCoarsenFace<Bf16>(const GmgFab<Bf16>&, GmgFab<Bf16>&, int, double, bool);

// Piecewise-constant prolongation + correction: fine cell += its coarse parent value (the adjoint
// of the volume-average restriction, up to the 1/8 factor).
template<class T>
void gmgProlongAdd(const GmgFab<T>& crse, GmgFab<T>& fine, bool onDevice)
{
    amrex::Gpu::LaunchSafeGuard lsg(onDevice);
    for (amrex::MFIter mfi(fine); mfi.isValid(); ++mfi)
    {
        const amrex::Box& vbx = mfi.validbox();
        const auto c = crse.const_array(mfi);
        const auto f = fine.array(mfi);
        amrex::HostDeviceParallelFor(
            vbx,
            [=] AMREX_GPU_HOST_DEVICE(int i, int j, int k) noexcept
            { f(i, j, k) += c(amrex::coarsen(i, 2), amrex::coarsen(j, 2), amrex::coarsen(k, 2)); }
        );
    }
}

template void gmgProlongAdd<double>(const GmgFab<double>&, GmgFab<double>&, bool);
template void gmgProlongAdd<float>(const GmgFab<float>&, GmgFab<float>&, bool);

// Fused residual + volume-average restriction: coarse rhs = mean of the 8 fine r = rhs - A sol,
// computed on the fly, saving the fine-grid resid read+write of two separate passes. Iterates the
// coarse box, so the fine sol's ghosts must already be filled.
template<class T>
void gmgResidRestrict(
    const GmgFab<T>& sol,
    const GmgFab<T>& rhs,
    GmgFab<T>& crhs,
    const FaceCoeffs<T>& fc,
    bool onDevice
)
{
    amrex::Gpu::LaunchSafeGuard lsg(onDevice);
    for (amrex::MFIter mfi(crhs); mfi.isValid(); ++mfi)
    {
        const amrex::Box& vbx = mfi.validbox();
        const auto psi = sol.const_array(mfi);
        const auto b = rhs.const_array(mfi);
        const auto cr = crhs.array(mfi);
        const auto al = fc.alpha->const_array(mfi);
        const FaceCoeffArrays fca {
            fc.ux->const_array(mfi),
            fc.lx->const_array(mfi),
            fc.uy->const_array(mfi),
            fc.ly->const_array(mfi),
            fc.uz->const_array(mfi),
            fc.lz->const_array(mfi)
        };
        amrex::HostDeviceParallelFor(
            vbx,
            [=] AMREX_GPU_HOST_DEVICE(int ic, int jc, int kc) noexcept
            {
                T acc = 0;
                for (int dk = 0; dk < 2; ++dk)
                {
                    for (int dj = 0; dj < 2; ++dj)
                    {
                        for (int di = 0; di < 2; ++di)
                        {
                            const int i = 2 * ic + di, j = 2 * jc + dj, k = 2 * kc + dk;
                            const auto c = loadFaceCoeffs<T>(fca, i, j, k);
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
                            acc += b(i, j, k) - (diag * psi(i, j, k) + off);
                        }
                    }
                }
                cr(ic, jc, kc) = static_cast<T>(0.125) * acc;
            }
        );
    }
}

template void gmgResidRestrict<double>(
    const GmgFab<double>&, const GmgFab<double>&, GmgFab<double>&, const FaceCoeffs<double>&, bool
);
template void gmgResidRestrict<float>(
    const GmgFab<float>&, const GmgFab<float>&, GmgFab<float>&, const FaceCoeffs<float>&, bool
);

} // namespace blockamr::la
