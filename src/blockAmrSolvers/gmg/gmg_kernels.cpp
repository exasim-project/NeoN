// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

// Definitions -- and, below, the explicit instantiations -- of the
// gmg_kernels.hpp twin functions that are reached from more than one .cpp
// translation unit (Class B/C in the T9 report): declaring them as ordinary
// header templates and giving each an AMREX_GPU_HOST_DEVICE lambda would make
// them extended lambdas instantiated in two CUDA TUs of the same final
// _blockamr.so (persistent.cpp via blockamr_solvers, bench/gmg_vcycle_bench.cpp
// via blockamr_kokkos) -- the exact nvcc trap T2 already hit for the fused
// Kokkos kernels (see gmgKokkos/kernels.cpp). The fix is the same one used
// there: define the kernel in exactly one TU and explicitly instantiate every
// (T) this TU's callers need, so every other including TU sees only the
// declaration in gmg_kernels.hpp and links against this single definition.

#include "gmg_kernels.hpp"

namespace blockamr::solvers
{

template<class T>
void gmgGsColor(
    GmgFab<T>& sol,
    const GmgFab<T>& rhs,
    const FaceCoeffs<T>& fc,
    int parity,
    double omega,
    bool onDevice
)
{
    amrex::Gpu::LaunchSafeGuard lsg(onDevice);
    const T om = static_cast<T>(omega);
    for (amrex::MFIter mfi(rhs); mfi.isValid(); ++mfi)
    {
        const amrex::Box& vbx = mfi.validbox();
        const auto psi = sol.array(mfi);
        const auto b = rhs.const_array(mfi);
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
                if (((i + j + k) & 1) != parity)
                {
                    return;
                }
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
    GmgFab<double>&, const GmgFab<double>&, const FaceCoeffs<double>&, int, double, bool
);
template void gmgGsColor<float>(
    GmgFab<float>&, const GmgFab<float>&, const FaceCoeffs<float>&, int, double, bool
);

// Piecewise-constant prolongation + correction: fine cell += coarse parent
// value (the adjoint of the volume-average restriction, up to the 1/8 factor).
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

// Fused residual + volume-average restriction: coarse rhs cell = mean of the 8
// fine residuals r = rhs - A sol, each computed on the fly. Iterates the coarse
// box (fine sol's ghosts must be filled). Saves the full fine-grid resid
// read+write of the separate residual + restriction passes (M4 item 3).
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
        const auto ax = fc.ux->const_array(mfi);
        const auto lxa = fc.lx->const_array(mfi);
        const auto ay = fc.uy->const_array(mfi);
        const auto lya = fc.ly->const_array(mfi);
        const auto az = fc.uz->const_array(mfi);
        const auto lza = fc.lz->const_array(mfi);
        const auto al = fc.alpha->const_array(mfi);
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

} // namespace blockamr::solvers
