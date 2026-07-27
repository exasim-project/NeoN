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

} // namespace blockamr::solvers
