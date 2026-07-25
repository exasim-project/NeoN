// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <Kokkos_Core.hpp>

#include <AMReX_Math.H>
#include <AMReX_MultiFab.H>

#include "../solvers/gmg_kernels.hpp"
#include "launch.hpp"

// ---------------------------------------------------------------------------
// Kokkos twins of the native GMG V-cycle kernels: a 1:1 port of the three
// *Device* kernels the timed V-cycle actually runs, in the same order, with the
// same signatures and the same cell arithmetic. Each one is the Kokkos sibling of
// a function in solvers/gmg_kernels.hpp, which already carries a *Device / *Host
// twin per kernel; these are the third twin, so the correspondence stays
// reviewable side by side:
//
//   gmgGsColorKokkos       <- gmgGsColorDevice        (gmg_kernels.hpp:330)
//   gmgResidRestrictKokkos <- gmgResidRestrictDevice  (gmg_kernels.hpp:636)
//   gmgProlongAddKokkos    <- gmgProlongAddDevice     (gmg_kernels.hpp:592)
//
// Kokkos writes the MultiFab memory directly on its own default execution space:
// no AMReX stream is borrowed and no execution space instance is managed. What that
// costs is ONE fence per kernel, because the V-cycle interleaves these kernels with
// AMReX FillBoundary and the two runtimes' streams are otherwise unordered. That is
// not a handicap against AMReX: production launches every one of these kernels from
// a default-MFItInfo MFIter, whose destructor stream-synchronizes at the end of the
// box loop (AMReX_MFIter.cpp:246) -- so both backends sync exactly once per kernel,
// each through its own runtime. The Kokkos twins therefore pass DisableDeviceSync,
// which drops AMReX's now-meaningless sync (no AMReX kernel was launched) rather
// than paying both.
//
// What is deliberately NOT ported, because it is not a cell kernel:
//
//   * FillBoundary -- the halo exchange. It is AMReX packing/unpacking its own
//     FabArray metadata (and MPI in general), not a loop over cells, so both
//     backends call the same AMReX FillBoundary. This is a real result about the
//     scope of a port, not a gap in this one.
//   * setVal on the coarse solution, and the hierarchy setup (coefficient
//     restriction / face coarsening): AMReX for both backends. Setup runs once and
//     is untimed.
//
// Accessors are amrex::Array4 for both backends, not an unmanaged Kokkos View.
// The operator bench already isolated that variable -- kokkos_md_a4 (Kokkos
// launcher, AMReX accessor) tracks kokkos_md within the noise floor -- so keeping
// Array4 here leaves the LAUNCHER as the only difference between the two columns.
// ---------------------------------------------------------------------------

namespace blockamr::bench
{

// The box loop launches no AMReX kernel, so AMReX has nothing to synchronize at the
// end of it; the Kokkos fence below is what orders this kernel against whatever
// AMReX does next.
inline amrex::MFItInfo gmgNoSync() { return amrex::MFItInfo().DisableDeviceSync(); }

// One red-black over-relaxation colour pass. Twin of gmgGsColorDevice.
template<class T>
void gmgGsColorKokkos(
    solvers::GmgFab<T>& sol,
    const solvers::GmgFab<T>& rhs,
    const solvers::GmgFab<T>& ux,
    const solvers::GmgFab<T>& lx,
    const solvers::GmgFab<T>& uy,
    const solvers::GmgFab<T>& ly,
    const solvers::GmgFab<T>& uz,
    const solvers::GmgFab<T>& lz,
    const solvers::GmgFab<T>& alpha,
    int parity,
    double omega
)
{
    const T om = static_cast<T>(omega);
    for (amrex::MFIter mfi(rhs, gmgNoSync()); mfi.isValid(); ++mfi)
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
        launchKokkosMdNamed(
            "gmg_gs",
            vbx,
            KOKKOS_LAMBDA(int i, int j, int k) {
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
                if (amrex::Math::abs(diag) > solvers::gmgDiagFloor<T>())
                {
                    const T gs = (b(i, j, k) - off) / diag;
                    psi(i, j, k) += om * (gs - psi(i, j, k));
                }
            }
        );
    }
    Kokkos::fence();
}

// Fused residual + volume-average restriction. Twin of gmgResidRestrictDevice.
template<class T>
void gmgResidRestrictKokkos(
    const solvers::GmgFab<T>& sol,
    const solvers::GmgFab<T>& rhs,
    solvers::GmgFab<T>& crhs,
    const solvers::GmgFab<T>& ux,
    const solvers::GmgFab<T>& lx,
    const solvers::GmgFab<T>& uy,
    const solvers::GmgFab<T>& ly,
    const solvers::GmgFab<T>& uz,
    const solvers::GmgFab<T>& lz,
    const solvers::GmgFab<T>& alpha
)
{
    for (amrex::MFIter mfi(crhs, gmgNoSync()); mfi.isValid(); ++mfi)
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
        launchKokkosMdNamed(
            "gmg_residrestrict",
            vbx,
            KOKKOS_LAMBDA(int ic, int jc, int kc) {
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
    Kokkos::fence();
}

// Piecewise-constant prolongation + correction. Twin of gmgProlongAddDevice.
template<class T>
void gmgProlongAddKokkos(const solvers::GmgFab<T>& crse, solvers::GmgFab<T>& fine)
{
    for (amrex::MFIter mfi(fine, gmgNoSync()); mfi.isValid(); ++mfi)
    {
        const amrex::Box& vbx = mfi.validbox();
        const auto c = crse.const_array(mfi);
        const auto f = fine.array(mfi);
        launchKokkosMdNamed(
            "gmg_prolong",
            vbx,
            KOKKOS_LAMBDA(int i, int j, int k) {
                f(i, j, k) += c(amrex::coarsen(i, 2), amrex::coarsen(j, 2), amrex::coarsen(k, 2));
            }
        );
    }
    Kokkos::fence();
}

} // namespace blockamr::bench
