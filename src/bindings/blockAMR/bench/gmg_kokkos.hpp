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
// Each comes in two launch forms with identical signatures and one shared body:
//
//   *Kokkos       one MDRangePolicy per box, the shape production is written in.
//   *KokkosFused  one TeamPolicy for all boxes of the level, so per-box launch
//                 cost cannot appear. This is the Kokkos twin of AMReX's own
//                 ParallelFor(mf, f) (AMReX_MFParallelForG.H) and the reason the
//                 cell arithmetic below is factored into *Cell structs: the two
//                 launchers must not be able to drift apart.
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
// What is deliberately NOT ported here, because it is not a cell kernel:
//
//   * FillBoundary -- the halo exchange. It is AMReX packing/unpacking its own
//     FabArray metadata (and MPI in general), not a loop over cells, so the `kokkos`
//     and `kokkos_fused` backends call the same AMReX FillBoundary. This is a real
//     result about the scope of a port, not a gap in this one.
//   * ParallelCopy between an agglomerated level and its fine neighbour's
//     decomposition (see gmg_vcycle.cpp) -- likewise a data movement.
//   * setVal on the coarse solution, and the hierarchy setup (coefficient
//     restriction / face coarsening): AMReX for both backends. Setup runs once and
//     is untimed.
//
// halo_kokkos.hpp then goes on to port the first three anyway, for the `kokkos_opt`
// backend alone. Not because they are cell loops -- they are not -- but because each
// one is a synchronisation point between the two runtimes, and a cycle with none of
// them left needs no host fence at all. The `fence` argument of the fused launchers
// below is what lets that backend drop the per-kernel fence while the baselines keep
// it; see gmg_vcycle.cpp.
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

// ---------------------------------------------------------------------------
// The cell arithmetic, one struct per kernel: the Array4s of a single box plus the
// update for one cell. The per-box launcher builds one on the host per box; the
// fused launcher builds one on the device per cell out of AMReX's cached Array4
// table. Written once so a fused launch cannot silently compute something else.
//
// The Array4s carry T, the level's STORAGE type; every local is declared C =
// solvers::GmgComputeT<T>, the type the arithmetic happens in. For T = double and
// T = float those are the same type, so these structs stay the character-for-
// character twins of the *Device kernels named above and generate the code they
// always did. For T = Bf16 they are what makes the hierarchy storage-only, and
// `acc`, `off` and `diag` are exactly where it matters: the residual
// b - (diag*psi + off) subtracts two numbers ~7000x larger than itself, which in
// bf16 round to the SAME value and cancel to exactly zero. See bf16.hpp.
// ---------------------------------------------------------------------------

template<class T>
struct GmgGsCell
{
    using C = solvers::GmgComputeT<T>;

    amrex::Array4<T> psi;
    amrex::Array4<const T> b, ax, lxa, ay, lya, az, lza, al;
    C om;
    int parity;

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void operator()(int i, int j, int k) const
    {
        if (((i + j + k) & 1) != parity)
        {
            return;
        }
        const C aE = ax(i + 1, j, k);
        const C aW = lxa(i, j, k);
        const C aN = ay(i, j + 1, k);
        const C aS = lya(i, j, k);
        const C aT = az(i, j, k + 1);
        const C aB = lza(i, j, k);
        const C off = aE * psi(i + 1, j, k) + aW * psi(i - 1, j, k) + aN * psi(i, j + 1, k)
                    + aS * psi(i, j - 1, k) + aT * psi(i, j, k + 1) + aB * psi(i, j, k - 1);
        const C diag = al(i, j, k) - (aE + aW + aN + aS + aT + aB);
        if (amrex::Math::abs(diag) > solvers::gmgDiagFloor<C>())
        {
            const C gs = (b(i, j, k) - off) / diag;
            psi(i, j, k) += om * (gs - static_cast<C>(psi(i, j, k)));
        }
    }
};

template<class T>
struct GmgResidRestrictCell
{
    using C = solvers::GmgComputeT<T>;

    amrex::Array4<T> cr;
    amrex::Array4<const T> psi, b, ax, lxa, ay, lya, az, lza, al;

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void operator()(int ic, int jc, int kc) const
    {
        C acc = 0;
        for (int dk = 0; dk < 2; ++dk)
        {
            for (int dj = 0; dj < 2; ++dj)
            {
                for (int di = 0; di < 2; ++di)
                {
                    const int i = 2 * ic + di, j = 2 * jc + dj, k = 2 * kc + dk;
                    const C aE = ax(i + 1, j, k);
                    const C aW = lxa(i, j, k);
                    const C aN = ay(i, j + 1, k);
                    const C aS = lya(i, j, k);
                    const C aT = az(i, j, k + 1);
                    const C aB = lza(i, j, k);
                    const C off = aE * psi(i + 1, j, k) + aW * psi(i - 1, j, k)
                                + aN * psi(i, j + 1, k) + aS * psi(i, j - 1, k)
                                + aT * psi(i, j, k + 1) + aB * psi(i, j, k - 1);
                    const C diag = al(i, j, k) - (aE + aW + aN + aS + aT + aB);
                    acc += b(i, j, k) - (diag * psi(i, j, k) + off);
                }
            }
        }
        cr(ic, jc, kc) = static_cast<C>(0.125) * acc;
    }
};

template<class T>
struct GmgProlongCell
{
    amrex::Array4<T> f;
    amrex::Array4<const T> c;

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void operator()(int i, int j, int k) const
    {
        f(i, j, k) += c(amrex::coarsen(i, 2), amrex::coarsen(j, 2), amrex::coarsen(k, 2));
    }
};

// ---------------------------------------------------------------------------
// Per-box launchers: one MDRangePolicy per box.
// ---------------------------------------------------------------------------

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
    const solvers::GmgComputeT<T> om = static_cast<solvers::GmgComputeT<T>>(omega);
    for (amrex::MFIter mfi(rhs, gmgNoSync()); mfi.isValid(); ++mfi)
    {
        const GmgGsCell<T> cell {
            sol.array(mfi),
            rhs.const_array(mfi),
            ux.const_array(mfi),
            lx.const_array(mfi),
            uy.const_array(mfi),
            ly.const_array(mfi),
            uz.const_array(mfi),
            lz.const_array(mfi),
            alpha.const_array(mfi),
            om,
            parity
        };
        launchKokkosMdNamed(
            "gmg_gs", mfi.validbox(), BENCH_LAMBDA(int i, int j, int k) { cell(i, j, k); }
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
        const GmgResidRestrictCell<T> cell {
            crhs.array(mfi),
            sol.const_array(mfi),
            rhs.const_array(mfi),
            ux.const_array(mfi),
            lx.const_array(mfi),
            uy.const_array(mfi),
            ly.const_array(mfi),
            uz.const_array(mfi),
            lz.const_array(mfi),
            alpha.const_array(mfi)
        };
        launchKokkosMdNamed(
            "gmg_residrestrict",
            mfi.validbox(),
            BENCH_LAMBDA(int ic, int jc, int kc) { cell(ic, jc, kc); }
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
        const GmgProlongCell<T> cell {fine.array(mfi), crse.const_array(mfi)};
        launchKokkosMdNamed(
            "gmg_prolong", mfi.validbox(), BENCH_LAMBDA(int i, int j, int k) { cell(i, j, k); }
        );
    }
    Kokkos::fence();
}

// ---------------------------------------------------------------------------
// Fused launchers: ONE TeamPolicy launch covers every box of the level, so the
// per-box launch cost that dominates the coarse levels of the hierarchy cannot
// appear. arrays()/const_arrays() is AMReX's cached device Array4 table, built
// once per FabArray, so the per-launch cost is a pointer copy.
//
// The fused loop is driven by the fab whose valid boxes define the iteration space
// -- rhs for the smoother, the COARSE rhs for the restriction, the FINE solution
// for the prolongation -- and every other field is addressed at the same local box
// index. That is exact whenever the fabs share a DistributionMapping and box order,
// which holds for the fields of one level and for the fine/coarse pair the
// inter-level kernels are handed (gmg_vcycle.cpp routes an agglomerated level
// through a transfer fab on the fine level's layout precisely to keep it true).
//
// `fence` is the ordering against whatever runs next. It has to be true whenever the
// next operation is AMReX's (the default, and what every backend but `kokkos_opt`
// needs after every kernel); it can be false when the next operation is another
// Kokkos kernel on the same execution space, which is already ordered by the stream.
// ---------------------------------------------------------------------------

template<class T>
void gmgGsColorKokkosFused(
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
    double omega,
    bool fence = true
)
{
    const solvers::GmgComputeT<T> om = static_cast<solvers::GmgComputeT<T>>(omega);
    const auto psi = sol.arrays();
    const auto b = rhs.const_arrays();
    const auto ax = ux.const_arrays();
    const auto lxa = lx.const_arrays();
    const auto ay = uy.const_arrays();
    const auto lya = ly.const_arrays();
    const auto az = uz.const_arrays();
    const auto lza = lz.const_arrays();
    const auto al = alpha.const_arrays();
    launchKokkosTeamNamed(
        "gmg_gs_fused",
        rhs,
        BENCH_LAMBDA(int ib, int i, int j, int k) {
            GmgGsCell<T> {
                psi[ib],
                b[ib],
                ax[ib],
                lxa[ib],
                ay[ib],
                lya[ib],
                az[ib],
                lza[ib],
                al[ib],
                om,
                parity
            }(i, j, k);
        }
    );
    if (fence)
    {
        Kokkos::fence();
    }
}

template<class T>
void gmgResidRestrictKokkosFused(
    const solvers::GmgFab<T>& sol,
    const solvers::GmgFab<T>& rhs,
    solvers::GmgFab<T>& crhs,
    const solvers::GmgFab<T>& ux,
    const solvers::GmgFab<T>& lx,
    const solvers::GmgFab<T>& uy,
    const solvers::GmgFab<T>& ly,
    const solvers::GmgFab<T>& uz,
    const solvers::GmgFab<T>& lz,
    const solvers::GmgFab<T>& alpha,
    bool fence = true
)
{
    const auto cr = crhs.arrays();
    const auto psi = sol.const_arrays();
    const auto b = rhs.const_arrays();
    const auto ax = ux.const_arrays();
    const auto lxa = lx.const_arrays();
    const auto ay = uy.const_arrays();
    const auto lya = ly.const_arrays();
    const auto az = uz.const_arrays();
    const auto lza = lz.const_arrays();
    const auto al = alpha.const_arrays();
    launchKokkosTeamNamed(
        "gmg_residrestrict_fused",
        crhs,
        BENCH_LAMBDA(int ib, int ic, int jc, int kc) {
            GmgResidRestrictCell<T> {
                cr[ib], psi[ib], b[ib], ax[ib], lxa[ib], ay[ib], lya[ib], az[ib], lza[ib], al[ib]
            }(ic, jc, kc);
        }
    );
    if (fence)
    {
        Kokkos::fence();
    }
}

template<class T>
void gmgProlongAddKokkosFused(
    const solvers::GmgFab<T>& crse, solvers::GmgFab<T>& fine, bool fence = true
)
{
    const auto f = fine.arrays();
    const auto c = crse.const_arrays();
    launchKokkosTeamNamed(
        "gmg_prolong_fused",
        fine,
        BENCH_LAMBDA(int ib, int i, int j, int k) {
            GmgProlongCell<T> {f[ib], c[ib]}(i, j, k);
        }
    );
    if (fence)
    {
        Kokkos::fence();
    }
}

} // namespace blockamr::bench
