// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <Kokkos_Core.hpp>

#include <AMReX_Math.H>
#include <AMReX_MultiFab.H>

#include "../gmg/gmg_kernels.hpp"
#include "launch.hpp"

// ---------------------------------------------------------------------------
// Kokkos twins of the native GMG V-cycle kernels: a 1:1 port of the three
// *Device* kernels the timed V-cycle actually runs, in the same order, with the
// same signatures and the same cell arithmetic. Each one is the Kokkos sibling of
// a function in gmg_kernels.hpp, which already carries a *Device / *Host
// twin per kernel; these are the third twin, so the correspondence stays
// reviewable side by side:
//
//   gmgGsColorKokkos       <- gmgGsColorDevice        (gmg_kernels.hpp:385)
//   gmgResidRestrictKokkos <- gmgResidRestrictDevice  (gmg_kernels.hpp:691)
//   gmgProlongAddKokkos    <- gmgProlongAddDevice     (gmg_kernels.hpp:647)
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
//     decomposition (see vcycle.hpp) -- likewise a data movement.
//   * setVal on the coarse solution, and the hierarchy setup (coefficient
//     restriction / face coarsening): AMReX for both backends. Setup runs once and
//     is untimed.
//
// halo.hpp then goes on to port the first three anyway, for the `kokkos_opt`
// backend alone. Not because they are cell loops -- they are not -- but because each
// one is a synchronisation point between the two runtimes, and a cycle with none of
// them left needs no host fence at all. The `fence` argument of the fused launchers
// below is what lets that backend drop the per-kernel fence while the baselines keep
// it; see vcycle.hpp.
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

// TC is the COEFFICIENT storage type, separate from the field storage type T and
// defaulted to it. The split exists because the two carry different error
// sensitivities: psi and b are what the residual is formed as a difference of, so
// rounding them is amplified by ||A|| ~ 6/dx^2 (bf16.hpp measured the damage), while
// a coefficient rounded to 0.4% is a 0.4% perturbation of the preconditioner's
// operator -- something a Krylov method absorbs into its iteration count without
// amplification, because the operator CG stops on is still the fp64 one. Making them
// one type forced the safe choice on both; making them two lets the coefficients --
// 4 of the 6 arrays a shared-coefficient colour sweep streams -- be the narrow ones.
template<class T, class TC = T>
struct GmgGsCell
{
    using C = solvers::GmgComputeT<T>;

    amrex::Array4<T> psi;
    amrex::Array4<const T> b;
    amrex::Array4<const TC> ax, lxa, ay, lya, az, lza, al;
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

template<class T, class TC = T>
struct GmgResidRestrictCell
{
    using C = solvers::GmgComputeT<T>;

    amrex::Array4<T> cr;
    amrex::Array4<const T> psi, b;
    amrex::Array4<const TC> ax, lxa, ay, lya, az, lza, al;

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
template<class T, class TC>
void gmgGsColorKokkos(
    solvers::GmgFab<T>& sol,
    const solvers::GmgFab<T>& rhs,
    const solvers::GmgFab<TC>& ux,
    const solvers::GmgFab<TC>& lx,
    const solvers::GmgFab<TC>& uy,
    const solvers::GmgFab<TC>& ly,
    const solvers::GmgFab<TC>& uz,
    const solvers::GmgFab<TC>& lz,
    const solvers::GmgFab<TC>& alpha,
    int parity,
    double omega
)
{
    const solvers::GmgComputeT<T> om = static_cast<solvers::GmgComputeT<T>>(omega);
    for (amrex::MFIter mfi(rhs, gmgNoSync()); mfi.isValid(); ++mfi)
    {
        const GmgGsCell<T, TC> cell {
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
template<class T, class TC>
void gmgResidRestrictKokkos(
    const solvers::GmgFab<T>& sol,
    const solvers::GmgFab<T>& rhs,
    solvers::GmgFab<T>& crhs,
    const solvers::GmgFab<TC>& ux,
    const solvers::GmgFab<TC>& lx,
    const solvers::GmgFab<TC>& uy,
    const solvers::GmgFab<TC>& ly,
    const solvers::GmgFab<TC>& uz,
    const solvers::GmgFab<TC>& lz,
    const solvers::GmgFab<TC>& alpha
)
{
    for (amrex::MFIter mfi(crhs, gmgNoSync()); mfi.isValid(); ++mfi)
    {
        const GmgResidRestrictCell<T, TC> cell {
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
// inter-level kernels are handed (vcycle.hpp routes an agglomerated level
// through a transfer fab on the fine level's layout precisely to keep it true).
//
// `fence` is the ordering against whatever runs next. It has to be true whenever the
// next operation is AMReX's (the default, and what every backend but `kokkos_opt`
// needs after every kernel); it can be false when the next operation is another
// Kokkos kernel on the same execution space, which is already ordered by the stream.
// ---------------------------------------------------------------------------

// Declared here, DEFINED (and explicitly instantiated for every {T, TC} the
// V-cycle needs) in kernels.cpp -- NOT header-inline, unlike every other
// launcher in this file. These three are the ones KokkosOptGmgBackend (vcycle.hpp)
// calls, and vcycle.hpp is included by BOTH apply.cpp (production) and
// bench/gmg_vcycle_bench.cpp (the bench harness, which also calls them directly for
// kokkos_fused): a header-inline template here would instantiate an identical
// extended-__host__-__device__-lambda-bearing function in TWO CUDA translation
// units feeding the SAME final shared object, which is an nvcc trap -- the linker's
// weak/COMDAT folding of the host-side stub does not keep the two TUs' device-side
// registrations consistent, and the result is a null function-pointer call at
// runtime (not a compile or link error). Emitting the definition in exactly one TU
// and only DECLARING it here removes the duplicate instantiation entirely. See
// kernels.cpp for the instantiation list.
template<class T, class TC>
void gmgGsColorKokkosFused(
    solvers::GmgFab<T>& sol,
    const solvers::GmgFab<T>& rhs,
    const solvers::GmgFab<TC>& ux,
    const solvers::GmgFab<TC>& lx,
    const solvers::GmgFab<TC>& uy,
    const solvers::GmgFab<TC>& ly,
    const solvers::GmgFab<TC>& uz,
    const solvers::GmgFab<TC>& lz,
    const solvers::GmgFab<TC>& alpha,
    int parity,
    double omega,
    bool fence = true
);

template<class T, class TC>
void gmgResidRestrictKokkosFused(
    const solvers::GmgFab<T>& sol,
    const solvers::GmgFab<T>& rhs,
    solvers::GmgFab<T>& crhs,
    const solvers::GmgFab<TC>& ux,
    const solvers::GmgFab<TC>& lx,
    const solvers::GmgFab<TC>& uy,
    const solvers::GmgFab<TC>& ly,
    const solvers::GmgFab<TC>& uz,
    const solvers::GmgFab<TC>& lz,
    const solvers::GmgFab<TC>& alpha,
    bool fence = true
);

template<class T>
void gmgProlongAddKokkosFused(
    const solvers::GmgFab<T>& crse, solvers::GmgFab<T>& fine, bool fence = true
);

} // namespace blockamr::bench
