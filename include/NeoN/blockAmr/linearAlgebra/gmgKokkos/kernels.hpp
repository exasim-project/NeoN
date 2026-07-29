// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <Kokkos_Core.hpp>

#include <AMReX_Math.H>
#include <AMReX_MultiFab.H>

#include "NeoN/blockAmr/linearAlgebra/gmg/gmgKernels.hpp"
#include "NeoN/blockAmr/core/launch.hpp"

// Kokkos twins of the three GMG V-cycle kernels the timed cycle runs -- same order,
// same signatures, same cell arithmetic as their gmgKernels.hpp siblings (gmgGsColor,
// gmgResidRestrict, gmgProlongAdd), so the correspondence stays reviewable side by
// side.
//
// Two launch forms per kernel, same signature, one shared body: *Kokkos is one
// MDRangePolicy per box (the shape production is written in), *KokkosFused one
// TeamPolicy for all boxes of the level, so per-box launch cost cannot appear. The cell
// arithmetic is factored into *Cell structs precisely so the two cannot drift apart.
//
// Kokkos writes the MultiFab memory on its own default execution space, which costs ONE
// fence per kernel: the V-cycle interleaves these with AMReX and the two runtimes'
// streams are unordered. Not a handicap -- production's default-MFItInfo MFIter
// stream-synchronizes in its destructor too, so both backends sync once per kernel. The
// twins pass DisableDeviceSync, dropping the AMReX sync that has no AMReX kernel left
// to wait for rather than paying both.
//
// Deliberately NOT ported here, because they are data movements and not cell loops:
// FillBoundary, the agglomerated-level ParallelCopy, setVal and the hierarchy setup.
// halo.hpp ports the first three anyway for `kokkos_opt` alone -- not as cell loops but
// because each is a cross-runtime synchronisation point, and a cycle with none left
// needs no host fence at all; the fused launchers' `fence` argument is what lets that
// backend drop it while the baselines keep it.
//
// Accessors are amrex::Array4 for both backends, not an unmanaged Kokkos View: the
// operator bench showed the accessor choice sits within the noise floor, so the
// LAUNCHER stays the only difference between the two columns.

namespace blockamr
{

// The box loop launches no AMReX kernel, so AMReX has nothing to synchronize; the
// Kokkos fence is what orders these kernels against whatever AMReX does next.
inline amrex::MFItInfo gmgNoSync() { return amrex::MFItInfo().DisableDeviceSync(); }

// The cell arithmetic, one struct per kernel: one box's Array4s plus the update of one
// cell. Written once so a fused launch cannot silently compute something else.
//
// The Array4s carry T, the level's STORAGE type; every local is C = la::GmgComputeT<T>,
// the type the arithmetic happens in. For double and float those coincide, so these
// stay the character-for-character twins of the *Device kernels. For Bf16 they are what
// keeps the hierarchy storage-only, and `acc`, `off` and `diag` are where it matters:
// the residual b - (diag*psi + off) subtracts two numbers ~7000x larger than itself,
// which in bf16 round to the SAME value and cancel to exactly zero (gmg/bf16.hpp).

// TC is the COEFFICIENT storage type, split from the field type T because the two have
// very different error sensitivities. The outer operator, residual and stopping test
// stay fp64, so a narrow cycle can cost iterations but never correctness. Measured:
// rounding the SOLUTION vector is amplified, because the cycle restricts b - A psi and
// storage error therefore reaches the coarse grid multiplied by ||A|| ~ 6/dx^2 -- the
// cycle weakens as n^2, CG counts more than double at 64^3 and blow up at 256^3, so
// bf16 FIELDS win at no size at all. Rounding a COEFFICIENT is not amplified that way:
// 0.4% is a 0.4% perturbation of the preconditioner's operator, which a Krylov method
// absorbs into its iteration count -- and coefficients are 4 of the 6 arrays a
// shared-coefficient colour sweep streams, which is why they are the ones worth
// narrowing.
template<class T, class TC = T>
struct GmgGsCell
{
    using C = la::GmgComputeT<T>;

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
        if (amrex::Math::abs(diag) > la::gmgDiagFloor<C>())
        {
            const C gs = (b(i, j, k) - off) / diag;
            psi(i, j, k) += om * (gs - static_cast<C>(psi(i, j, k)));
        }
    }
};

template<class T, class TC = T>
struct GmgResidRestrictCell
{
    using C = la::GmgComputeT<T>;

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

// Per-box launchers: one MDRangePolicy per box.

// One red-black over-relaxation colour pass. Twin of gmgGsColor.
template<class T, class TC>
void gmgGsColorKokkos(
    la::GmgFab<T>& sol,
    const la::GmgFab<T>& rhs,
    const la::GmgFab<TC>& ux,
    const la::GmgFab<TC>& lx,
    const la::GmgFab<TC>& uy,
    const la::GmgFab<TC>& ly,
    const la::GmgFab<TC>& uz,
    const la::GmgFab<TC>& lz,
    const la::GmgFab<TC>& alpha,
    int parity,
    double omega
)
{
    const la::GmgComputeT<T> om = static_cast<la::GmgComputeT<T>>(omega);
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
            "gmg_gs", mfi.validbox(), BLOCKAMR_LAMBDA(int i, int j, int k) { cell(i, j, k); }
        );
    }
    Kokkos::fence();
}

// Fused residual + volume-average restriction. Twin of gmgResidRestrict.
template<class T, class TC>
void gmgResidRestrictKokkos(
    const la::GmgFab<T>& sol,
    const la::GmgFab<T>& rhs,
    la::GmgFab<T>& crhs,
    const la::GmgFab<TC>& ux,
    const la::GmgFab<TC>& lx,
    const la::GmgFab<TC>& uy,
    const la::GmgFab<TC>& ly,
    const la::GmgFab<TC>& uz,
    const la::GmgFab<TC>& lz,
    const la::GmgFab<TC>& alpha
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
            BLOCKAMR_LAMBDA(int ic, int jc, int kc) { cell(ic, jc, kc); }
        );
    }
    Kokkos::fence();
}

// Piecewise-constant prolongation + correction. Twin of gmgProlongAdd.
template<class T>
void gmgProlongAddKokkos(const la::GmgFab<T>& crse, la::GmgFab<T>& fine)
{
    for (amrex::MFIter mfi(fine, gmgNoSync()); mfi.isValid(); ++mfi)
    {
        const GmgProlongCell<T> cell {fine.array(mfi), crse.const_array(mfi)};
        launchKokkosMdNamed(
            "gmg_prolong", mfi.validbox(), BLOCKAMR_LAMBDA(int i, int j, int k) { cell(i, j, k); }
        );
    }
    Kokkos::fence();
}

// Fused launchers: ONE TeamPolicy launch per level, so the per-box launch cost that
// dominates the coarse levels cannot appear. arrays()/const_arrays() is AMReX's cached
// device Array4 table, so the per-launch cost is a pointer copy.
//
// The loop is driven by the fab whose valid boxes define the iteration space -- rhs for
// the smoother, the COARSE rhs for the restriction, the FINE solution for the
// prolongation -- and every other field is addressed at the same local box index. Exact
// whenever the fabs share a DistributionMapping and box order, which vcycle.hpp keeps
// true by routing an agglomerated level through a transfer fab on the fine layout.
//
// `fence` orders against whatever runs next: true whenever that is AMReX (the default,
// and what every backend but `kokkos_opt` needs), false when it is another Kokkos
// kernel on the same execution space.

// Declaration-only here, DEFINED and explicitly instantiated in kernels.cpp -- unlike
// every other launcher in this file. nvcc rule: a kernel reached from more than one
// CUDA TU (here apply.cpp and bench/gmgVcycleBench.cpp, both feeding one shared object)
// must be declaration-only in the header, because duplicating an extended __host__
// __device__ lambda's instantiation leaves the two TUs' device-side registrations
// inconsistent. WARNING: a missing instantiation is then a NULL DEVICE FUNCTION POINTER
// AT RUNTIME, not a link error -- kernels.cpp carries 12 {T, TC} instantiations for
// these kernels and they must stay in sync.
template<class T, class TC>
void gmgGsColorKokkosFused(
    la::GmgFab<T>& sol,
    const la::GmgFab<T>& rhs,
    const la::GmgFab<TC>& ux,
    const la::GmgFab<TC>& lx,
    const la::GmgFab<TC>& uy,
    const la::GmgFab<TC>& ly,
    const la::GmgFab<TC>& uz,
    const la::GmgFab<TC>& lz,
    const la::GmgFab<TC>& alpha,
    int parity,
    double omega,
    bool fence = true
);

// Declaration-only for the same reason; a kernels.cpp instantiation missing here is a
// null device function pointer at runtime, not a link error.
template<class T, class TC>
void gmgResidRestrictKokkosFused(
    const la::GmgFab<T>& sol,
    const la::GmgFab<T>& rhs,
    la::GmgFab<T>& crhs,
    const la::GmgFab<TC>& ux,
    const la::GmgFab<TC>& lx,
    const la::GmgFab<TC>& uy,
    const la::GmgFab<TC>& ly,
    const la::GmgFab<TC>& uz,
    const la::GmgFab<TC>& lz,
    const la::GmgFab<TC>& alpha,
    bool fence = true
);

// Declaration-only for the same reason; a kernels.cpp instantiation missing here is a
// null device function pointer at runtime, not a link error.
template<class T>
void gmgProlongAddKokkosFused(const la::GmgFab<T>& crse, la::GmgFab<T>& fine, bool fence = true);

} // namespace blockamr
