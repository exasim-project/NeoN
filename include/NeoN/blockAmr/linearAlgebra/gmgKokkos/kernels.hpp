// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <Kokkos_Core.hpp>

#include <AMReX_Math.H>
#include <AMReX_MultiFab.H>

#include "NeoN/blockAmr/linearAlgebra/gmg/gmgKernels.hpp"
#include "NeoN/blockAmr/core/launch.hpp"

// Kokkos twins of the three V-cycle kernels the timed cycle runs (gmgGsColor,
// gmgResidRestrict, gmgProlongAdd): same order, same signatures, same cell arithmetic, two
// launch forms each. Fences, accessors and what is NOT ported: report/blockamr-gmg-notes.md.

namespace blockamr
{

// The box loop launches no AMReX kernel, so only the Kokkos fence orders these kernels.
inline amrex::MFItInfo gmgNoSync() { return amrex::MFItInfo().DisableDeviceSync(); }

// The cell arithmetic, one struct per kernel, written once so a fused launch cannot silently
// compute something else. Array4s carry the STORAGE type T; locals are C = la::GmgComputeT<T>.

// TC is the COEFFICIENT storage type, split from the field type T: rounding a coefficient
// perturbs only the preconditioner's operator, while rounding a FIELD reaches the coarse grid
// multiplied by ||A|| ~ 6/dx^2 -- report/blockamr-precision-measurements.md.
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

// Fused launchers: ONE TeamPolicy launch per level, so per-box launch cost cannot appear. The
// loop is driven by the fab defining the iteration space and every other field is addressed at
// the same local box index; `fence` orders against what runs next (notes#kokkos-twins).

// Declaration-only, DEFINED and explicitly instantiated in kernels.cpp -- the nvcc rule for a
// kernel reached from more than one CUDA TU. A miss is a NULL DEVICE FUNCTION POINTER AT
// RUNTIME, not a link error; the 12 {T, TC} instantiations there must stay in sync.
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

// Same rule: a miss is a null device fnptr at runtime, not a link error.
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

// Same rule: a miss is a null device fnptr at runtime, not a link error.
template<class T>
void gmgProlongAddKokkosFused(const la::GmgFab<T>& crse, la::GmgFab<T>& fine, bool fence = true);

} // namespace blockamr
