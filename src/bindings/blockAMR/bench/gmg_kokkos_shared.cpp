// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

// The definitions -- and, below, the explicit instantiations -- of the handful of
// Kokkos launchers that gmg_kokkos.hpp and halo_kokkos.hpp only DECLARE:
// gmgGsColorKokkosFused, gmgResidRestrictKokkosFused, gmgProlongAddKokkosFused,
// execCopyPlan and gmgZeroKokkos.
//
// Why these five and not the rest of the bench's launchers: every one of them opens
// an extended __host__ __device__ lambda (BENCH_LAMBDA / KOKKOS_LAMBDA) directly, and
// every one of them is reached from KokkosOptGmgBackend (vcycle.hpp), which is shared
// by bench/gmg_apply.cpp (production) and bench/gmg_vcycle_bench.cpp (the bench
// harness -- kokkos_fused also calls the first three of the five directly). Both
// files land in the same final shared object (blockamr_bench is an OBJECT library
// linked into _blockamr, not a separate .so -- see CMakeLists.txt), so a
// header-inline template definition here used to be instantiated once per including
// TU: two textually-identical extended-lambda-bearing functions compiled into two
// CUDA translation units of the SAME binary. That is an nvcc trap, not a portable
// C++ template pattern: the linker's weak/COMDAT folding keeps one TU's host-side
// stub, but the two TUs' device-side kernel registrations are not guaranteed
// consistent with it, and the observed failure is a null function-pointer call at
// runtime -- not a compile or link diagnostic (found via T2 retry 1: 43/102 gate
// tests SIGSEGV inside launchKokkosTeamNamed for exactly the kokkos_fused/kokkos_opt
// cases, i.e. exactly the callers of these five functions).
//
// The fix is the standard one for a kernel-launching template that must be visible
// from more than one TU: define it in exactly one TU and explicitly instantiate every
// combination that TU's callers need, so every OTHER including TU sees only a
// declaration and links against this single definition instead of generating its own.

#include "../solvers/bf16.hpp"
#include "../solvers/gmg_kernels.hpp"
#include "gmg_kokkos.hpp"
#include "halo_kokkos.hpp"

namespace blockamr::bench
{

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
    bool fence
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
            GmgGsCell<T, TC> {
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
    bool fence
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
            GmgResidRestrictCell<T, TC> {
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
void gmgProlongAddKokkosFused(const solvers::GmgFab<T>& crse, solvers::GmgFab<T>& fine, bool fence)
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

template<class T>
void execCopyPlan(
    const char* name,
    const amrex::MultiArray4<T>& dst,
    const amrex::MultiArray4<const T>& src,
    const CopyPlan& plan
)
{
    const int nblocks = plan.size();
    if (nblocks == 0)
    {
        return;
    }
    constexpr int VL = 32;
    // Function-local: a namespace-scope constexpr has no device symbol to reference
    // from the kernel body below.
    constexpr int MT = kCopyBlock;
    const auto tasks = plan.tasks;
    using Policy = Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>;
    Kokkos::parallel_for(
        name,
        Policy(nblocks, MT / VL, VL),
        KOKKOS_LAMBDA(const Policy::member_type& team) {
            const CopyTask t = tasks(team.league_rank());
            const int nx = t.len[0];
            const int nxy = t.len[0] * t.len[1];
            const int left = nxy * t.len[2] - t.base;
            Kokkos::parallel_for(
                Kokkos::TeamVectorRange(team, left < MT ? left : MT),
                [&](const int q)
                {
                    const int c = t.base + q;
                    const int i = t.lo[0] + c % nx;
                    const int j = t.lo[1] + (c / nx) % t.len[1];
                    const int k = t.lo[2] + c / nxy;
                    // The compute type, not T: for a bf16 level the two arms of the
                    // ternary would otherwise be Bf16 and float, each convertible to
                    // the other, which is ambiguous. Both the load and the sign flip
                    // are exact in it, so the copy still moves values unchanged.
                    const solvers::GmgComputeT<T> v =
                        src[t.src](i + t.sh[0], j + t.sh[1], k + t.sh[2]);
                    dst[t.dst](i, j, k) = (t.sign < 0) ? -v : v;
                }
            );
        }
    );
}

template<class T>
void gmgZeroKokkos(solvers::GmgFab<T>& mf)
{
    const auto a = mf.arrays();
    launchKokkosTeamNamed(
        "gmg_zero", mf, BENCH_LAMBDA(int ib, int i, int j, int k) { a[ib](i, j, k) = T(0); }
    );
}

// The six (field, coefficient) pairs Vcycle<KokkosOptGmgBackend, T, TC> is
// instantiated for (both here and in vcycle.hpp -- see PrecPair), plus kokkos_fused's
// (double, double), which is the same pair.
template void gmgGsColorKokkosFused<double, double>(
    solvers::GmgFab<double>&,
    const solvers::GmgFab<double>&,
    const solvers::GmgFab<double>&,
    const solvers::GmgFab<double>&,
    const solvers::GmgFab<double>&,
    const solvers::GmgFab<double>&,
    const solvers::GmgFab<double>&,
    const solvers::GmgFab<double>&,
    const solvers::GmgFab<double>&,
    int,
    double,
    bool
);
template void gmgGsColorKokkosFused<double, float>(
    solvers::GmgFab<double>&,
    const solvers::GmgFab<double>&,
    const solvers::GmgFab<float>&,
    const solvers::GmgFab<float>&,
    const solvers::GmgFab<float>&,
    const solvers::GmgFab<float>&,
    const solvers::GmgFab<float>&,
    const solvers::GmgFab<float>&,
    const solvers::GmgFab<float>&,
    int,
    double,
    bool
);
template void gmgGsColorKokkosFused<double, solvers::Bf16>(
    solvers::GmgFab<double>&,
    const solvers::GmgFab<double>&,
    const solvers::GmgFab<solvers::Bf16>&,
    const solvers::GmgFab<solvers::Bf16>&,
    const solvers::GmgFab<solvers::Bf16>&,
    const solvers::GmgFab<solvers::Bf16>&,
    const solvers::GmgFab<solvers::Bf16>&,
    const solvers::GmgFab<solvers::Bf16>&,
    const solvers::GmgFab<solvers::Bf16>&,
    int,
    double,
    bool
);
template void gmgGsColorKokkosFused<float, float>(
    solvers::GmgFab<float>&,
    const solvers::GmgFab<float>&,
    const solvers::GmgFab<float>&,
    const solvers::GmgFab<float>&,
    const solvers::GmgFab<float>&,
    const solvers::GmgFab<float>&,
    const solvers::GmgFab<float>&,
    const solvers::GmgFab<float>&,
    const solvers::GmgFab<float>&,
    int,
    double,
    bool
);
template void gmgGsColorKokkosFused<float, solvers::Bf16>(
    solvers::GmgFab<float>&,
    const solvers::GmgFab<float>&,
    const solvers::GmgFab<solvers::Bf16>&,
    const solvers::GmgFab<solvers::Bf16>&,
    const solvers::GmgFab<solvers::Bf16>&,
    const solvers::GmgFab<solvers::Bf16>&,
    const solvers::GmgFab<solvers::Bf16>&,
    const solvers::GmgFab<solvers::Bf16>&,
    const solvers::GmgFab<solvers::Bf16>&,
    int,
    double,
    bool
);
template void gmgGsColorKokkosFused<solvers::Bf16, solvers::Bf16>(
    solvers::GmgFab<solvers::Bf16>&,
    const solvers::GmgFab<solvers::Bf16>&,
    const solvers::GmgFab<solvers::Bf16>&,
    const solvers::GmgFab<solvers::Bf16>&,
    const solvers::GmgFab<solvers::Bf16>&,
    const solvers::GmgFab<solvers::Bf16>&,
    const solvers::GmgFab<solvers::Bf16>&,
    const solvers::GmgFab<solvers::Bf16>&,
    const solvers::GmgFab<solvers::Bf16>&,
    int,
    double,
    bool
);

template void gmgResidRestrictKokkosFused<double, double>(
    const solvers::GmgFab<double>&,
    const solvers::GmgFab<double>&,
    solvers::GmgFab<double>&,
    const solvers::GmgFab<double>&,
    const solvers::GmgFab<double>&,
    const solvers::GmgFab<double>&,
    const solvers::GmgFab<double>&,
    const solvers::GmgFab<double>&,
    const solvers::GmgFab<double>&,
    const solvers::GmgFab<double>&,
    bool
);
template void gmgResidRestrictKokkosFused<double, float>(
    const solvers::GmgFab<double>&,
    const solvers::GmgFab<double>&,
    solvers::GmgFab<double>&,
    const solvers::GmgFab<float>&,
    const solvers::GmgFab<float>&,
    const solvers::GmgFab<float>&,
    const solvers::GmgFab<float>&,
    const solvers::GmgFab<float>&,
    const solvers::GmgFab<float>&,
    const solvers::GmgFab<float>&,
    bool
);
template void gmgResidRestrictKokkosFused<double, solvers::Bf16>(
    const solvers::GmgFab<double>&,
    const solvers::GmgFab<double>&,
    solvers::GmgFab<double>&,
    const solvers::GmgFab<solvers::Bf16>&,
    const solvers::GmgFab<solvers::Bf16>&,
    const solvers::GmgFab<solvers::Bf16>&,
    const solvers::GmgFab<solvers::Bf16>&,
    const solvers::GmgFab<solvers::Bf16>&,
    const solvers::GmgFab<solvers::Bf16>&,
    const solvers::GmgFab<solvers::Bf16>&,
    bool
);
template void gmgResidRestrictKokkosFused<float, float>(
    const solvers::GmgFab<float>&,
    const solvers::GmgFab<float>&,
    solvers::GmgFab<float>&,
    const solvers::GmgFab<float>&,
    const solvers::GmgFab<float>&,
    const solvers::GmgFab<float>&,
    const solvers::GmgFab<float>&,
    const solvers::GmgFab<float>&,
    const solvers::GmgFab<float>&,
    const solvers::GmgFab<float>&,
    bool
);
template void gmgResidRestrictKokkosFused<float, solvers::Bf16>(
    const solvers::GmgFab<float>&,
    const solvers::GmgFab<float>&,
    solvers::GmgFab<float>&,
    const solvers::GmgFab<solvers::Bf16>&,
    const solvers::GmgFab<solvers::Bf16>&,
    const solvers::GmgFab<solvers::Bf16>&,
    const solvers::GmgFab<solvers::Bf16>&,
    const solvers::GmgFab<solvers::Bf16>&,
    const solvers::GmgFab<solvers::Bf16>&,
    const solvers::GmgFab<solvers::Bf16>&,
    bool
);
template void gmgResidRestrictKokkosFused<solvers::Bf16, solvers::Bf16>(
    const solvers::GmgFab<solvers::Bf16>&,
    const solvers::GmgFab<solvers::Bf16>&,
    solvers::GmgFab<solvers::Bf16>&,
    const solvers::GmgFab<solvers::Bf16>&,
    const solvers::GmgFab<solvers::Bf16>&,
    const solvers::GmgFab<solvers::Bf16>&,
    const solvers::GmgFab<solvers::Bf16>&,
    const solvers::GmgFab<solvers::Bf16>&,
    const solvers::GmgFab<solvers::Bf16>&,
    const solvers::GmgFab<solvers::Bf16>&,
    bool
);

// Field type only -- prolongation never touches the coefficients.
template void gmgProlongAddKokkosFused<double>(
    const solvers::GmgFab<double>&, solvers::GmgFab<double>&, bool
);
template void gmgProlongAddKokkosFused<float>(
    const solvers::GmgFab<float>&, solvers::GmgFab<float>&, bool
);
template void gmgProlongAddKokkosFused<solvers::Bf16>(
    const solvers::GmgFab<solvers::Bf16>&, solvers::GmgFab<solvers::Bf16>&, bool
);

// execCopyPlan/gmgZeroKokkos are templated on the FIELD type alone (the data
// movements they drive -- halo, zero fill, agglomeration transfer -- never touch the
// coefficients), so three instantiations cover every level the amrexFree_ path builds.
template void execCopyPlan<double>(
    const char*,
    const amrex::MultiArray4<double>&,
    const amrex::MultiArray4<const double>&,
    const CopyPlan&
);
template void execCopyPlan<float>(
    const char*,
    const amrex::MultiArray4<float>&,
    const amrex::MultiArray4<const float>&,
    const CopyPlan&
);
template void execCopyPlan<solvers::Bf16>(
    const char*,
    const amrex::MultiArray4<solvers::Bf16>&,
    const amrex::MultiArray4<const solvers::Bf16>&,
    const CopyPlan&
);

template void gmgZeroKokkos<double>(solvers::GmgFab<double>&);
template void gmgZeroKokkos<float>(solvers::GmgFab<float>&);
template void gmgZeroKokkos<solvers::Bf16>(solvers::GmgFab<solvers::Bf16>&);

} // namespace blockamr::bench
