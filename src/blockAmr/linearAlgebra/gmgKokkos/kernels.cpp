// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

// The five kernels.hpp/halo.hpp launchers that open an extended device lambda and are reached from
// both apply.cpp and bench/gmgVcycleBench.cpp: nvcc needs such a kernel declaration-only in the
// header, defined here. Why: report/blockamr-linear-algebra-notes.md#the-nvcc-multi-tu-trap

#include "NeoN/blockAmr/linearAlgebra/gmg/bf16.hpp"
#include "NeoN/blockAmr/linearAlgebra/gmg/gmgKernels.hpp"
#include "NeoN/blockAmr/linearAlgebra/gmgKokkos/halo.hpp"
#include "NeoN/blockAmr/linearAlgebra/gmgKokkos/kernels.hpp"

namespace blockamr
{

template<class T, class TC>
void gmgGsColorKokkosFused(
    la::GmgFab<T>& sol,
    const la::GmgFab<T>& rhs,
    const la::FaceCoeffs<TC>& fc,
    la::GsSweep sweep,
    bool fence
)
{
    const la::GmgComputeT<T> om = static_cast<la::GmgComputeT<T>>(sweep.omega);
    const int parity = sweep.parity;
    const auto psi = sol.arrays();
    const auto b = rhs.const_arrays();
    const auto ax = fc.ux->const_arrays();
    const auto lxa = fc.lx->const_arrays();
    const auto ay = fc.uy->const_arrays();
    const auto lya = fc.ly->const_arrays();
    const auto az = fc.uz->const_arrays();
    const auto lza = fc.lz->const_arrays();
    const auto al = fc.alpha->const_arrays();
    launchKokkosTeamNamed(
        "gmg_gs_fused",
        rhs,
        BLOCKAMR_LAMBDA(int ib, int i, int j, int k) {
            const la::FaceCoeffArrays<TC> faces {ax[ib], lxa[ib], ay[ib], lya[ib], az[ib], lza[ib]};
            GmgGsCell<T, TC> {psi[ib], b[ib], faces, al[ib], om, parity}(i, j, k);
        }
    );
    if (fence)
    {
        Kokkos::fence();
    }
}

template<class T, class TC>
void gmgResidRestrictKokkosFused(
    const la::GmgFab<T>& sol,
    const la::GmgFab<T>& rhs,
    la::GmgFab<T>& crhs,
    const la::FaceCoeffs<TC>& fc,
    bool fence
)
{
    const auto cr = crhs.arrays();
    const auto psi = sol.const_arrays();
    const auto b = rhs.const_arrays();
    const auto ax = fc.ux->const_arrays();
    const auto lxa = fc.lx->const_arrays();
    const auto ay = fc.uy->const_arrays();
    const auto lya = fc.ly->const_arrays();
    const auto az = fc.uz->const_arrays();
    const auto lza = fc.lz->const_arrays();
    const auto al = fc.alpha->const_arrays();
    launchKokkosTeamNamed(
        "gmg_residrestrict_fused",
        crhs,
        BLOCKAMR_LAMBDA(int ib, int ic, int jc, int kc) {
            const la::FaceCoeffArrays<TC> faces {ax[ib], lxa[ib], ay[ib], lya[ib], az[ib], lza[ib]};
            GmgResidRestrictCell<T, TC> {cr[ib], psi[ib], b[ib], faces, al[ib]}(ic, jc, kc);
        }
    );
    if (fence)
    {
        Kokkos::fence();
    }
}

template<class T>
void gmgProlongAddKokkosFused(const la::GmgFab<T>& crse, la::GmgFab<T>& fine, bool fence)
{
    const auto f = fine.arrays();
    const auto c = crse.const_arrays();
    launchKokkosTeamNamed(
        "gmg_prolong_fused",
        fine,
        BLOCKAMR_LAMBDA(int ib, int i, int j, int k) {
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
    // Function-local: a namespace-scope constexpr has no device symbol for the kernel body.
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
                    // The compute type, not T: on a bf16 level the ternary's arms Bf16 and
                    // float are mutually convertible, hence ambiguous. Load and sign flip
                    // are exact in it, so the copy still moves values unchanged.
                    const la::GmgComputeT<T> v = src[t.src](i + t.sh[0], j + t.sh[1], k + t.sh[2]);
                    dst[t.dst](i, j, k) = (t.sign < 0) ? -v : v;
                }
            );
        }
    );
}

template<class T>
void gmgZeroKokkos(la::GmgFab<T>& mf)
{
    const auto a = mf.arrays();
    launchKokkosTeamNamed(
        "gmg_zero", mf, BLOCKAMR_LAMBDA(int ib, int i, int j, int k) { a[ib](i, j, k) = T(0); }
    );
}

// The six (field, coefficient) pairs Vcycle<KokkosOptGmgBackend, T, TC> is instantiated for.
// A missing instantiation is a null device function pointer at runtime, not a link error.
template void gmgGsColorKokkosFused<double, double>(
    la::GmgFab<double>&, const la::GmgFab<double>&, const la::FaceCoeffs<double>&, la::GsSweep, bool
);
template void gmgGsColorKokkosFused<double, float>(
    la::GmgFab<double>&, const la::GmgFab<double>&, const la::FaceCoeffs<float>&, la::GsSweep, bool
);
template void gmgGsColorKokkosFused<double, la::Bf16>(
    la::GmgFab<double>&,
    const la::GmgFab<double>&,
    const la::FaceCoeffs<la::Bf16>&,
    la::GsSweep,
    bool
);
template void gmgGsColorKokkosFused<float, float>(
    la::GmgFab<float>&, const la::GmgFab<float>&, const la::FaceCoeffs<float>&, la::GsSweep, bool
);
template void gmgGsColorKokkosFused<float, la::Bf16>(
    la::GmgFab<float>&, const la::GmgFab<float>&, const la::FaceCoeffs<la::Bf16>&, la::GsSweep, bool
);
template void gmgGsColorKokkosFused<la::Bf16, la::Bf16>(
    la::GmgFab<la::Bf16>&,
    const la::GmgFab<la::Bf16>&,
    const la::FaceCoeffs<la::Bf16>&,
    la::GsSweep,
    bool
);

template void gmgResidRestrictKokkosFused<double, double>(
    const la::GmgFab<double>&,
    const la::GmgFab<double>&,
    la::GmgFab<double>&,
    const la::FaceCoeffs<double>&,
    bool
);
template void gmgResidRestrictKokkosFused<double, float>(
    const la::GmgFab<double>&,
    const la::GmgFab<double>&,
    la::GmgFab<double>&,
    const la::FaceCoeffs<float>&,
    bool
);
template void gmgResidRestrictKokkosFused<double, la::Bf16>(
    const la::GmgFab<double>&,
    const la::GmgFab<double>&,
    la::GmgFab<double>&,
    const la::FaceCoeffs<la::Bf16>&,
    bool
);
template void gmgResidRestrictKokkosFused<float, float>(
    const la::GmgFab<float>&,
    const la::GmgFab<float>&,
    la::GmgFab<float>&,
    const la::FaceCoeffs<float>&,
    bool
);
template void gmgResidRestrictKokkosFused<float, la::Bf16>(
    const la::GmgFab<float>&,
    const la::GmgFab<float>&,
    la::GmgFab<float>&,
    const la::FaceCoeffs<la::Bf16>&,
    bool
);
template void gmgResidRestrictKokkosFused<la::Bf16, la::Bf16>(
    const la::GmgFab<la::Bf16>&,
    const la::GmgFab<la::Bf16>&,
    la::GmgFab<la::Bf16>&,
    const la::FaceCoeffs<la::Bf16>&,
    bool
);

// Field type only -- prolongation never touches the coefficients.
template void
gmgProlongAddKokkosFused<double>(const la::GmgFab<double>&, la::GmgFab<double>&, bool);
template void gmgProlongAddKokkosFused<float>(const la::GmgFab<float>&, la::GmgFab<float>&, bool);
template void
gmgProlongAddKokkosFused<la::Bf16>(const la::GmgFab<la::Bf16>&, la::GmgFab<la::Bf16>&, bool);

// Field type alone: halo, zero fill and agglomeration transfer never touch the coefficients, so
// three instantiations cover every level the amrexFree_ path builds.
template void execCopyPlan<
    double>(const char*, const amrex::MultiArray4<double>&, const amrex::MultiArray4<const double>&, const CopyPlan&);
template void execCopyPlan<
    float>(const char*, const amrex::MultiArray4<float>&, const amrex::MultiArray4<const float>&, const CopyPlan&);
template void execCopyPlan<
    la::Bf16>(const char*, const amrex::MultiArray4<la::Bf16>&, const amrex::MultiArray4<const la::Bf16>&, const CopyPlan&);

template void gmgZeroKokkos<double>(la::GmgFab<double>&);
template void gmgZeroKokkos<float>(la::GmgFab<float>&);
template void gmgZeroKokkos<la::Bf16>(la::GmgFab<la::Bf16>&);

} // namespace blockamr
