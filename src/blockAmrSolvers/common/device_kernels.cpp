// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

// Definitions -- and, below, the explicit instantiations -- of the bc.hpp and
// transfer.hpp device kernels that are reached from more than one .cpp
// translation unit (Class B in the T9 report): declaring them as ordinary header
// templates (or, for fillDomainBcGhostsInhomDevice, as an inline function) and
// giving each an AMREX_GPU_DEVICE lambda would make them extended lambdas
// instantiated in two or three CUDA TUs of the same final _blockamr.so
// (persistent.cpp / face_coeff_op.cpp / mlmg_ops.cpp via blockamr_solvers,
// gmgKokkos/apply.cpp and bench/gmg_vcycle_bench.cpp via blockamr_kokkos) -- the
// exact nvcc trap T2 already hit for the fused Kokkos kernels (see
// gmgKokkos/kernels.cpp). The fix is the same one gmg/gmg_kernels.cpp uses:
// define the kernel in exactly one TU and explicitly instantiate every (V, FA)
// this TU's callers need, so every other including TU sees only the declaration
// and links against this single definition.
//
// Lives in blockamr_kokkos rather than next to bc.cpp in blockamr_solvers
// because apply.cpp and gmg_vcycle_bench.cpp need these symbols in a build
// WITHOUT Ginkgo, where blockamr_solvers does not exist at all
// (blockAmrSolvers/CMakeLists.txt). gmg_kernels.hpp is included for GmgFab and
// Bf16, the level fab types the instantiation lists are spelled in.

#include "bc.hpp"
#include "transfer.hpp"

#include "../gmg/gmg_kernels.hpp"

namespace blockamr::solvers
{

// Fill the domain-boundary ghost layer of `mf` (1 ghost, component 0) so the
// face-coefficient stencil folds homogeneous BCs with the matrix untouched:
// Dirichlet -> ghost = -interior, Neumann -> ghost = interior.
template<class FA>
void fillDomainBcGhostsDevice(FA& mf, const amrex::Box& domain, const BcArray& bc)
{
    using T = typename FA::value_type;
    for (amrex::MFIter mfi(mf); mfi.isValid(); ++mfi)
    {
        const amrex::Box& vbx = mfi.validbox();
        const auto a = mf.array(mfi);
        for (int s = 0; s < 6; ++s)
        {
            BcGhostFill f;
            if (!bcGhostFill(vbx, domain, bc, s, f))
            {
                continue;
            }
            const T sign = static_cast<T>(f.sign);
            const int di = f.di, dj = f.dj, dk = f.dk;
            amrex::ParallelFor(
                f.gbx,
                [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                { a(i, j, k) = sign * a(i + di, j + dj, k + dk); }
            );
        }
    }
}

// The FP64 operator MultiFab (face_coeff_op.cpp, persistent.cpp) and the three
// level storage types the GMG hierarchy is built in (persistent.cpp via
// gmg_precond.hpp, apply.cpp and gmg_vcycle_bench.cpp via vcycle.hpp).
template void
fillDomainBcGhostsDevice<amrex::MultiFab>(amrex::MultiFab&, const amrex::Box&, const BcArray&);
template void
fillDomainBcGhostsDevice<GmgFab<double>>(GmgFab<double>&, const amrex::Box&, const BcArray&);
template void
fillDomainBcGhostsDevice<GmgFab<float>>(GmgFab<float>&, const amrex::Box&, const BcArray&);
template void
fillDomainBcGhostsDevice<GmgFab<Bf16>>(GmgFab<Bf16>&, const amrex::Box&, const BcArray&);

// Inhomogeneous twin: ghost = sign*interior + scale*g, with g read from the SAME
// ghost cell of `bcdata`.
void fillDomainBcGhostsInhomDevice(
    amrex::MultiFab& mf,
    const amrex::MultiFab& bcdata,
    const amrex::Box& domain,
    const BcArray& bc,
    const amrex::Real* dx
)
{
    for (amrex::MFIter mfi(mf); mfi.isValid(); ++mfi)
    {
        const amrex::Box& vbx = mfi.validbox();
        const auto a = mf.array(mfi);
        const auto g = bcdata.const_array(mfi);
        for (int s = 0; s < 6; ++s)
        {
            BcGhostFill f;
            if (!bcGhostFill(vbx, domain, bc, s, f))
            {
                continue;
            }
            const amrex::Real sign = static_cast<amrex::Real>(f.sign);
            const amrex::Real scale =
                (bc[static_cast<std::size_t>(s)] == 1) ? amrex::Real(2.0) : dx[s / 2];
            const int di = f.di, dj = f.dj, dk = f.dk;
            amrex::ParallelFor(
                f.gbx,
                [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                { a(i, j, k) = sign * a(i + di, j + dj, k + dk) + scale * g(i, j, k); }
            );
        }
    }
}

// Device pack/unpack between a contiguous Ginkgo vector (device memory) and a
// device-resident MultiFab. The flat index MUST match the host gather/scatter of
// transfer.hpp: MFIter order; within a valid box the index runs fastest in i,
// then j, then k.
template<class V, class FA>
void scatter_device(const V* vec, FA& mf)
{
    using T = typename FA::value_type;
    long off = 0;
    for (amrex::MFIter mfi(mf); mfi.isValid(); ++mfi)
    {
        const amrex::Box& vbx = mfi.validbox();
        const auto a = mf.array(mfi);
        const auto lo = amrex::lbound(vbx);
        const int ni = vbx.length(0);
        const int nj = vbx.length(1);
        const long o = off;
        amrex::ParallelFor(
            vbx,
            [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
            {
                const long idx =
                    o + (static_cast<long>(k - lo.z) * nj + (j - lo.y)) * ni + (i - lo.x);
                a(i, j, k) = static_cast<T>(vec[idx]);
            }
        );
        off += vbx.numPts();
    }
}

template<class V, class FA>
void gather_device(const FA& mf, V* vec, double scale)
{
    long off = 0;
    for (amrex::MFIter mfi(mf); mfi.isValid(); ++mfi)
    {
        const amrex::Box& vbx = mfi.validbox();
        const auto a = mf.const_array(mfi);
        const auto lo = amrex::lbound(vbx);
        const int ni = vbx.length(0);
        const int nj = vbx.length(1);
        const long o = off;
        amrex::ParallelFor(
            vbx,
            [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
            {
                const long idx =
                    o + (static_cast<long>(k - lo.z) * nj + (j - lo.y)) * ni + (i - lo.x);
                vec[idx] = static_cast<V>(scale * static_cast<double>(a(i, j, k)));
            }
        );
        off += vbx.numPts();
    }
}

// The flat vector is double for the fp64 Krylov and float for the
// mixed-precision inner solve; the fab is the caller's FP64 MultiFab (mlmg_ops,
// persistent) or a level fab in any of the three storage types (persistent via
// gmg_precond/gmg_bottom, apply.cpp via vcycle's applyFlat<V>). The MultiFab
// side is fp64-only: the mixed-precision path meets AMReX through the level
// fabs, never through the caller's fields.
template void scatter_device<double, amrex::MultiFab>(const double*, amrex::MultiFab&);
template void scatter_device<double, GmgFab<double>>(const double*, GmgFab<double>&);
template void scatter_device<double, GmgFab<float>>(const double*, GmgFab<float>&);
template void scatter_device<double, GmgFab<Bf16>>(const double*, GmgFab<Bf16>&);
template void scatter_device<float, GmgFab<double>>(const float*, GmgFab<double>&);
template void scatter_device<float, GmgFab<float>>(const float*, GmgFab<float>&);
template void scatter_device<float, GmgFab<Bf16>>(const float*, GmgFab<Bf16>&);

template void gather_device<double, amrex::MultiFab>(const amrex::MultiFab&, double*, double);
template void gather_device<double, GmgFab<double>>(const GmgFab<double>&, double*, double);
template void gather_device<double, GmgFab<float>>(const GmgFab<float>&, double*, double);
template void gather_device<double, GmgFab<Bf16>>(const GmgFab<Bf16>&, double*, double);
template void gather_device<float, GmgFab<double>>(const GmgFab<double>&, float*, double);
template void gather_device<float, GmgFab<float>>(const GmgFab<float>&, float*, double);
template void gather_device<float, GmgFab<Bf16>>(const GmgFab<Bf16>&, float*, double);

} // namespace blockamr::solvers
