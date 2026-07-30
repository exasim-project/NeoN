// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

// The bc.hpp/transfer.hpp device kernels reached from more than one CUDA TU: declaration-only in
// the header, defined here, and a missed instantiation is a null device function pointer at
// runtime, not a link error. Why: report/blockamr-linear-algebra-notes.md#the-nvcc-multi-tu-trap

#include "NeoN/blockAmr/core/bc.hpp"
#include "NeoN/blockAmr/core/parallelAlgorithms.hpp"
#include "NeoN/blockAmr/linearAlgebra/transfer.hpp"

#include "NeoN/blockAmr/linearAlgebra/gmg/gmgKernels.hpp"

namespace blockamr::la
{

// Fill the 1-cell domain-boundary ghost layer so the stencil folds homogeneous BCs with the matrix
// untouched: Dirichlet -> ghost = -interior, Neumann -> ghost = interior.
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

// The FP64 operator MultiFab plus the three level storage types the GMG hierarchy is built in.
template void
fillDomainBcGhostsDevice<amrex::MultiFab>(amrex::MultiFab&, const amrex::Box&, const BcArray&);
template void
fillDomainBcGhostsDevice<GmgFab<double>>(GmgFab<double>&, const amrex::Box&, const BcArray&);
template void
fillDomainBcGhostsDevice<GmgFab<float>>(GmgFab<float>&, const amrex::Box&, const BcArray&);
template void
fillDomainBcGhostsDevice<GmgFab<Bf16>>(GmgFab<Bf16>&, const amrex::Box&, const BcArray&);

// Inhomogeneous twin: ghost = sign*interior + scale*g, g read from the SAME ghost cell of bcdata.
void fillDomainBcGhostsInhomDevice(
    const NeoN::Executor& exec,
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
            blockamr::parallelFor(
                exec,
                f.gbx,
                [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                { a(i, j, k) = sign * a(i + di, j + dj, k + dk) + scale * g(i, j, k); }
            );
        }
    }
}

// Device pack/unpack between a contiguous Ginkgo vector and a MultiFab. The flat index MUST match
// transfer.hpp's host gather/scatter: MFIter order, then fastest in i, then j, then k.
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

// The flat vector is double (fp64 Krylov) or float (mixed-precision inner solve); the fab is the
// caller's MultiFab or a level fab. The MultiFab side is fp64-only: the mixed-precision path meets
// AMReX through the level fabs, never through the caller's fields.
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

} // namespace blockamr::la
