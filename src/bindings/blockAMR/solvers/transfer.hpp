// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <AMReX_Arena.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_MultiFab.H>

#include <cstddef>
#include <vector>

namespace blockamr::solvers
{

// Flat-vector <-> MultiFab transfer (component 0, valid cells only).
// gather and scatter MUST traverse cells in the identical order: MFIter
// without tiling, then k,j,i over the valid box. MultiFabs live in device
// memory by default in GPU builds, so access is staged through explicit
// host copies unless the arena is host-accessible. `scale` lets gather
// apply the SPD sign flip (-L) in the same pass.
// Templated on the FabArray type so the same host path serves the FP64
// MultiFab (Ginkgo double vector) and the FP32 GMG level fields
// (FabArray<BaseFab<float>>), and on the flat buffer's value type V so it
// compiles for the fp32 Krylov instantiation as well -- that path is device-only
// and its constructor says so, but the host branch still has to be valid code.
template<class V, class FA>
void gather(const FA& mf, V* buf, double scale)
{
    using T = typename FA::value_type;
    const bool hostOk = mf.arena()->isHostAccessible();
    amrex::Gpu::streamSynchronize();
    std::size_t idx = 0;
    for (amrex::MFIter mfi(mf); mfi.isValid(); ++mfi)
    {
        const amrex::Box& vbx = mfi.validbox();
        const auto& fab = mf[mfi];
        const amrex::Box& fbx = fab.box();
        std::vector<T> stage;
        auto arr = fab.const_array();
        if (!hostOk)
        {
            // Component 0 occupies the first numPts() elements of the fab.
            stage.resize(static_cast<std::size_t>(fbx.numPts()));
            amrex::Gpu::dtoh_memcpy(stage.data(), fab.dataPtr(), stage.size() * sizeof(T));
            arr = amrex::makeArray4<const T>(stage.data(), fbx, 1);
        }
        const auto lo = amrex::lbound(vbx);
        const auto hi = amrex::ubound(vbx);
        for (int k = lo.z; k <= hi.z; ++k)
        {
            for (int j = lo.y; j <= hi.y; ++j)
            {
                for (int i = lo.x; i <= hi.x; ++i)
                {
                    buf[idx++] = static_cast<V>(scale * static_cast<double>(arr(i, j, k)));
                }
            }
        }
    }
}

template<class V, class FA>
void scatter(const V* buf, FA& mf)
{
    using T = typename FA::value_type;
    const bool hostOk = mf.arena()->isHostAccessible();
    amrex::Gpu::streamSynchronize();
    std::size_t idx = 0;
    for (amrex::MFIter mfi(mf); mfi.isValid(); ++mfi)
    {
        const amrex::Box& vbx = mfi.validbox();
        auto& fab = mf[mfi];
        const amrex::Box& fbx = fab.box();
        std::vector<T> stage;
        auto arr = fab.array();
        if (!hostOk)
        {
            // Round-trip the full fab so ghost values survive the update.
            stage.resize(static_cast<std::size_t>(fbx.numPts()));
            amrex::Gpu::dtoh_memcpy(stage.data(), fab.dataPtr(), stage.size() * sizeof(T));
            arr = amrex::makeArray4<T>(stage.data(), fbx, 1);
        }
        const auto lo = amrex::lbound(vbx);
        const auto hi = amrex::ubound(vbx);
        for (int k = lo.z; k <= hi.z; ++k)
        {
            for (int j = lo.y; j <= hi.y; ++j)
            {
                for (int i = lo.x; i <= hi.x; ++i)
                {
                    arr(i, j, k) = static_cast<T>(buf[idx++]);
                }
            }
        }
        if (!hostOk)
        {
            amrex::Gpu::htod_memcpy(fab.dataPtr(), stage.data(), stage.size() * sizeof(T));
        }
    }
}

// Device pack/unpack between a contiguous Ginkgo vector (device memory) and a
// device-resident MultiFab, via amrex::ParallelFor so the whole mat-vec runs
// on the GPU with NO host round-trip per Krylov iteration. The flat index MUST
// match the host gather/scatter above (MFIter order; within a valid box the
// index runs fastest in i, then j, then k), because the one-time RHS pack and
// solution unpack in the solve still use the host path.
// Templated on the FabArray type (see the host twins) AND on the flat vector's
// value type: the fab may be double (FP64 path) or float (FP32 GMG level), and
// the flat vector is double for the fp64 Krylov and float for the mixed-precision
// one, so the per-cell copy converts between the two on the device. V is deduced
// from the pointer, so every existing double call site is unchanged.
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

} // namespace blockamr::solvers
