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
// (FabArray<BaseFab<float>>): the flat Ginkgo buffer is always double, so the
// per-cell read/write converts to/from the fab's value_type.
template<class FA>
void gather(const FA& mf, double* buf, double scale)
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
                    buf[idx++] = scale * static_cast<double>(arr(i, j, k));
                }
            }
        }
    }
}

template<class FA>
void scatter(const double* buf, FA& mf)
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
// Templated on the FabArray type (see the host twins): the flat Ginkgo vector
// is double; the fab may be double (FP64 path) or float (FP32 GMG level), so the
// per-cell copy converts through the fab's value_type on the device.
template<class FA>
void scatter_device(const double* vec, FA& mf)
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

template<class FA>
void gather_device(const FA& mf, double* vec, double scale)
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
                vec[idx] = scale * static_cast<double>(a(i, j, k));
            }
        );
        off += vbx.numPts();
    }
}

} // namespace blockamr::solvers
