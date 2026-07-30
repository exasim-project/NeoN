// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <AMReX_Arena.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_MultiFab.H>

#include <cstddef>
#include <vector>

namespace blockamr::la
{

// Cells this rank owns -- the length of the flat vectors below. NEVER
// boxArray().numPts(), which counts EVERY rank's cells: that was the long-standing
// multi-rank bug. Counted by the same MFIter walk gather/scatter use, so it agrees.
template<class FA>
inline std::size_t localCount(const FA& mf)
{
    std::size_t n = 0;
    for (amrex::MFIter mfi(mf); mfi.isValid(); ++mfi)
    {
        n += static_cast<std::size_t>(mfi.validbox().numPts());
    }
    return n;
}

// Flat-vector <-> MultiFab transfer (component 0, valid cells only). INVARIANT: gather and
// scatter MUST traverse in the identical order -- MFIter without tiling, then k,j,i over the
// valid box. `scale` lets gather apply the SPD sign flip (-L) in one pass.
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

// Device pack/unpack between a Ginkgo vector and a device-resident MultiFab, so the mat-vec
// never round-trips the host; same flat index as gather/scatter above. nvcc TRAP:
// declaration-only, or the lambda is extended across three CUDA TUs; deviceKernels.cpp.
template<class V, class FA>
void scatter_device(const V* vec, FA& mf);

template<class V, class FA>
void gather_device(const FA& mf, V* vec, double scale);

} // namespace blockamr::la
