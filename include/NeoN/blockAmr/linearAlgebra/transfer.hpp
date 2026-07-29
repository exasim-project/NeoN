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

// Cells this rank owns -- the length of the flat vectors below and the local row
// count of the distributed Ginkgo vector over them.
//
// NEVER boxArray().numPts(), which counts EVERY rank's cells: sizing a flat vector
// globally while gather/scatter fill it from the rank's own boxes with a local
// offset was the long-standing multi-rank bug (each rank laid out a different
// vector over the same index range, so Ginkgo's dots and norms gave a different
// CG iteration per rank). Counted by the same MFIter walk gather/scatter use, so
// it agrees with them by construction rather than by a matching formula.
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

// Flat-vector <-> MultiFab transfer (component 0, valid cells only).
//
// INVARIANT: gather and scatter MUST traverse cells in the identical order --
// MFIter without tiling, then k,j,i over the valid box. In GPU builds MultiFabs
// are device-resident, so access is staged through host copies unless the arena is
// host-accessible. `scale` lets gather apply the SPD sign flip (-L) in one pass.
// Templated on the FabArray type (FP64 MultiFab, or FP32 GMG level fields) and on
// the flat buffer's type V, so it also compiles for the device-only fp32 Krylov
// instantiation.
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

// Device pack/unpack between a contiguous Ginkgo vector and a device-resident
// MultiFab, via amrex::ParallelFor, so the mat-vec runs entirely on the GPU with
// NO host round-trip per Krylov iteration.
//
// INVARIANT: the flat index MUST match the host gather/scatter above (MFIter
// order, fastest in i then j then k) -- the one-time RHS pack and solution unpack
// still use the host path. Templated on the fab type and on the flat vector's V,
// so the per-cell copy converts precisions on the device; V is deduced.
//
// nvcc TRAP: declaration-only here on purpose. These are reached from
// persistent.cpp and mlmgOps.cpp (via gmgPrecond.hpp / gmgBottom.hpp and the MLMG
// operators) AND from gmgKokkos/apply.cpp (via vcycle.hpp's applyFlat), all in the
// same _blockamr.so, so an AMREX_GPU_DEVICE lambda here would be an extended
// lambda instantiated in three CUDA TUs of one binary. The single definition plus
// its explicit instantiations live in core/deviceKernels.cpp.
template<class V, class FA>
void scatter_device(const V* vec, FA& mf);

template<class V, class FA>
void gather_device(const FA& mf, V* vec, double scale);

} // namespace blockamr::la
