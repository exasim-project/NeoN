// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>

#include <Kokkos_Core.hpp>

#include <AMReX_Array4.H>
#include <AMReX_Box.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_GpuDevice.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_MFParallelFor.H>
#include <AMReX_MultiFab.H>

// BLOCKAMR_LAMBDA, the one lambda form for all launchers, is defined there.
#include "NeoN/blockAmr/core/parallelAlgorithms.hpp"

// The launchers under comparison. PER-BOX, one launch per amrex::Box as NeoN's operators
// are written: launchAmrex (baseline), launchKokkosMd, launchKokkosFlat. FUSED, every
// valid cell in ONE launch: launchAmrexFused (the honest baseline), launchKokkosTeam.

namespace blockamr
{

template<class F>
void launchAmrex(const amrex::Box& bx, F const& f)
{
    amrex::ParallelFor(bx, f);
}

template<class F>
void launchKokkosMd(const amrex::Box& bx, F const& f)
{
    const auto lo = bx.smallEnd();
    const auto hi = bx.bigEnd();
    Kokkos::parallel_for(
        "bench_md",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
            {lo[0], lo[1], lo[2]}, {hi[0] + 1, hi[1] + 1, hi[2] + 1}
        ),
        f
    );
}

template<class F>
void launchKokkosFlat(const amrex::Box& bx, F const& f)
{
    const auto lo = bx.smallEnd();
    const int nx = bx.length(0);
    const int ny = bx.length(1);
    const long n = static_cast<long>(nx) * bx.length(1) * bx.length(2);
    const int lx = lo[0], ly = lo[1], lz = lo[2];
    Kokkos::parallel_for(
        "bench_flat",
        Kokkos::RangePolicy<>(0, n),
        KOKKOS_LAMBDA(const long t) {
            // i fastest, matching Array4's i-contiguous layout.
            const int i = static_cast<int>(t % nx);
            const int j = static_cast<int>((t / nx) % ny);
            const int k = static_cast<int>(t / (static_cast<long>(nx) * ny));
            f(i + lx, j + ly, k + lz);
        }
    );
}

// Streams: AMReX round-robins its box loop over numGpuStreams() so short per-box kernels
// overlap; a plain Kokkos::parallel_for issues to ONE -- the measured multi-box penalty.

// Heap-owned, not a function-local static: the instances own cudaStreams and must die
// BEFORE Kokkos::finalize and amrex::Finalize, not at static teardown afterwards.
inline std::vector<Kokkos::DefaultExecutionSpace>*& streamPoolPtr()
{
    static std::vector<Kokkos::DefaultExecutionSpace>* pool = nullptr;
    return pool;
}

inline const Kokkos::DefaultExecutionSpace& benchStream(int ibox)
{
    auto*& pool = streamPoolPtr();
    if (pool == nullptr)
    {
        // Exactly as many streams as AMReX uses, so the comparison is of the launcher.
        const int n = std::max(1, amrex::Gpu::Device::numGpuStreams());
        pool = new std::vector<Kokkos::DefaultExecutionSpace>(Kokkos::Experimental::partition_space(
            Kokkos::DefaultExecutionSpace {}, std::vector<int>(static_cast<std::size_t>(n), 1)
        ));
    }
    return (*pool)[static_cast<std::size_t>(ibox) % pool->size()];
}

inline void releaseStreamPool()
{
    delete streamPoolPtr();
    streamPoolPtr() = nullptr;
}

template<class F>
void launchKokkosMdStream(const amrex::Box& bx, int ibox, F const& f)
{
    const auto lo = bx.smallEnd();
    const auto hi = bx.bigEnd();
    Kokkos::parallel_for(
        "bench_md_stream",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
            benchStream(ibox), {lo[0], lo[1], lo[2]}, {hi[0] + 1, hi[1] + 1, hi[2] + 1}
        ),
        f
    );
}

// launchKokkosMd under a caller-chosen kernel name, so a profile can tell callers apart.
template<class F>
void launchKokkosMdNamed(const char* name, const amrex::Box& bx, F const& f)
{
    const auto lo = bx.smallEnd();
    const auto hi = bx.bigEnd();
    Kokkos::parallel_for(
        name,
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
            {lo[0], lo[1], lo[2]}, {hi[0] + 1, hi[1] + 1, hi[2] + 1}
        ),
        f
    );
}

// Fused launchers: one kernel for every box. f takes (ibox, i, j, k).

template<class F>
void launchAmrexFused(const amrex::MultiFab& mf, F const& f)
{
    amrex::ParallelFor(mf, f);
}

// The Kokkos twin of AMReX's fused path: MT, nblocks_per_box and the cached DEVICE
// BoxIndexer table are AMReX's own, so the comparison is one portable TeamPolicy against
// AMReX's per-backend launch. Valid cells only -- the MDRangePolicy fallback below rejects
// a negative lower bound.
template<class MF, class F>
void launchKokkosTeamNamed(const char* name, const MF& mf, F const& f)
{
    const int nboxes = static_cast<int>(mf.IndexArray().size());
    if (nboxes == 0)
    {
        return;
    }
    if (nboxes == 1)
    {
        // The same fallback AMReX makes: with one box there is nothing to fuse and the
        // team mapping costs, so the fused column would report a handicap AMReX does not.
        const amrex::Box bx = mf.box(mf.IndexArray()[0]);
        launchKokkosMdNamed(
            name, bx, BLOCKAMR_LAMBDA(int i, int j, int k) { f(0, i, j, k); }
        );
        return;
    }

    constexpr int MT = 128; // threads per block, == AMReX's CUDA MT
    constexpr int VL = 32;  // vector lanes = one warp, so blockDim is (32, 4, 1)

    const auto& info = mf.getParForInfo(amrex::IntVect(0));
    const int nblocksPerBox = info.getNBlocksPerBox(MT);
    const amrex::BoxIndexer* boxes = info.getBoxes();

    using Policy = Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>;
    Kokkos::parallel_for(
        name,
        Policy(nboxes * nblocksPerBox, MT / VL, VL),
        KOKKOS_LAMBDA(const Policy::member_type& team) {
            const int blk = team.league_rank();
            const int ibox = blk / nblocksPerBox;
            const auto base = static_cast<std::uint64_t>(blk - ibox * nblocksPerBox) * MT;
            const amrex::BoxIndexer& indexer = boxes[ibox];
            Kokkos::parallel_for(
                Kokkos::TeamVectorRange(team, MT),
                [&](const int t)
                {
                    const std::uint64_t icell = base + static_cast<std::uint64_t>(t);
                    if (icell < indexer.numPts())
                    {
                        const auto ijk = indexer(icell);
                        f(ibox, ijk.x, ijk.y, ijk.z);
                    }
                }
            );
        }
    );
}

template<class MF, class F>
void launchKokkosTeam(const MF& mf, F const& f)
{
    launchKokkosTeamNamed("bench_team", mf, f);
}

// Accessors: both take GLOBAL (i, j, k). amrex::Array4 already does; ViewAcc gives an
// unmanaged Kokkos View the same convention by subtracting the fab box origin.

using FabView = Kokkos::View<
    double***,
    Kokkos::LayoutLeft,
    Kokkos::DefaultExecutionSpace::memory_space,
    Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

struct ViewAcc
{
    FabView v;
    int ox, oy, oz;

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE double& operator()(int i, int j, int k) const
    {
        return v(i - ox, j - oy, k - oz);
    }
};

// Unmanaged View over the whole fab box, origin recorded so the accessor is global.
inline ViewAcc viewAcc(amrex::FArrayBox& fab)
{
    const amrex::Box& b = fab.box();
    return ViewAcc {
        FabView(fab.dataPtr(), b.length(0), b.length(1), b.length(2)),
        b.smallEnd(0),
        b.smallEnd(1),
        b.smallEnd(2)
    };
}

} // namespace blockamr
