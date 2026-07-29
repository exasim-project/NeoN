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

// The launchers under comparison, in two families.
//
// PER-BOX -- an amrex::Box, f(i, j, k) over its GLOBAL range, one launch per box,
// which is how NeoN's operators are written today: launchAmrex (amrex::ParallelFor,
// the baseline), launchKokkosMd (MDRangePolicy, idiomatic tiled Kokkos) and
// launchKokkosFlat (RangePolicy + manual ijk, matching AMReX's own scheme and the
// only form NeoN::parallelFor can express today).
//
// FUSED -- the MultiFab, f(ibox, i, j, k) for every valid cell in ONE launch, so
// per-box launch cost cannot appear at all: launchAmrexFused (AMReX's own fused
// path, the honest baseline) and launchKokkosTeam (a TeamPolicy over the same
// decomposition, reading the same cached BoxIndexer table AMReX built).

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

// Streams. AMReX round-robins its box loop over numGpuStreams() CUDA streams, so
// short per-box kernels overlap; a plain Kokkos::parallel_for issues to ONE stream
// and serializes them -- the measured multi-box penalty. partition_space gives
// Kokkos the same width.

// Heap-owned rather than a function-local static: the instances own their
// cudaStreams and must be destroyed BEFORE Kokkos::finalize and amrex::Finalize,
// not at static teardown afterwards (releaseStreamPool() from kokkosFinalize()).
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
        // Exactly as many streams as AMReX uses, so the comparison is of the
        // launcher, not the stream count.
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

// launchKokkosMd under a caller-chosen kernel name, so a profile can tell the GMG
// kernels apart.
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

// The Kokkos twin of AMReX's fused path. Everything that is not the launch itself
// is shared with it -- MT, nblocks_per_box and the cached DEVICE BoxIndexer table
// are AMReX's own -- so the comparison is of one portable TeamPolicy against
// AMReX's hand-written per-backend launch.
//
// Templated on the FabArray, not fixed to MultiFab: everything it touches lives on
// FabArrayBase and the GMG levels are FabArray<BaseFab<T>>. `name` labels the
// kernel so a profile can tell callers apart.
//
// Valid cells only: a grown iteration space would need the MDRangePolicy fallback
// below to accept a negative lower bound, which its unsigned index type rejects.
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
        // The same fallback AMReX makes: with one box there is nothing to fuse and
        // the team mapping measurably costs, so without this the fused column would
        // report a handicap AMReX does not take at 1 box.
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

// Accessors: both take GLOBAL (i, j, k). amrex::Array4 already does and is
// trivially copyable into a Kokkos lambda; ViewAcc gives an unmanaged Kokkos View
// the same convention by subtracting the fab box origin.

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

// Unmanaged View over the whole fab box (ghosts included), origin recorded so the
// accessor speaks global indices like Array4.
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
