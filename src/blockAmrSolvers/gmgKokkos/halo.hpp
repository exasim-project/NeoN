// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include <Kokkos_Core.hpp>

#include <AMReX_BoxArray.H>
#include <AMReX_BoxList.H>
#include <AMReX_Periodicity.H>

#include "../common/bc.hpp"
#include "../gmg/gmg_kernels.hpp"
#include "launch.hpp"

// ---------------------------------------------------------------------------
// The data movements of the V-cycle, on Kokkos: the ghost exchange, the copy
// between two decompositions of the same region, and the zero fill.
//
// These are the operations kernels.hpp deliberately does NOT port, because they
// are not cell loops -- both the `kokkos` and `kokkos_fused` backends hand them to
// AMReX (FillBoundary / ParallelCopy / setVal). That choice has a cost the V-cycle
// bench made visible: every AMReX operation between two Kokkos kernels is a
// synchronisation point, since the two runtimes' streams are unordered. A colour
// sweep is `fence -> FillBoundary -> streamSynchronizeAll -> kernel`, so the host
// waits on the device TWICE per colour and nothing can overlap. Removing the
// per-kernel Kokkos::fence alone buys nothing: the FillBoundary that follows it
// needs the same ordering anyway.
//
// So the halo exchange is the pivot. With it on Kokkos the whole timed cycle is one
// stream of Kokkos kernels, correctly ordered with no host fence at all, and the
// host is free to run ahead of the device -- which is what the coarse levels, where
// launch cost dominates the arithmetic, actually need.
//
// The plan is AMReX's decomposition, resolved once. Building it needs the BoxArray
// adjacency and periodic images that FillBoundary works out internally, so it uses
// the same primitives (boxDiff for the ghost shell, BoxArray::intersections for who
// covers it, Periodicity::shiftIntVect for the images) -- at SETUP, once per level,
// untimed. What is left for the timed cycle is a flat device table of rectangular
// copies and one launch to execute it.
//
// Single rank only, deliberately: with more than one rank a halo exchange is MPI and
// the interesting part stops being the launch. Every box is local here, so every
// task is a device-to-device copy.
//
// That is a limit of the PLANS, not of the V-cycle they serve. A task names two LOCAL
// box indices, so a ghost cell covered by a box on another rank has no address to
// copy from and no task can be emitted for it -- which is why nothing here consults
// the rank count or falls back: on >1 rank these builders are simply not called, and
// the caller routes the same three movements through AMReX instead
// (vcycle.hpp, Vcycle::amrexFree_).
// ---------------------------------------------------------------------------

namespace blockamr::bench
{

// Threads per work block, and so the team size of the copy kernel.
constexpr int kCopyBlock = 128;

// One work block of one rectangular region copy, in GLOBAL cell indices:
//   dst[dstBox](i, j, k) = sign * src[srcBox](i + sh[0], j + sh[1], k + sh[2])
// over cells [base, base + kCopyBlock) of the region [lo, lo + len), counted i
// fastest. Box indices are LOCAL, matching the order of FabArray::arrays() and
// IndexArray(). `sh` is zero for a same-region copy and the negated periodic image
// shift for a wrapped ghost region.
//
// Regions are split into fixed-size blocks rather than mapped one-team-per-region
// because their sizes differ by orders of magnitude. A level of 512 boxes of 32^3 has
// thousands of ~1k-cell regions and either mapping saturates the GPU; a level of ONE
// box of 256^3 has 26 regions, six of them 65k cells, and one team per region would
// hand each of those to 128 threads. That is not a corner case -- it is every
// single-box level, which is what a hierarchy built by coarsening one box in place is
// made of.
struct CopyTask
{
    int dst;
    int src;
    int lo[3];
    int len[3];
    int sh[3];
    int base;
    // +1 for every data movement; -1 only for a reflect-odd (homogeneous Dirichlet)
    // domain ghost, which is a copy of the mirror cell with the sign flipped. Carrying
    // it here is what lets the boundary fill be the same kernel as the halo exchange.
    int sign;
};

// A whole exchange as one device table, so it executes in one launch.
struct CopyPlan
{
    Kokkos::View<CopyTask*> tasks;

    [[nodiscard]] int size() const { return static_cast<int>(tasks.extent(0)); }
};

namespace detail
{

inline CopyPlan toDevice(const std::vector<CopyTask>& host, const char* name)
{
    CopyPlan plan;
    // std::string, not the char*: view_alloc reads a raw pointer as memory to wrap.
    plan.tasks = Kokkos::View<CopyTask*>(
        Kokkos::view_alloc(std::string(name), Kokkos::WithoutInitializing), host.size()
    );
    auto mirror = Kokkos::create_mirror_view(plan.tasks);
    for (std::size_t t = 0; t < host.size(); ++t)
    {
        mirror(t) = host[t];
    }
    Kokkos::deep_copy(plan.tasks, mirror);
    return plan;
}

inline void addTask(
    std::vector<CopyTask>& out,
    int dst,
    int src,
    const amrex::Box& region,
    const amrex::IntVect& sh,
    int sign = 1
)
{
    CopyTask t {};
    t.dst = dst;
    t.src = src;
    t.sign = sign;
    for (int d = 0; d < 3; ++d)
    {
        t.lo[d] = region.smallEnd(d);
        t.len[d] = region.length(d);
        t.sh[d] = sh[d];
    }
    const int npts = t.len[0] * t.len[1] * t.len[2];
    for (int base = 0; base < npts; base += kCopyBlock)
    {
        t.base = base;
        out.push_back(t);
    }
}

} // namespace detail

// The ghost exchange of one FabArray: for every ghost cell of every box, the valid
// cell of the box -- or of the periodic image of a box -- that FillBoundary would
// copy it from.
//
// Ghost cells are enumerated as the shell boxDiff(grow(valid, ng), valid), which
// excludes the valid region itself, so no task ever copies a box onto itself. For
// each shell piece and each periodic image shift the covering source boxes come from
// the BoxArray's own hash, which is what makes the result the same partition of the
// shell that FillBoundary uses: the valid regions of all boxes and all their images
// tile space, so every ghost cell is covered exactly once and the order tasks run in
// cannot matter. Ghost cells outside a non-periodic domain get no task at all -- as
// in FillBoundary, which leaves physical ghosts to the boundary-condition code.
template<class FAB>
CopyPlan makeHaloPlan(const amrex::FabArray<FAB>& mf, const amrex::Periodicity& period)
{
    const amrex::BoxArray& ba = mf.boxArray();
    const amrex::IntVect ng = mf.nGrowVect();
    const std::vector<amrex::IntVect> shifts = period.shiftIntVect();

    std::vector<CopyTask> tasks;
    std::vector<std::pair<int, amrex::Box>> isects;
    for (int li = 0; li < mf.local_size(); ++li)
    {
        const amrex::Box valid = ba[mf.IndexArray()[li]];
        for (const amrex::Box& shell : amrex::boxDiff(amrex::grow(valid, ng), valid))
        {
            for (const amrex::IntVect& img : shifts)
            {
                // Cells of `shell` are covered by (source box + img), so shift the
                // query back into the source frame and the answer forward again.
                ba.intersections(amrex::shift(shell, -img), isects);
                for (const auto& is : isects)
                {
                    detail::addTask(
                        tasks, li, mf.localindex(is.first), amrex::shift(is.second, img), -img
                    );
                }
            }
        }
    }
    return detail::toDevice(tasks, "gmg_halo_plan");
}

// The homogeneous domain-boundary ghost fill, as a plan: for every valid box touching
// a non-periodic domain face, the one-cell ghost layer outside that face and the
// mirror interior cell to reflect into it (sign -1 for Dirichlet, +1 for Neumann).
//
// This is the twin of fillDomainBcGhosts* (solvers/bc.hpp) and it shares that path's
// geometry rather than restating it: solvers::bcGhostFill decides, per box and side,
// whether the side fires and what the layer, sign and offset are. So the two fills
// cannot drift apart, and the Kokkos one is testable against the AMReX one to the bit.
//
// Face layers only, as in production: the 7-point stencil never reads edge or corner
// ghosts, so nothing writes them. Runs AFTER the halo plan, which is the same order
// the production fillGhosts uses (FillBoundary first, then reflection) -- it matters
// on a box that touches a physical face in one direction and a periodic neighbour in
// another, where the reflection must see the already-exchanged interior values.
template<class FAB>
CopyPlan
makeBcPlan(const amrex::FabArray<FAB>& mf, const amrex::Box& domain, const solvers::BcArray& bc)
{
    const amrex::BoxArray& ba = mf.boxArray();
    std::vector<CopyTask> tasks;
    for (int li = 0; li < mf.local_size(); ++li)
    {
        const amrex::Box valid = ba[mf.IndexArray()[li]];
        for (int s = 0; s < 6; ++s)
        {
            solvers::BcGhostFill f;
            if (!solvers::bcGhostFill(valid, domain, bc, s, f))
            {
                continue;
            }
            // Same box on both sides: a ghost layer of a box is filled from that
            // box's own interior, so dst and src are the one local index.
            detail::addTask(
                tasks, li, li, f.gbx, amrex::IntVect(f.di, f.dj, f.dk), (f.sign < 0.0) ? -1 : 1
            );
        }
    }
    return detail::toDevice(tasks, "gmg_bc_plan");
}

// The valid-to-valid copy between two FabArrays over the same region on different
// decompositions -- what ParallelCopy does, and what an agglomerated level needs in
// both directions. Cell-centred only: face-centred BoxArrays share their internal
// faces, so a face cell can have more than one source and the plan would then depend
// on task order (harmless for the coefficients, which are copied once at setup and
// stay with AMReX).
template<class FAB>
CopyPlan makeCopyPlan(const amrex::FabArray<FAB>& dst, const amrex::FabArray<FAB>& src)
{
    std::vector<CopyTask> tasks;
    std::vector<std::pair<int, amrex::Box>> isects;
    for (int li = 0; li < dst.local_size(); ++li)
    {
        src.boxArray().intersections(dst.boxArray()[dst.IndexArray()[li]], isects);
        for (const auto& is : isects)
        {
            detail::addTask(tasks, li, src.localindex(is.first), is.second, amrex::IntVect(0));
        }
    }
    return detail::toDevice(tasks, "gmg_copy_plan");
}

// Declared here, DEFINED (and explicitly instantiated for every field type T the
// V-cycle needs) in kernels.cpp, NOT header-inline: this is the launcher
// KokkosOptGmgBackend's amrexFree_ path (vcycle.hpp) drives from BOTH
// apply.cpp and bench/gmg_vcycle_bench.cpp, and instantiating a function
// with an extended __host__ __device__ lambda identically in two CUDA TUs that feed
// the same shared object is the nvcc trap kernels.hpp's Fused kernels document
// above their own (now equally out-of-line) declarations -- a null function-pointer
// call at runtime, not a build failure. See kernels.cpp.
//
// Execute a plan in ONE launch: one team per work block. A block short of a full
// kCopyBlock (the tail of a region, or a whole corner region of one cell) leaves lanes
// idle, which is cheaper than any mapping that would even it out.
template<class T>
void execCopyPlan(
    const char* name,
    const amrex::MultiArray4<T>& dst,
    const amrex::MultiArray4<const T>& src,
    const CopyPlan& plan
);

// Twin of FillBoundary(periodicity): fills this fab's ghosts from its own valid data.
// Reads are valid cells and writes are ghost cells, disjoint, so the whole exchange
// is safe in a single kernel with no intermediate buffer.
template<class T>
void gmgFillBoundaryKokkos(solvers::GmgFab<T>& mf, const CopyPlan& plan)
{
    execCopyPlan("gmg_halo", mf.arrays(), mf.const_arrays(), plan);
}

// Twin of fillDomainBcGhostsDevice(mf, domain, bc), from a plan built by makeBcPlan.
// Reads interior cells and writes ghost cells, disjoint, so one kernel is safe.
template<class T>
void gmgFillDomainBcKokkos(solvers::GmgFab<T>& mf, const CopyPlan& plan)
{
    execCopyPlan("gmg_bc", mf.arrays(), mf.const_arrays(), plan);
}

// Twin of dst.ParallelCopy(src, 0, 0, 1).
template<class T>
void gmgCopyKokkos(solvers::GmgFab<T>& dst, const solvers::GmgFab<T>& src, const CopyPlan& plan)
{
    execCopyPlan("gmg_copy", dst.arrays(), src.const_arrays(), plan);
}

// Stands in for setVal(0) on a coarse solution, in one launch for all boxes.
//
// Valid cells only, where setVal also clears the ghosts -- equivalent here, and only
// here: the V-cycle reads a coarse solution's ghosts nowhere except inside a colour
// sweep, and every colour sweep is preceded by a full ghost exchange (smooth() calls
// fillGhosts first, and the plan above covers the entire ghost shell of a periodic
// domain). Prolongation reads valid cells only. Give this file a non-periodic domain
// -- which it does not have, physical BCs being out of its scope -- and the physical
// ghosts would have to be cleared here as well.
//
// Declared here, defined out-of-line in kernels.cpp, same reason as
// execCopyPlan above: it launches its own extended lambda directly and is driven
// from both apply.cpp and bench/gmg_vcycle_bench.cpp.
template<class T>
void gmgZeroKokkos(solvers::GmgFab<T>& mf);

} // namespace blockamr::bench
