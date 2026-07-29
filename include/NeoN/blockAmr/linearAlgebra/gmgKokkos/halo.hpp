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

#include "NeoN/blockAmr/core/bc.hpp"
#include "NeoN/blockAmr/linearAlgebra/gmg/gmgKernels.hpp"
#include "NeoN/blockAmr/core/launch.hpp"

// The V-cycle's data movements on Kokkos: the ghost exchange, the copy between two
// decompositions of one region, and the zero fill.
//
// kernels.hpp leaves these to AMReX because they are not cell loops, and that costs a
// synchronisation point per operation, the two runtimes' streams being unordered: a
// colour sweep becomes `fence -> FillBoundary -> streamSynchronizeAll -> kernel`, so
// the host waits on the device twice per colour. Dropping the Kokkos fence alone buys
// nothing -- the FillBoundary needs the same ordering. The halo exchange is therefore
// the pivot: with it on Kokkos the whole timed cycle is one correctly ordered Kokkos
// stream with no host fence, and the host can run ahead of the device, which is what
// the launch-bound coarse levels need.
//
// A plan is AMReX's decomposition resolved once, built from the same primitives
// FillBoundary uses internally (boxDiff for the ghost shell, BoxArray::intersections
// for who covers it, Periodicity::shiftIntVect for the images) at SETUP, untimed. The
// timed cycle sees a flat device table of rectangular copies and one launch.
//
// Single rank only, and that is a limit of the PLANS: a task names two LOCAL box
// indices, so a ghost covered by a box on another rank has no address to copy from.
// Nothing here consults the rank count -- on >1 rank these builders are simply not
// called and the caller routes the same three movements through AMReX (vcycle.hpp,
// Vcycle::amrexFree_).

namespace blockamr
{

// Threads per work block, and so the team size of the copy kernel.
constexpr int kCopyBlock = 128;

// One work block of one rectangular region copy, in GLOBAL cell indices:
//   dst[dstBox](i, j, k) = sign * src[srcBox](i + sh[0], j + sh[1], k + sh[2])
// over cells [base, base + kCopyBlock) of the region [lo, lo + len), i fastest. Box
// indices are LOCAL, matching FabArray::arrays() and IndexArray(); `sh` is zero for a
// same-region copy and the negated periodic image shift for a wrapped ghost region.
//
// Fixed-size blocks rather than one team per region, because region sizes differ by
// orders of magnitude: a single-box 256^3 level has 26 regions, six of them 65k cells,
// and one team per region would hand each of those to 128 threads. Not a corner case --
// every level of a hierarchy coarsened in place from one box looks like that.
struct CopyTask
{
    int dst;
    int src;
    int lo[3];
    int len[3];
    int sh[3];
    int base;
    // -1 only for a reflect-odd (homogeneous Dirichlet) domain ghost, a mirror copy
    // with the sign flipped; carrying it here makes the boundary fill the halo's own
    // kernel.
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

// The ghost exchange of one FabArray: for every ghost cell, the valid cell -- of a box
// or of a periodic image of one -- that FillBoundary would copy it from.
//
// Ghosts are enumerated as the shell boxDiff(grow(valid, ng), valid), which excludes
// the valid region, so no task copies a box onto itself; the covering sources come from
// the BoxArray's own hash, which is what makes this the same partition of the shell
// FillBoundary uses -- valid regions and their images tile space, so every ghost is
// covered exactly once and task order cannot matter. Ghosts outside a non-periodic
// domain get no task, as in FillBoundary, which leaves those to the boundary-condition
// code.
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
                // Cells of `shell` are covered by (source box + img): shift the query
                // back into the source frame and the answer forward again.
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

// The homogeneous domain-boundary ghost fill as a plan: for every valid box touching a
// non-periodic domain face, the one-cell ghost layer outside it and the mirror interior
// cell to reflect into it (sign -1 for Dirichlet, +1 for Neumann).
//
// Twin of fillDomainBcGhosts*, sharing that path's geometry rather than restating it
// (la::bcGhostFill decides per box and side whether it fires and what the layer, sign
// and offset are), so the two cannot drift apart and this one is testable to the bit.
//
// Face layers only, as in production: the 7-point stencil never reads edge or corner
// ghosts. Runs AFTER the halo plan, production's order -- it matters on a box that is
// physical in one direction and periodic in another, where the reflection must see the
// already-exchanged interior values.
template<class FAB>
CopyPlan makeBcPlan(const amrex::FabArray<FAB>& mf, const amrex::Box& domain, const la::BcArray& bc)
{
    const amrex::BoxArray& ba = mf.boxArray();
    std::vector<CopyTask> tasks;
    for (int li = 0; li < mf.local_size(); ++li)
    {
        const amrex::Box valid = ba[mf.IndexArray()[li]];
        for (int s = 0; s < 6; ++s)
        {
            la::BcGhostFill f;
            if (!la::bcGhostFill(valid, domain, bc, s, f))
            {
                continue;
            }
            // Same box on both sides: a ghost layer is filled from that box's own
            // interior, so dst and src are the one local index.
            detail::addTask(
                tasks, li, li, f.gbx, amrex::IntVect(f.di, f.dj, f.dk), (f.sign < 0.0) ? -1 : 1
            );
        }
    }
    return detail::toDevice(tasks, "gmg_bc_plan");
}

// The valid-to-valid copy between two decompositions of the same region -- what
// ParallelCopy does, and what an agglomerated level needs both ways. Cell-centred only:
// face BoxArrays share their internal faces, so a face cell can have several sources
// and the result would depend on task order (harmless for the coefficients, which are
// copied once at setup and stay with AMReX).
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

// Declaration-only here, defined and explicitly instantiated in kernels.cpp: driven
// from both apply.cpp and bench/gmgVcycleBench.cpp, and a kernel reached from >1 CUDA
// TU must not be instantiated twice (see kernels.hpp -- the failure is a null device
// function pointer at runtime, not a link error).
//
// Executes a plan in ONE launch, one team per work block. A block short of a full
// kCopyBlock leaves lanes idle, which is cheaper than any mapping that would even it
// out.
template<class T>
void execCopyPlan(
    const char* name,
    const amrex::MultiArray4<T>& dst,
    const amrex::MultiArray4<const T>& src,
    const CopyPlan& plan
);

// Twin of FillBoundary(periodicity). Reads valid cells and writes ghosts, disjoint, so
// the whole exchange is safe in one kernel with no intermediate buffer.
template<class T>
void gmgFillBoundaryKokkos(la::GmgFab<T>& mf, const CopyPlan& plan)
{
    execCopyPlan("gmg_halo", mf.arrays(), mf.const_arrays(), plan);
}

// Twin of fillDomainBcGhostsDevice, from a makeBcPlan plan. Reads interior cells and
// writes ghosts, disjoint, so one kernel is safe.
template<class T>
void gmgFillDomainBcKokkos(la::GmgFab<T>& mf, const CopyPlan& plan)
{
    execCopyPlan("gmg_bc", mf.arrays(), mf.const_arrays(), plan);
}

// Twin of dst.ParallelCopy(src, 0, 0, 1).
template<class T>
void gmgCopyKokkos(la::GmgFab<T>& dst, const la::GmgFab<T>& src, const CopyPlan& plan)
{
    execCopyPlan("gmg_copy", dst.arrays(), src.const_arrays(), plan);
}

// Stands in for setVal(0) on a coarse solution, one launch for all boxes.
//
// Valid cells only, where setVal also clears the ghosts -- equivalent only because the
// sole reader of a coarse solution's ghosts is a colour sweep and smooth() runs a full
// ghost fill first, while prolongation reads valid cells only. Anything that read those
// ghosts unfilled would have to clear them here.
//
// Declaration-only, defined in kernels.cpp, same nvcc reason as execCopyPlan above.
template<class T>
void gmgZeroKokkos(la::GmgFab<T>& mf);

} // namespace blockamr
