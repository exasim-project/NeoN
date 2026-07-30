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
// decompositions of one region, and the zero fill. ONE RANK ONLY -- a plan names LOCAL box
// indices. Why, and how a plan is built: report/blockamr-gmg-notes.md#halo-plans.

namespace blockamr
{

// Threads per work block, and so the team size of the copy kernel.
constexpr int kCopyBlock = 128;

// One work block of one rectangular region copy, in GLOBAL cell indices:
//   dst[dstBox](i, j, k) = sign * src[srcBox](i + sh[0], j + sh[1], k + sh[2])
// over cells [base, base + kCopyBlock) of [lo, lo + len), i fastest. Box indices are LOCAL;
// `sh` is 0 for a same-region copy, the negated image shift for a wrapped ghost. Why
// fixed-size blocks and not one team per region: notes#halo-plans.
struct CopyTask
{
    int dst;
    int src;
    int lo[3];
    int len[3];
    int sh[3];
    int base;
    // -1 only for a reflect-odd (homogeneous Dirichlet) domain ghost: a mirror copy, negated.
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

// The ghost exchange of one FabArray: for every ghost cell, the valid cell -- of a box or of a
// periodic image -- that FillBoundary would copy it from. Why the shell partition makes task
// order irrelevant: notes#halo-plans.
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
                // Shift the query back into the source frame and the answer forward again.
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

// The homogeneous domain-boundary ghost fill as a plan: per valid box on a non-periodic domain
// face, the one-cell ghost layer and the mirror interior cell (sign -1 Dirichlet, +1 Neumann).
// Twin of fillDomainBcGhosts* via la::bcGhostFill; face layers only, and runs AFTER the halo.
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
            // A ghost layer comes from its own box's interior, so dst and src are one index.
            detail::addTask(
                tasks, li, li, f.gbx, amrex::IntVect(f.di, f.dj, f.dk), (f.sign < 0.0) ? -1 : 1
            );
        }
    }
    return detail::toDevice(tasks, "gmg_bc_plan");
}

// The valid-to-valid copy between two decompositions of one region -- what ParallelCopy does.
// Cell-centred only: a face cell can have several sources, so the result would depend on task
// order (notes#halo-plans).
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

// Executes a plan in ONE launch, one team per work block.
// Instantiated in kernels.cpp; a miss is a null device fnptr at runtime, not a link error.
template<class T>
void execCopyPlan(
    const char* name,
    const amrex::MultiArray4<T>& dst,
    const amrex::MultiArray4<const T>& src,
    const CopyPlan& plan
);

// Twin of FillBoundary(periodicity). Reads valid cells and writes ghosts, so one kernel is safe.
template<class T>
void gmgFillBoundaryKokkos(la::GmgFab<T>& mf, const CopyPlan& plan)
{
    execCopyPlan("gmg_halo", mf.arrays(), mf.const_arrays(), plan);
}

// Twin of fillDomainBcGhostsDevice, from a makeBcPlan plan; disjoint, so one kernel is safe.
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

// Stands in for setVal(0) on a coarse solution, one launch for all boxes. VALID cells only,
// which is equivalent here but need not stay so (notes#halo-plans).
// Instantiated in kernels.cpp; a miss is a null device fnptr at runtime, not a link error.
template<class T>
void gmgZeroKokkos(la::GmgFab<T>& mf);

} // namespace blockamr
