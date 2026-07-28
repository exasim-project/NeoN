// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

// FROZEN at B29 (review.md §4, Q29a/Q29b; shipped provisionally by B28).
//
// Q29(b) rules the method-agnostic geometry layer (design §2.1) into ONE packed
// eight-component MultiFab behind a device-copyable POD view:
//
//     comp 0      sdf          f64, > 0 in the fluid
//     comp 1..3   normal       unit into-fluid normal at the cell centre
//     comp 4..6   wall_point   body intercept x - sdf * n
//     comp 7      patch        owning body id, carried as f64
//
// ---------------------------------------------------------------------------
// Why this file is frozen, and what "frozen" costs
// ---------------------------------------------------------------------------
// This header is included by every compiled consumer of the immersed geometry:
// `ibm/cell_type.cpp` (B28) today, and `ibm/ghost_cell.cpp` (B31) plus the three
// `schemes/boundary/*_ghost_cell.cpp` wall pairs (B32/B33/B34) as they land —
// five CUDA translation units, the largest single-header rebuild fan-out in the
// v2 IBM tree. design §1.3's rule 3 ("changing one pair rebuilds one TU") holds
// for the `.cpp` files; this header is the deliberate exception. A change to it
// after B29 is therefore a compile session of its own, never a rider on another
// task. The accessors are member functions rather than raw members precisely so
// that a *later* addition (a curvature, an sdf at a face) is additive: a new
// component index does not disturb an existing call site.
//
// ---------------------------------------------------------------------------
// What the layout guarantees, and what it deliberately does not carry
// ---------------------------------------------------------------------------
// * The view is READ-ONLY by construction: `Array4<const Real>` and no mutable
//   spelling exists anywhere in this type, so a wall functor is *unable* to
//   write the geometry it reads (design §2.4, M1/M2). The fill happens once per
//   grid generation, through the plain `MultiFab` Python holds.
// * `patch` is logically an integer id (design §2.1's dtype column says int8)
//   and is physically carried as an f64 component of the packed fab — Q29(b)'s
//   stated deviation. It is `argmin`-derived, so it is a small non-negative
//   integer that round-trips through f64 exactly.
// * There is NO `depth`, no `MAX_DEPTH`, no stencil shape and no ray walk. That
//   is the whole of "`IbmGeometry` loses `depth`" on the compiled side: the
//   member does not exist, so nothing in the v2 path can ask for it, and the
//   question it used to answer is answered by the marker (`ibm/cell_type.H`).
//   v1's Python `IbmGeometry` dataclass keeps `depth` until B36/B37; the two
//   coexist by not sharing a name.
// * The view carries no box, no `nGrow` and no `MFIter`: it addresses whatever
//   its `Array4` addresses (the fab box, ghosts included). The ghost-width
//   promise below is checked ONCE PER FAB, host-side, never per cell.
//
// ---------------------------------------------------------------------------
// THE GHOST CONTRACT (review F10, honoured at B29) — two halves, both binding
// ---------------------------------------------------------------------------
// (1) WIDTH. `geometry.nGrow() >= marker.nGrow()`. Pass 1 of `classifyDefault`
//     runs on `mfi.growntilebox()` and reads `sdf` at every one of those
//     indices; `validateCellType` runs over `mfi.fabbox()` and does the same.
//     `requireGeometryGhosts` below is that check.
//
//     What the grown analytic geometry buys, stated narrowly (B29-R, I-1): it
//     is what makes pass 1's SOLID/FLUID correct in a ghost cell OUTSIDE A
//     NON-PERIODIC DOMAIN FACE, where `FillBoundary` has no source at all.
//     Everywhere else — across a box edge, across a periodic seam — the halo
//     exchange supplies the ghosts. `classifyDefault` does run two
//     `ct.FillBoundary` calls, and a ghost WALL marker can only have come from
//     the second one, because pass 2 launches on the valid box. So this is not
//     a claim that no marker exchange is needed; it is the claim that no
//     exchange COULD have filled that particular shell.
//
// (2) VALUE. The geometry's ghost cells must be MEANINGFUL over the whole fab
//     box, and specifically:
//       - across a periodic seam a ghost must carry the geometry of the
//         **wrapped** cell — that halo cell *is* the wrapped cell, body and
//         all. This is not a stylistic choice: `classifyDefault` fills the
//         marker's ghosts with `FillBoundary`, i.e. from the wrapped valid
//         cell, and the always-on M5 check then compares that marker against
//         *this* fab's ghost sdf at the same index. Fill the ghost by
//         evaluating the analytic body at the UNWRAPPED coordinate and every
//         classification of a body near a periodic boundary throws M5.
//       - outside a non-periodic domain face there is nothing to wrap onto and
//         nothing for `FillBoundary` to copy from, so the ghost must be filled
//         by evaluating the analytic body at the extended coordinate. A
//         `FillBoundary`-based geometry fill cannot satisfy this half at all;
//         it is the reason the fill is analytic and grown.
//     Both halves are exactly `blockamr.ibm.classify`'s documented conventions,
//     and the v2 builder (`ibm/geometry.packed_box_geometry`, uploaded through
//     `MultiFab.copy_grown_from` per Q29(d)) honours them by evaluating on
//     `_index_grid(grid, ngrow)` at `_index_coords`, which wraps.

#include <AMReX_Array4.H>
#include <AMReX_Extension.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_MFIter.H>
#include <AMReX_MultiFab.H>
#include <AMReX_REAL.H>

#include <stdexcept>
#include <string>

namespace ibm
{

//! Component offsets of the packed geometry fab. Q29(b)'s layout, in one place.
inline constexpr int GEOM_SDF = 0;
inline constexpr int GEOM_NORMAL = 1;
inline constexpr int GEOM_WALL_POINT = 4;
inline constexpr int GEOM_PATCH = 7;
inline constexpr int GEOM_NCOMP = 8;

//! The host-side container (api §6.2's name for it).
//!
//! FROZEN as an alias, not promoted to a distinct struct (B28-R's S5). The fab
//! is allocated and filled from Python — Q29(d) makes the upload path the
//! already-bound `MultiFab.copy_grown_from` — so a distinct C++ type would need
//! its own nanobind class, its own constructor and its own copy-in path to buy
//! nothing the packed-layout guard below does not already buy. The layout IS
//! the contract, and `requireGeometryLayout` is where it is enforced; every
//! compiled signature still spells `IbmGeometryFab`, so promoting the alias
//! later is a one-line change here plus the guards, not a call-site sweep.
using IbmGeometryFab = amrex::MultiFab;

//! The device-copyable POD a kernel captures by value (design §1.3/§2.4).
struct IbmGeometryView
{
    amrex::Array4<const amrex::Real> a;

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real sdf(int i, int j, int k) const noexcept
    {
        return a(i, j, k, GEOM_SDF);
    }

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
    normal(int i, int j, int k, int d) const noexcept
    {
        return a(i, j, k, GEOM_NORMAL + d);
    }

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
    wallPoint(int i, int j, int k, int d) const noexcept
    {
        return a(i, j, k, GEOM_WALL_POINT + d);
    }

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE int patch(int i, int j, int k) const noexcept
    {
        return static_cast<int>(a(i, j, k, GEOM_PATCH));
    }
};

//! One box of the packed fab, as the device view.
inline IbmGeometryView makeGeometryView(const IbmGeometryFab& g, const amrex::MFIter& mfi)
{
    return IbmGeometryView {g.const_array(mfi)};
}

//! Host guard: the packed layout is a contract, and a wrong width is a sentence.
inline void requireGeometryLayout(const IbmGeometryFab& g, const char* who)
{
    if (g.nComp() != GEOM_NCOMP)
        throw std::runtime_error(
            std::string(who) + ": the immersed-body geometry is one packed MultiFab of "
            + std::to_string(GEOM_NCOMP)
            + " components (sdf, normal x3, wall_point x3, patch), but this one has "
            + std::to_string(g.nComp()) + "; rebuild the geometry"
        );
}

//! Host guard: half (1) of the ghost contract above — the geometry fab against
//! the width its reader needs, once per fab, never per cell.
//!
//! This is the FAB-level check and it is deliberately named apart from
//! `requireGhostWidth` (api §8, B30's `wall_apply.H`), which polices a *field*
//! fab against a functor's declared stencil reach, per `MFIter`, per launch.
//! Two subjects, two checks, one message shape: the width it needs and the
//! width it has.
//!
//! Not defensive coding against an impossible state (CLAUDE.md §2): the
//! geometry is cached on `(lev, grid_version)` with no ngrow in the key
//! (design §8) while a marker's width follows the stencil of the method that
//! allocates it, so a geometry built for a one-cell marker and then handed to a
//! two-cell one is an ordinary sequence of user calls. The Python builder makes
//! it not *happen* (it grows monotonically); this makes it not *silently*
//! happen.
//!
//! The sentence states the PAIRING, not a read pattern, and that is B31-R's
//! I-2 / review.md §4 Q37(ii), closed at B30a. It used to say "the immersed
//! geometry is read over the marker's grown box" — true at `classifyDefault`
//! and `validateCellType`, which do exactly that, and FALSE at
//! `ghost_cell.cpp`'s `preprocess`, which reads the geometry at valid cells
//! only and passes the marker's width as a proxy for "this marker came from
//! this geometry" (see the call-site comment there). One guard with three
//! callers cannot narrate one caller's read pattern, so it narrates the
//! invariant all three share.
inline void requireGeometryGhosts(const char* fn, const IbmGeometryFab& g, int needs)
{
    const int has = g.nGrowVect().min();
    if (has >= needs) return;
    throw std::runtime_error(
        std::string(fn)
        + ": the marker is classified against this geometry over the marker's whole fab box, so "
          "the geometry's ghost width must be at least the marker's "
        + std::to_string(needs) + ", but the geometry MultiFab has " + std::to_string(has)
        + "; grow the geometry"
    );
}

} // namespace ibm
