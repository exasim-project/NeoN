// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

// HOST STAGING for the underscore-private row hooks (api §10.6) — B32.
//
// api §10.6 and tasks.md §3's verify column require a wall functor to be
// callable **on the host, for one cell**, against a `RecordSink`. On this build
// the marker and the packed geometry live in device memory, so "callable
// host-side" is a claim about the FUNCTOR, not about where its inputs happen to
// sit: the values are the real ones, staged into pinned host memory, and the
// code that reads them is the same `AMREX_GPU_HOST_DEVICE` member the kernel
// calls. `RobinView` needs no staging — its tables are managed memory, which is
// why they are.
//
// ---------------------------------------------------------------------------
// WHY THIS IS A HEADER AND NOT A COPY (api §10.4)
// ---------------------------------------------------------------------------
// `stageGeometryBox` was `wall_frame.cpp`'s file-local helper (B30a). B32 is the
// second row hook that needs it and B33/B34 are the third and fourth, so it is
// LIFTED here rather than copied: api §10.4 names "a private helper that a
// second file copies" as the shape to avoid. Nothing about the arithmetic moves
// with it, and nothing on an evaluate path calls it — these are the test
// bindings' staging and no production sweep stages anything.
//
// ---------------------------------------------------------------------------
// HOW A BOX IS SELECTED — VALID BOX FIRST, FAB BOX ONLY AS A FALLBACK (B33)
// ---------------------------------------------------------------------------
// Every helper below resolves the cell through `localBoxOf`: a pass over the
// local boxes selecting the one whose **valid** box contains it, and only then
// a second pass over the **fab** boxes. Until B33 there was only the second
// pass, and on a decomposed level it could pick a box for which the cell is a
// GHOST — a functor reading a face neighbour then walked off the staged fab and
// the row hook returned rows with arms silently missing (measured at B32 on
// `K5`, `max_size = 8`, at cells on a box seam).
//
// The fab-box fallback is KEPT, and it is not a leftover: a host row at a cell
// on the edge of a box must still be able to read the marker's ghosts, which is
// the `stageMarkerBox` argument below, unchanged. What the two passes add is an
// ORDER — the box that owns the cell wins over a box that merely sees it — so
// the staged fab is the same fab the kernel would resolve there.
//
// The selection is spelled WITHOUT an `MFIter` (`local_size()`, `IndexArray()`,
// `atLocalIdx()` — `AMReX_FabArray.H:535-536`). AMReX refuses a nested `MFIter`
// with an abort rather than an exception, so a staging helper that opened one
// could not be called from inside a caller's own loop; these can. `ibm::rowAt`
// (`ibm/ghost_cell.cpp`) is rewritten over the same idiom, for the same reason.
//
// **One exception, and it is the reason `localCellBoxOf` exists** (B34, B33-R's
// I-1). `localBoxOf` compares `iv` against the container's OWN boxes. For a
// cell-centred container the valid boxes TILE the level, so "the first box whose
// valid box contains it" is the owner by construction. For a FACE-centred one
// they OVERLAP on the shared face: a cell on a box seam lies in the face valid
// box of its own box AND of the box below it, and the pass returns whichever
// `IndexArray()` reaches first. `stageFaceBox` therefore resolves its cell
// through `localCellBoxOf`, which maps each candidate back to its CELL box
// first.
//
// The production sweep is unaffected by any of this — the frame resolves
// `ct.const_array(mfi)` per box inside its own `MFIter` loop and never comes
// here.

#include "../../ibm/cell_type.H"
#include "../../ibm/geometry_view.H"

#include <AMReX_Arena.H>
#include <AMReX_BaseFab.H>
#include <AMReX_Box.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_IntVect.H>
#include <AMReX_MultiFab.H>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ibm
{

//! The local index of the box that owns `iv`: a VALID box first, and only then
//! a fab box (the ghost of some other box). `-1` if no local box has it.
//!
//! No `MFIter`, deliberately — see the header note. The two passes are separate
//! loops rather than one loop with a remembered fallback because "some box owns
//! it" must beat "an earlier box sees it", whatever order the boxes come in.
template<class FA>
inline int localBoxOf(const FA& fa, const amrex::IntVect& iv)
{
    for (int li = 0; li < fa.local_size(); ++li)
        if (fa.boxArray()[fa.IndexArray()[li]].contains(iv)) return li;
    for (int li = 0; li < fa.local_size(); ++li)
        if (fa.atLocalIdx(li).box().contains(iv)) return li;
    return -1;
}

//! The local index of the box that owns the CELL `iv`, in a container whose
//! boxes may be FACE-centred (B34; B33-R's I-1).
//!
//! `localBoxOf` compares `iv` against the container's own boxes. For a
//! cell-centred container the valid boxes TILE the level, so the first pass is
//! the owner by construction. For a face-centred one in direction `dd` they
//! OVERLAP on the shared face: a cell on a box seam is contained in the face
//! valid box of its own box AND of the box below, and the pass returns whichever
//! `IndexArray()` reaches first — measured at 32/32/20 of 320 WALL cells per
//! axis on the D5 geometry at `max_size = 8`, with 80 of 96 seam cells refused
//! outright when the flux carries `ngrow = 0`.
//!
//! `enclosedCells` maps each candidate back to its CELL box, and cell boxes
//! tile, so the answer is the owning box — the same box the kernel's `MFIter`
//! resolves there. On a cell-centred container `enclosedCells` is the identity,
//! so this and `localBoxOf` agree; the two cell-centred stagers keep the
//! simpler spelling rather than being respelled for symmetry.
template<class FA>
inline int localCellBoxOf(const FA& fa, const amrex::IntVect& iv)
{
    for (int li = 0; li < fa.local_size(); ++li)
        if (amrex::enclosedCells(fa.boxArray()[fa.IndexArray()[li]]).contains(iv)) return li;
    for (int li = 0; li < fa.local_size(); ++li)
        if (amrex::enclosedCells(fa.atLocalIdx(li).box()).contains(iv)) return li;
    return -1;
}

//! The sentence a stage helper raises when no local box owns the cell.
//!
//! Shared so that the two helpers below cannot drift apart, and so that a
//! caller's own name — never "stageGeometryBox" — is what the user reads
//! (api §9).
inline std::runtime_error stageMiss(const char* fn, const amrex::IntVect& iv, const char* what)
{
    return std::runtime_error(
        std::string(fn) + ": cell [" + std::to_string(iv[0]) + ", " + std::to_string(iv[1]) + ", "
        + std::to_string(iv[2]) + "] lies in no local box of the " + what
        + " (ghosts included); there is nothing to read a row at"
    );
}

//! A host-resident copy of the packed geometry fab that owns `iv`, so a functor
//! can be called ON THE HOST for one cell.
inline void stageGeometryBox(
    const char* fn, const IbmGeometryFab& g, const amrex::IntVect& iv, amrex::FArrayBox& out
)
{
    const int li = localBoxOf(g, iv);
    if (li < 0) throw stageMiss(fn, iv, "geometry");

    const amrex::FArrayBox& src = g.atLocalIdx(li);
    const amrex::Box fb = src.box();
    out.resize(fb, GEOM_NCOMP, amrex::The_Pinned_Arena());
    const std::size_t nelem = static_cast<std::size_t>(fb.numPts()) * GEOM_NCOMP;
    amrex::Gpu::copy(amrex::Gpu::deviceToHost, src.dataPtr(), src.dataPtr() + nelem, out.dataPtr());
}

//! The same, for the marker — GHOSTS INCLUDED, and that is the point.
//!
//! A real pair gates each of its six arms on `m(i +- 1, ...)`, so a host-side
//! row at a cell on the edge of a box reads the marker's ghost region. Staging
//! the whole FAB box rather than the valid box is what makes the host row the
//! same row the kernel computes there — `localBoxOf` chooses *which* box, and
//! this copies all of it, ghosts and all.
inline void stageMarkerBox(
    const char* fn,
    const CellTypeFab& ct,
    const amrex::IntVect& iv,
    amrex::BaseFab<std::uint8_t>& out
)
{
    const int li = localBoxOf(ct, iv);
    if (li < 0) throw stageMiss(fn, iv, "cell_type marker");

    const amrex::BaseFab<std::uint8_t>& src = ct.atLocalIdx(li);
    const amrex::Box fb = src.box();
    out.resize(fb, 1, amrex::The_Pinned_Arena());
    const std::size_t nelem = static_cast<std::size_t>(fb.numPts());
    amrex::Gpu::copy(amrex::Gpu::deviceToHost, src.dataPtr(), src.dataPtr() + nelem, out.dataPtr());
}

//! The same, for a FACE-centred field — `div`'s three face fluxes (B33).
//!
//! `iv` is a CELL index and the fab is face-centred in `dd`. That is enough for
//! both reads a width-1 face balance makes: a face fab's valid box runs from the
//! cell box's low face to its high one, so it contains `iv` and `iv + e_dd`
//! together for every cell of the cell box — there is no ghost width required of
//! the flux, and none is asked for.
//!
//! The box is resolved through `localCellBoxOf` and NOT `localBoxOf` (B34;
//! B33-R's I-1): face valid boxes overlap on the shared face, so a cell on a box
//! seam is contained in two of them and the plain pass returns whichever comes
//! first in `IndexArray()`. See that helper's note for the measurement.
inline void stageFaceBox(
    const char* fn,
    const amrex::MultiFab& f,
    int dd,
    const amrex::IntVect& iv,
    amrex::FArrayBox& out
)
{
    const int li = localCellBoxOf(f, iv);
    if (li < 0) throw stageMiss(fn, iv, "face flux");

    const amrex::FArrayBox& src = f.atLocalIdx(li);
    const amrex::Box fb = src.box();
    // S-3 (B33-R): NOT `stageMiss`. A box WAS found here — its fab box simply
    // does not reach the cell's high face — and `stageMiss`'s sentence ("lies in
    // no local box of the face flux") states a different and wrong fact.
    if (!fb.contains(iv + amrex::IntVect::TheDimensionVector(dd)))
        throw std::runtime_error(
            std::string(fn) + ": cell [" + std::to_string(iv[0]) + ", " + std::to_string(iv[1])
            + ", " + std::to_string(iv[2])
            + "] is owned by a local box of the face flux whose fab box does not contain the "
              "cell's HIGH face in direction "
            + std::to_string(dd)
            + "; a width-1 face balance reads both of the cell's faces, so the flux must be on "
              "the marker's BoxArray converted to face centring in that direction"
        );

    out.resize(fb, 1, amrex::The_Pinned_Arena());
    const std::size_t nelem = static_cast<std::size_t>(fb.numPts());
    amrex::Gpu::copy(amrex::Gpu::deviceToHost, src.dataPtr(), src.dataPtr() + nelem, out.dataPtr());
}

} // namespace ibm
