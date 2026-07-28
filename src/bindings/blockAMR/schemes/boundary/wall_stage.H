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
// KNOWN LIMITATION — SINGLE-BOX LEVELS ONLY, OWED AT B33 (measured at B32)
// ---------------------------------------------------------------------------
// Both helpers below select the FIRST local box whose **fab** box contains the
// cell. On a decomposed level that can be a box for which the cell is a GHOST,
// and then a functor reading a face neighbour walks off the staged fab: at
// `max_size = 8` the row hook returned rows with arms silently missing
// (measured on `K5` at cells on a box seam). The production sweep is
// unaffected — the frame resolves `ct.const_array(mfi)` per box inside its own
// `MFIter` loop and never comes here — and B32's rows that use the hook all run
// on single-box levels, where the selection is unambiguous.
//
// The fix is three lines and needs a build slot, which B32 had already spent
// when this was found: pass 1 over the boxes selecting `mfi.validbox()`, and
// only then fall back to the fab box. B33 owns it.

#include "../../ibm/cell_type.H"
#include "../../ibm/geometry_view.H"

#include <AMReX_Arena.H>
#include <AMReX_BaseFab.H>
#include <AMReX_Box.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_IntVect.H>
#include <AMReX_MFIter.H>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ibm
{

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
    for (amrex::MFIter mfi(g); mfi.isValid(); ++mfi)
    {
        const amrex::Box fb = mfi.fabbox();
        if (!fb.contains(iv)) continue;

        const amrex::FArrayBox& src = g[mfi];
        out.resize(fb, GEOM_NCOMP, amrex::The_Pinned_Arena());
        const std::size_t nelem = static_cast<std::size_t>(fb.numPts()) * GEOM_NCOMP;
        amrex::Gpu::copy(
            amrex::Gpu::deviceToHost, src.dataPtr(), src.dataPtr() + nelem, out.dataPtr()
        );
        return;
    }

    throw stageMiss(fn, iv, "geometry");
}

//! The same, for the marker — GHOSTS INCLUDED, and that is the point.
//!
//! A real pair gates each of its six arms on `m(i +- 1, ...)`, so a host-side
//! row at a cell on the edge of a box reads the marker's ghost region. Staging
//! `mfi.fabbox()` rather than the valid box is what makes the host row the same
//! row the kernel computes there.
inline void stageMarkerBox(
    const char* fn,
    const CellTypeFab& ct,
    const amrex::IntVect& iv,
    amrex::BaseFab<std::uint8_t>& out
)
{
    for (amrex::MFIter mfi(ct); mfi.isValid(); ++mfi)
    {
        const amrex::Box fb = mfi.fabbox();
        if (!fb.contains(iv)) continue;

        const amrex::BaseFab<std::uint8_t>& src = ct[mfi];
        out.resize(fb, 1, amrex::The_Pinned_Arena());
        const std::size_t nelem = static_cast<std::size_t>(fb.numPts());
        amrex::Gpu::copy(
            amrex::Gpu::deviceToHost, src.dataPtr(), src.dataPtr() + nelem, out.dataPtr()
        );
        return;
    }

    throw stageMiss(fn, iv, "cell_type marker");
}

} // namespace ibm
