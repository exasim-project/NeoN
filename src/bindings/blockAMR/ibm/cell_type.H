// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

// The marker (design §2.2, api §6). Three values, one byte, and no method may
// add a fourth — that is conformance check M4, and `validateCellType` below is
// what turns a fourth value into a sentence instead of a silent `FLUID`.

#include "geometry_view.H"

#include <AMReX_BaseFab.H>
#include <AMReX_FabArray.H>
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_REAL.H>

#include <cstdint>
#include <type_traits>

namespace ibm
{

//! The Python-facing name (api §6.3): `blockamr.CellType.SOLID` and friends.
enum class CellType : std::uint8_t
{
    SOLID = 0, //!< inside a body: no-op, nothing writes it, no live stencil reads it
    WALL = 1,  //!< a fluid cell whose discretisation the surface changes
    FLUID = 2  //!< everything else: the interior scheme, unchanged
};

//! The device-side vocabulary (api §6.1). Kernels compare against these; they
//! are derived from the enum so the two spellings cannot drift apart.
inline constexpr std::uint8_t SOLID = static_cast<std::uint8_t>(CellType::SOLID);
inline constexpr std::uint8_t WALL = static_cast<std::uint8_t>(CellType::WALL);
inline constexpr std::uint8_t FLUID = static_cast<std::uint8_t>(CellType::FLUID);

//! Same BoxArray, same DistributionMapping and the same ghost region as the
//! fields it accompanies — which is why nothing in api §8 has to check that.
using CellTypeFab = amrex::FabArray<amrex::BaseFab<std::uint8_t>>;

static_assert(
    std::is_same_v<std::uint8_t, unsigned char>,
    "CellTypeFab's FillBoundary resolves ParallelDescriptor::Mpi_typemap<unsigned char>; "
    "a uint8_t that is not unsigned char has no typemap"
);

//! The marker's ghost-width FLOOR (design §2.1, review F10) — **not** an
//! allocation size, and never a value to allocate a `CellTypeFab` at.
//!
//! Pass 2 of `classifyDefault` reads a face neighbour of every valid cell, so
//! the default rule cannot work with fewer than one ghost cell. That is all
//! this constant says. A method whose stencil reaches further needs a wider
//! marker and is entitled to one: `requireMarkerGhostWidth` compares against
//! `ct.nGrowVect().min()` rather than against this number, so a wider marker is
//! already legal and already forces a proportionally wider geometry through
//! `requireGeometryGhosts`. W1's degrade (B35) reads `m(i +- 2s)` and will want
//! **2** — allocate the marker at the width the method that owns it needs, and
//! nothing here has to move when B35 lands. Exercised green at ngrow = 2 by
//! `test_ibm_cell_type.py` so the wide-marker path is not first tried at B35.
inline constexpr int MARKER_NGROW = 1;

//! The default classification (design §2.2), shared by every method that does
//! not override it. Two passes and two `FillBoundary`s, then M4 + M5.
void classifyDefault(CellTypeFab& ct, const IbmGeometryFab& g, const amrex::Geometry& geom);

//! M4 + M5 (design §2.4, api §8): only the three values anywhere including the
//! ghosts, and every `SOLID` has `sdf <= 0` while every `WALL` has `sdf > 0`.
void validateCellType(const CellTypeFab& ct, const IbmGeometryFab& g);

//! The v2 `SOLID` pin (design §7, Q3; joins this TU by Q29c). The only write
//! this architecture makes to a user field, and it happens at classification
//! time, so `evaluate` never writes the field.
void pinSolid(amrex::MultiFab& phi, const CellTypeFab& ct, amrex::Real pin, int ncomp);

} // namespace ibm
