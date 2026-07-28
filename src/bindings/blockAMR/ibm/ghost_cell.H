// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

// `ghostCell`'s own preprocessed data (design §2.4, api §10.3) — B31.
//
// The method-agnostic layer (the marker, the packed geometry) is `cell_type.H`
// and `geometry_view.H`. What is left, and what this file owns, is the one
// extra the sharp-interface method's boundary schemes need: per FLUID
// wall-layer cell an image point on the outward normal, the trilinear stencil
// that reads the field there, and the distance from that point to the surface.
//
// THE DATA TYPE AND NOTHING ELSE. There is no wall arithmetic here: no closure,
// no alpha/beta/gamma, no 1/d. That is design §6's check S7 refuted in the file
// tree — the closure lives in `robin.H` (B30b) and the functors in
// `schemes/boundary/*_ghost_cell.cpp` (B32-B34), and none of them may push a
// formula back into this header.
//
// v1's numpy peer is `src/blockamr/ibm/ghost_cell.py`; the arithmetic is
// transcribed in `ghost_cell.cpp` and pinned to it BITWISE by
// `test/blockamr/test_ibm_ghost_cell_cpp.py`.

#include "cell_type.H"
#include "geometry_view.H"

#include <AMReX_GpuContainers.H>
#include <AMReX_Geometry.H>
#include <AMReX_REAL.H>

#include <string>
#include <type_traits>
#include <vector>

namespace ibm
{

//! Trilinear stencil size — the donor slot count of `GhostCellData`, and the
//! literal mirror of `ghost_cell.py`'s `K = 8`. Slot `s` carries the corner
//! offset `o = (s >> 2 & 1, s >> 1 & 1, s & 1)`, which is `geometry.py`'s
//! `_OFFSETS` order (`for i in (0,1) for j in (0,1) for k in (0,1)`).
inline constexpr int K = 8;

// The two dtype facts the bitwise-parity bar rests on. They are asserted here
// rather than in the TU so that every future consumer of this header inherits
// them (B32-B34 capture `GhostCellView` by value in their functors).
static_assert(
    std::is_same_v<amrex::Real, double>,
    "GhostCellData's parity bar is to f64: v1's numpy image points, weights and "
    "distances are float64, and a single-precision amrex::Real would pass "
    "assert_allclose while failing assert_array_equal. Build with "
    "NeoN_DEFINE_DP_SCALAR ON."
);
static_assert(
    sizeof(int) == 4,
    "GhostCellData::donor is exported to numpy as int32, which is the dtype "
    "test_ibm_ghost_cell.py pins on the v1 side (`data.donor.dtype == np.int32`)."
);

//! `ghostCell`'s per-level data, SoA and flat (api §10.3).
//!
//! One row per FLUID wall-layer cell (`m == ibm::WALL`), per local box in
//! `MFIter` order, and within a box in v1's lexicographic `(i, j, k)` order
//! with **k varying fastest** — `np.argwhere`'s C order. That row order is a
//! CONTRACT, not an implementation detail: B32's boundary schemes index this
//! data by the same rank, and `preprocess` therefore assigns ranks by an
//! exclusive scan over the box's k-fastest linear index and never by an atomic
//! append (which would be both a different order and a non-deterministic one).
//!
//! `nrows` is carried explicitly because it is `image_point.size() / 3` for
//! every count but zero, and the empty band is a real and supported case.
struct GhostCellData
{
    amrex::Gpu::DeviceVector<amrex::Real> image_point; //!< [nrows * 3], `r*3 + d`
    amrex::Gpu::DeviceVector<int> donor;               //!< [nrows * K * 3], `(r*K + s)*3 + d`
    amrex::Gpu::DeviceVector<amrex::Real> weight;      //!< [nrows * K], `r*K + s`
    amrex::Gpu::DeviceVector<amrex::Real> distance;    //!< [nrows], `r`
    int nrows = 0;
};

//! The device-copyable POD a boundary functor captures by value (design §1.3).
//!
//! Read-only by construction, like `IbmGeometryView`: a wall functor cannot
//! write the stencil it reads.
struct GhostCellView
{
    const amrex::Real* image_point;
    const int* donor;
    const amrex::Real* weight;
    const amrex::Real* distance;
    int nrows;
};

//! The view of `d`. Host-side; the result is what a kernel captures.
inline GhostCellView makeGhostCellView(const GhostCellData& d) noexcept
{
    return GhostCellView {
        d.image_point.data(), d.donor.data(), d.weight.data(), d.distance.data(), d.nrows
    };
}

//! `ghostCell`'s preprocessing (v1: `GhostCell.preprocess`), once per
//! `(method, lev, grid_version)` and pure geometry — it reads no field value.
//!
//! `ct` is the classified marker and `g` the packed geometry it was classified
//! against; `geom` supplies `dx` and `prob_lo`. `patchNames` is
//! `sorted(mesh.bodies)`, used only to name the offending patch when Invariant
//! F fails — pass an empty vector to have the raise name the patch id instead.
//!
//! Raises `std::runtime_error` when a live trilinear donor is not a fluid cell
//! (Invariant F, `plans/IBM/row-contract.md` §8).
void preprocess(
    GhostCellData& out,
    const CellTypeFab& ct,
    const IbmGeometryFab& g,
    const amrex::Geometry& geom,
    const std::vector<std::string>& patchNames
);

} // namespace ibm
