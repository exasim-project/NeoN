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
#include <AMReX_iMultiFab.H>

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
//! **The rank is also published per cell**, as `row` — B32's addition, and the
//! half of the contract B31 could not express. A wall functor is called at a
//! cell and this data is indexed by rank, so without a cell -> rank map the two
//! cannot meet: the scan that produced the ranks used to be a transient
//! per-box `DeviceVector` freed on return. `row` is that scan, retained
//! (4 bytes per cell of the level, review.md §4 Q49(d); the O(nWall) compaction
//! plus a binary search is the named reduction if production sizes ever hurt).
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

    //! `(i, j, k) -> rank in the four arrays above`, and `-1` wherever the
    //! marker is not `WALL`.
    //!
    //! On the MARKER's own `BoxArray` and `DistributionMapping` with
    //! `ngrow = 0`, which is what lets a wall functor take it as
    //! `row.const_array(mfi)` inside the frame's `MFIter` loop and be paired
    //! with `phi`, `out` and `ct` by local index. That pairing is checked once
    //! per sweep by the pair's `Maker::validate` (B30a-R's S-5), because the
    //! frame cannot see a method's data type.
    amrex::iMultiFab row;
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

//! The rank of one cell, read back HOST-side out of `row` — one `int` across
//! the bus, for a test binding and never for a sweep (a kernel reads
//! `row.const_array(mfi)` and never this).
//!
//! `-1` where the marker is not `WALL`. A cell no local box owns raises naming
//! `fn`, because "this level has no row there" and "you asked about a cell that
//! is not on this level" are different answers and only one of them is a `-1`.
//!
//! Opens **no** `MFIter` (B33), so it is callable from inside one: the box is
//! found through `IndexArray()` + `atLocalIdx()`, which is `wall_stage.H`'s
//! `localBoxOf` idiom. Until B33 it did open one, and AMReX refuses a nested
//! `MFIter` with an abort rather than an exception — a readback loop that called
//! this per cell from inside `MFIterator` aborted (measured at B32), and the
//! workaround was to materialise the whole map first. That is no longer needed.
int rowAt(const GhostCellData& d, const char* fn, int i, int j, int k);

} // namespace ibm
