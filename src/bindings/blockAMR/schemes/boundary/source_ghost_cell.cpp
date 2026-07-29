// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

// `source x ghostCell` — THE FOURTH PAIR, and the last one v1 still owned.
//
// design §6's one place an operator and a method meet, compiled. It is the
// smallest pair there can be, and that is exactly why it is the last: with it
// the row path has no production caller left, and `band_table.cpp`,
// `wall_table.cpp` and the v1 numpy row assembly are deletable.
//
// ===========================================================================
// THE SOURCE OF TRUTH
// ===========================================================================
// `src/blockamr/schemes/boundary/ghost_cell.py::GhostCellSource.rows` — v1's
// only boundary scheme that never builds a `_BandContext`:
//
//     c    = float(term.coeff) * _band_cell_values(term.field, lev, band, ncomp)
//     c[band.depth <= 0] = 0.0
//     a    = zeros((nrows, 1))          # nnz = 0: it reads no cell at all
//     nnz  = zeros(nrows, int32)
//
// So the whole row is ONE CONSTANT and no linear entry. The explicit (Su)
// source is a *coefficient field*, not the unknown: `exp.source(S)`'s operand
// IS the value, so the term's contribution at a cell is `coeff * S(cell)` and
// it reads neither `phi` nor a neighbour. There is no wall closure, no image
// point, no `ibm_bc` and no geometry in it — the pair needs the wall marker
// only to know WHICH cells are its own.
//
//   emission            value
//   constant(c)         S(i, j, k, n)          <-- and nothing else, ever
//
// `coeff` is NOT folded in here: the frame multiplies the finished sink value
// by it (`wall_apply.H`). v1 folds it into `c` instead, and the two are the
// same bits — `constant_scale` is exactly `1.0` on the apply path, so v1's
// `1.0 * (coeff * S)` and this file's `coeff * (1.0 * S)` are both the
// correctly-rounded product `coeff * S`. That is the whole of the H-5 hazard
// the other three pairs record, and here it is provably inert rather than
// merely measured: there is one multiplication on each side.
//
// ---------------------------------------------------------------------------
// WHAT HAPPENED TO v1's DEEPER BAND ROWS, AND TO ITS SOLID ROWS
// ---------------------------------------------------------------------------
// v1 emits a source row at EVERY cell of the equation's band, because its
// first term `Overwrite`s the band and a term that emitted nothing there would
// have its interior sweep erased (`ghost_cell.py`'s "the only one whose row
// touches no wall at all"). Two of those three groups are not this pair's:
//
//   * `depth == 1` (the WALL layer) — this pair, cell for cell;
//   * `depth <= 0` (SOLID) — v1's `c = 0` row. Here the mask is
//     `blockamr.pin_solid` on the accumulated result, once per level, after
//     the terms (`ibm/driver.py`, OPEN-C);
//   * `depth >= 2` — deeper FLUID cells, which exist in a v1 band only because
//     a width-2 term widened it. v2's wall sweep does not write a FLUID cell at
//     all: the interior sweep owns it, and W1's marker-aware sibling
//     (`stencil_kernels.cpp`) is what degrades it. design §5's correction at
//     B35 says this in the general case; the source term is the case where it
//     is trivially right, since a pointwise row and a pointwise interior kernel
//     compute the same number.
//
// So the parity claim this pair makes is stated at the WALL cells, which is
// where the two architectures both put a source row.
//
// ---------------------------------------------------------------------------
// FLOATING POINT
// ---------------------------------------------------------------------------
// This TU IS on `CMakeLists.txt`'s per-file `--fmad=false` list, and it is the
// one entry there that buys nothing: the functor's body is a single load and a
// single `sink.constant`, so there is no multiply-add anywhere in it for a
// contraction to move, and it includes `robin.H` not at all. It is listed
// because it includes `ibm/ghost_cell.H` — its binding takes a `GhostCellData`
// for the canonical twelve — and the rule the list is kept honest by is
// mechanical: `test_ibm_laplacian_ghost_cell.py` parses the
// `set_source_files_properties` call and requires EVERY includer of `robin.H`
// or `ibm/ghost_cell.H` under this tree to appear in it. A rule with a
// reasoned-around exception is not a rule, and the reasoning above is exactly
// the kind that is right until the file grows a second line of arithmetic.
//
// ---------------------------------------------------------------------------
// Q34 — WHERE THE GEOMETRY IS READ
// ---------------------------------------------------------------------------
// Nowhere. `stencil_reach = 0` is honest: the functor reads the source field at
// the cell it is called on and reads nothing else, off-centre or otherwise. It
// is the only pair for which that is true, and it is why the guard family in
// `validate` below is short.

#include "../../ibm/cell_type.H"
#include "../../ibm/geometry_view.H"
#include "../../ibm/ghost_cell.H"
#include "robin_data.H"
#include "wall_apply.H"
#include "wall_stage.H"
#include "wall_value.H"

#include <nanobind/nanobind.h>

#include <AMReX_Array4.H>
#include <AMReX_BaseFab.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_Geometry.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_MFIter.H>
#include <AMReX_MultiFab.H>
#include <AMReX_IntVect.H>
#include <AMReX_REAL.H>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace nb = nanobind;

namespace
{

//! The pair's functor: one wall row of `source x ghostCell`, at one cell.
//!
//! Captured BY VALUE into every thread, so every member is a view or a scalar
//! (design §4.3). File-local: nothing outside this TU names it, which is
//! design §1.3's rule 3 — changing one pair rebuilds one translation unit.
//!
//! ONE member, which is the shape of the claim: an explicit source reads the
//! source field and nothing else. No marker (the frame has already gated on
//! it), no geometry, no method data, no Robin table.
struct WallSourceGhostCell
{
    //! Zero, and load-bearing: this row names no cell but its own, so the
    //! frame's ghost-width guard has nothing to require of `phi` or of the
    //! marker. Every other pair declares 1.
    static constexpr int stencil_reach = 0;

    amrex::Array4<const amrex::Real> src; //!< the SOURCE field S, at (i,j,k) only

    template<class Sink>
    AMREX_GPU_HOST_DEVICE void operator()(int i, int j, int k, int n, Sink& sink) const
    {
        // S2/R1: the value is a datum, so it reaches the row through
        // `constant` and through nothing else — which is also what makes
        // `constant_scale = 0` (the Krylov matvec, R2) drop an explicit source
        // entirely, as it must: an Su term is a right-hand side.
        //
        // NOT `sink.linear(i, j, k, S)`: that would multiply by `phi(i,j,k)`.
        // v1's `nnz = 0` says the same thing in the row format — "it reads no
        // cell, not even its own".
        sink.constant(src(i, j, k, n));
    }
};

//! The Maker: it holds what the frame refuses to know (design §4.1) and hands
//! the functor a per-box view of it.
struct MakeWallSourceGhostCell
{
    using functor_type = WallSourceGhostCell;

    const ibm::CellTypeFab* ct;
    const ibm::IbmGeometryFab* g;
    const amrex::MultiFab* source;
    ibm::RobinView robin;

    WallSourceGhostCell operator()(const amrex::MFIter& mfi) const
    {
        return WallSourceGhostCell {source->const_array(mfi)};
    }

    //! S-5 (B30a-R): the guards the frame has no types for, made once per
    //! sweep, before any launch.
    void validate(const char* fn, int ncomp) const
    {
        // The geometry and the Robin table are unread by this functor, and are
        // still checked: they are arguments of the canonical twelve, the caller
        // is the one driver that builds them for every pair on the level, and a
        // pair that accepted a malformed one silently would make "the twelve is
        // a shape contract" a shape contract only.
        ibm::requireGeometryLayout(*g, fn);
        ibm::requireGeometryGhosts(fn, *g, ct->nGrowVect().min());
        ibm::requireRobinComponents(fn, robin.ncomp, ncomp);

        // The source is resolved by `MFIter` LOCAL INDEX beside `phi`, `out`
        // and the marker, exactly like B30a-R's I-2 guard in the frame — and
        // for the same reason: on different grids the pairing is by position,
        // which reads another box's cells.
        if (source->boxArray() != ct->boxArray()
            || source->DistributionMap() != ct->DistributionMap())
            throw std::runtime_error(
                std::string(fn)
                + ": the source field is not on this level's grids — a source wall row reads it "
                  "beside phi, out and the cell_type marker by MFIter local index, so it must "
                  "share the marker's BoxArray and DistributionMapping; rebuild the source field "
                  "for this level's grids"
            );

        // v1's `_band_cell_values` refusal (`ghost_cell.py`), transcribed. A
        // source narrower than the solved field is an out-of-bounds component
        // read, not a wrong number. v1 names the source field; the compiled
        // pair has no field name, so it names the entry point instead — the
        // B31 Invariant-F precedent, reconciled on the Python side by
        // `GhostCellSource.wall_coeff` (api §9).
        if (source->nComp() < ncomp)
            throw std::runtime_error(
                std::string(fn)
                + ": the source field has ncomp = " + std::to_string(source->nComp())
                + " but this sweep has ncomp = " + std::to_string(ncomp)
                + "; a source row carries one constant per component of the solved field, so the "
                  "two must agree"
            );
    }
};

} // namespace

void registerSourceGhostCell(nb::module_& m)
{
    // ----------------------------------------------------------------------
    // THE PRODUCTION ENTRY POINT — the canonical twelve (design §4.4) PLUS the
    // one argument the source needs.
    //
    // Q39, ruled at B32: a REGISTERED pair carries all twelve `nb::arg`s in
    // that order, with no defaults, `t` included even though an explicit source
    // has no datum and no schedule at all. Q29(f) makes the twelve a MINIMUM,
    // and B33 exercised that for `div`'s fluxes; here the thirteenth is the
    // source field itself.
    //
    // Five of the twelve are unread by this pair (`geom_ibm`, `method_data`,
    // `geom`, `t`, and `phi` beyond the frame's own use of it). They are taken
    // anyway, because the shape contract is what lets `WallEvaluation.apply`
    // call every pair from ONE keyword call site.
    // ----------------------------------------------------------------------
    m.def(
        "wall_source_ghost_cell",
        [](amrex::MultiFab& out,
           const amrex::MultiFab& phi,
           const ibm::CellTypeFab& cell_type,
           const ibm::IbmGeometryFab& geom_ibm,
           const ibm::GhostCellData&, // method_data: an Su row has no image point
           const ibm::RobinData& robin,
           const amrex::Geometry&, // geom: no distance is taken here
           double,                 // t: no datum, therefore no schedule
           double coeff,
           int ncomp,
           ibm::WallMode mode,
           double constant_scale,
           const amrex::MultiFab& source)
        {
            ibm::applyWall(
                "wall_source_ghost_cell",
                out,
                phi,
                cell_type,
                MakeWallSourceGhostCell {&cell_type, &geom_ibm, &source, robin.view()},
                coeff,
                ncomp,
                mode,
                constant_scale
            );
            amrex::Gpu::streamSynchronize();
        },
        nb::arg("out"),
        nb::arg("phi"),
        nb::arg("cell_type"),
        nb::arg("geom_ibm"),
        nb::arg("method_data"),
        nb::arg("robin"),
        nb::arg("geom"),
        nb::arg("t"),
        nb::arg("coeff"),
        nb::arg("ncomp"),
        nb::arg("mode"),
        nb::arg("constant_scale"),
        nb::arg("source"),
        "source x ghostCell over every WALL cell of the level: the explicit (Su) term's plain "
        "interior value, coeff * S(i, j, k, n). It closes no wall, reads no image point and "
        "reads no ibm_bc — an explicit source is a coefficient field, not the unknown, so its "
        "row has no linear entry at all and the whole value reaches it through 'constant'. "
        "That is also why constant_scale = 0 drops it entirely: an Su term is a right-hand "
        "side. Overwrite assigns, Add accumulates, Assemble raises. SOLID and FLUID cells are "
        "not written at all. Bitwise equal to v1's ghost_cell.GhostCellSource.rows at the "
        "level's WALL cells, row for row."
    );

    // ----------------------------------------------------------------------
    // TEST binding (api §4, §10.6) — the same functor, on the HOST, at ONE
    // cell, against a `RecordSink`. Underscore-private, never registered, never
    // on an evaluate path, and exempt from the twelve by Q39.
    //
    // It takes the marker as well as the source, because "is this cell a WALL
    // cell" is the frame's question and the hook must be able to refuse a cell
    // the frame would never have called the functor on.
    // ----------------------------------------------------------------------
    m.def(
        "_wall_row_source_ghost_cell",
        [](const ibm::CellTypeFab& cell_type,
           const amrex::MultiFab& source,
           int i,
           int j,
           int k,
           int n) -> nb::tuple
        {
            static constexpr const char* FN = "_wall_row_source_ghost_cell";

            if (n < 0 || n >= source.nComp())
                throw std::runtime_error(
                    std::string(FN) + ": component " + std::to_string(n)
                    + " is outside the source field's " + std::to_string(source.nComp())
                );

            const amrex::IntVect iv(i, j, k);

            // The same staging shape as `wall_stage.H`'s three helpers: find the
            // local box that owns the cell, copy its whole fab to the host, read
            // it there. A cell-centred FIELD stager is the fourth of that family
            // and is deliberately NOT lifted into the header — `wall_stage.H` is
            // included by `wall_frame.cpp`, and B33/B34 already recorded that the
            // family moves only when a third caller needs the same one.
            amrex::BaseFab<std::uint8_t> hostM;
            ibm::stageMarkerBox(FN, cell_type, iv, hostM);

            const int li = ibm::localBoxOf(source, iv);
            if (li < 0) throw ibm::stageMiss(FN, iv, "source field");
            const amrex::FArrayBox& fab = source.atLocalIdx(li);
            amrex::FArrayBox hostS(fab.box(), source.nComp(), amrex::The_Pinned_Arena());
            const std::size_t nelem = static_cast<std::size_t>(fab.box().numPts()) * source.nComp();
            amrex::Gpu::copy(
                amrex::Gpu::deviceToHost, fab.dataPtr(), fab.dataPtr() + nelem, hostS.dataPtr()
            );
            amrex::Gpu::streamSynchronize();

            if (hostM.const_array()(i, j, k) != ibm::WALL)
                throw std::runtime_error(
                    std::string(FN) + ": cell [" + std::to_string(i) + ", " + std::to_string(j)
                    + ", " + std::to_string(k)
                    + "] is not a WALL cell of this level; a wall row exists only where the "
                      "marker is WALL"
                );

            const WallSourceGhostCell f {hostS.const_array()};
            ibm::RecordSink rec;
            f(i, j, k, n, rec);

            nb::list entries;
            for (int e = 0; e < rec.count; ++e)
                entries.append(nb::make_tuple(
                    rec.entries[e].i, rec.entries[e].j, rec.entries[e].k, rec.entries[e].a
                ));
            return nb::make_tuple(entries, rec.c);
        },
        nb::arg("cell_type"),
        nb::arg("source"),
        nb::arg("i"),
        nb::arg("j"),
        nb::arg("k"),
        nb::arg("n"),
        "TEST ONLY. source x ghostCell's row at one WALL cell, computed on the HOST against a "
        "RecordSink: returns ([(i, j, k, a), ...], c). The list is ALWAYS empty — an explicit "
        "source reads no cell — and c is S(i, j, k, n). Raises if the cell is not a WALL cell "
        "of this level."
    );
}
