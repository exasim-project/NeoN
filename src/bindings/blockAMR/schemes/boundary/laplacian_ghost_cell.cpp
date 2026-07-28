// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

// `laplacian x ghostCell` — THE FIRST REAL `(operator, method)` PAIR (B32).
//
// design §6's one place an operator and a method meet, compiled. The frame
// (`wall_apply.H`), the sinks (`wall_value.H`), the BC transport
// (`robin_data.H`), the wall formula (`robin.H`) and the method's data
// (`ibm/ghost_cell.H`) all already exist and none of them moves here: what this
// file adds is the ASSEMBLY — which cells a row names, in which order, with
// which coefficients.
//
// ===========================================================================
// THE SOURCE OF TRUTH
// ===========================================================================
// `src/blockamr/schemes/boundary/ghost_cell.py::_closed_flux_rows`, whose row
// is `coeff * sum_d (G_d^+ - G_d^-) / dx_d`: an arm whose face neighbour is
// FLUID keeps the interior scheme's own formula exactly (which is what makes a
// constant annihilate to the last bit), and an arm whose face neighbour is not
// fluid is taken at the surface instead, `G_d = n_d * dphi/dn|_w` out of the
// closure's gradient half.
//
// Read together with its consumer (`band_table.cpp:673-694`), a v1 wall row is
// ONE CONSTANT THEN FIFTEEN SLOTS, in this exact order:
//
//   emission            slot    value
//   constant(c)          --     sum over WALL arms of  scale * grad_constant
//   linear(P, .)          0     -sum over FLUID arms of 1/dx_d^2   (ACCUMULATED)
//   linear(P +- e_d, .) 1..6    +1/dx_d^2, for a FLUID arm only
//   linear(donor_q, .)  7..14   sum over WALL arms of (scale*grad_linear)*w_q
//
// and the loop order is `for d in 0,1,2: for step in +1, -1` — **+1 first**.
//
// ---------------------------------------------------------------------------
// THE FIVE TRANSCRIPTION HAZARDS, EACH MARKED AT ITS SITE BELOW
// ---------------------------------------------------------------------------
// H-1  v1 iterates `for step in (1, -1)`. api §5.3's published sketch writes
//      `for (step = -1; step <= 1; step += 2)` — MINUS ONE FIRST — which is a
//      different accumulation order for both the diagonal and the donor sums.
//      Measured pre-build: it moves bits on 8 of 8 acceptance configurations.
// H-2  `1.0 / (dx[d] * dx[d])`. `x ** 2` on a numpy f64 scalar is one
//      correctly-rounded squaring, identical to `x*x`; `(1.0/dx) * (1.0/dx)`
//      is NOT. Do not "save a divide". Moves bits on 8 of 8.
// H-3  the diagonal is ACCUMULATED in a register and emitted once. Emitting
//      `-inv` per arm — api §5.3's sketch shape — is bit-different from v1 on
//      every wall row of all 8 configurations.
// H-4  `const Real sg = scale * grad_linear;` ONCE PER ARM, then
//      `wdon[q] += sg * weight[q]`. The algebraically equal
//      `scale * (grad_linear * weight[q])` is a different number: caught on 5
//      of 8 configurations and INVISIBLE on the two obvious rung geometries.
//      This is review.md §4 Q35's trap, discharged by measurement.
// H-5  **the one documented departure from v1's association.** v1 scales the
//      coefficients BEFORE the dot product (`a *= coeff; c *= coeff`); the
//      frame scales the sum AFTER (`coeff * sink.value()`). The two are
//      identical only when `coeff` is exactly `1.0`, which every rung and every
//      parity row in the acceptance set uses (`exp.laplacian(1.0, T)` ->
//      `_coefficient` returns `1.0`). It is RECORDED, not fixed: fixing it
//      means editing `wall_apply.H`'s contract for every pair.
//
// A measured NON-hazard, stated so it is not "fixed" later: `step*(n/dx)` for
// `(step*n)/dx` moves nothing, because `step` is exactly `+-1.0`. It is carried
// in the suite as a CONTROL — a mutant that must change no bit — and never as
// coverage.
//
// ===========================================================================
// FLOATING POINT (review.md §4 Q36, RULED at this task)
// ===========================================================================
// This TU is on `CMakeLists.txt`'s per-file `--fmad=false` /
// `-Xcompiler=-ffp-contract=off` list, and it MUST be: `robin.H` is a header,
// so its `beta - alpha*distance` is inlined with THIS TU's flags, and the
// accumulations below (`wdon[q] += sg * w`, `cacc += scale * gc`,
// `diag -= 1.0/(dx*dx)`) are multiply-accumulate shapes of this file's own.
// The flag is load-bearing for B32's own bitwise bar, not merely inherited.
//
// Q36 ruled per-file opt-in rather than a directory posture (the reasons are in
// `robin.H`'s header), and the list is kept honest mechanically: a row in
// `test_ibm_laplacian_ghost_cell.py` asserts that every includer of `robin.H`
// or `ibm/ghost_cell.H` under this tree appears in that CMake list.
//
// ===========================================================================
// Q34 — WHERE THE GEOMETRY IS READ
// ===========================================================================
// At the target cell and nowhere else: `g.patch(i, j, k)` and
// `g.normal(i, j, k, dd)`. No `normal`, `sdf` or `wall_point` at a neighbour or
// a ghost index, which is the question B29's freeze left open and which
// `wall_apply.H`'s ghost-width guard is written against. This is not an
// inspection claim — `test_ibm_wall_functors.py` perturbs the geometry fab at a
// neighbour index and requires the row to be bit-identical, and perturbs it at
// the cell itself and requires it to change (B30a-R's I-1 lesson).
//
// ===========================================================================
// THE POLE IS NOT GUARDED HERE EITHER (review.md §4 Q46, RULED at this task)
// ===========================================================================
// `robin.H`'s `den = beta - alpha*d` reaches exactly zero for `Mixed(f)` with
// `d = (1 - f)/f`, and v1 divides anyway and returns `+-inf`. This pair adds NO
// raise: a raise where v1 returns a number is a behaviour change, and this
// task's whole claim is v1<->v2 bitwise parity. The consequence is stated
// rather than hidden — an `inf` row reaching a real sweep makes `ApplySink::acc`
// `inf`/`NaN` and the frame writes it into `out` for the whole cell, silently,
// because `RecordSink::constant` accumulates from `+0.0` and `ApplySink` only
// multiplies. The configuration is not reachable in the acceptance set (it
// needs `Mixed(f)` with `f ~ 0.97` on these meshes) and the behaviour is PINNED
// by a row, so a later well-meaning guard turns a green row red and is read as
// the behaviour change it is. Guarding it is a post-G2 question, beside the
// `w = beta/(beta + alpha d)` fallback.
//
// ===========================================================================
// S7 — NO NEW WALL ARITHMETIC
// ===========================================================================
// The only wall formula reached from here is `ibm::closure(...)`, called once
// per (cell, component) and read for `grad_linear` and `grad_constant` alone.
// `atLinear`/`atConstant` are `div`/`grad`'s (B33/B34) and are not called. No
// alpha, no beta, no `1/d` is re-derived in this file.

#include "../../ibm/cell_type.H"
#include "../../ibm/geometry_view.H"
#include "../../ibm/ghost_cell.H"
#include "host_ghost_cell.H"
#include "robin.H"
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
#include <AMReX_IntVect.H>
#include <AMReX_MFIter.H>
#include <AMReX_MultiFab.H>
#include <AMReX_REAL.H>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace nb = nanobind;

namespace
{

//! The pair's functor: one wall row of `laplacian x ghostCell`, at one cell.
//!
//! Captured BY VALUE into every thread, so every member is a view or a scalar
//! (design §4.3). File-local: nothing outside this TU names it, which is
//! design §1.3's rule 3 — changing one pair rebuilds one translation unit.
struct WallLaplacianGhostCell
{
    static constexpr int stencil_reach = 1;

    amrex::Array4<const std::uint8_t> m; //!< the marker, for the arm gate
    amrex::Array4<const int> row;        //!< (i,j,k) -> rank, -1 off the wall layer
    ibm::IbmGeometryView g;              //!< read at (i, j, k) ONLY — Q34
    ibm::GhostCellView d;                //!< the method's rows, indexed by rank
    ibm::RobinView robin;                //!< the per-patch (alpha, beta, gamma)
    amrex::Real t;                       //!< the time gamma(t) is read at
    amrex::GpuArray<amrex::Real, 3> dx;  //!< the level's cell size

    template<class Sink>
    AMREX_GPU_HOST_DEVICE void operator()(int i, int j, int k, int n, Sink& sink) const
    {
        // The frame calls a functor at WALL cells only, and every WALL cell has
        // a row by construction of the map (`ibm/ghost_cell.cpp`, pass 1).
        const int r = row(i, j, k);
        const int p = g.patch(i, j, k);

        // The closure ONCE per (cell, component), before the arms — v1
        // computes it once per row and reuses it in all six.
        const ibm::WallClosure w =
            ibm::closure(robin.alpha[p], robin.beta[p], robin.gammaAt(p, n, t), d.distance[r]);

        amrex::Real diag = 0.0;
        amrex::Real cacc = 0.0;
        amrex::Real wdon[ibm::K] = {};

        // -------------------------------------------------------------------
        // pass 1 — ACCUMULATE, in v1's loop order: d ascending, +1 then -1
        // -------------------------------------------------------------------
        for (int dd = 0; dd < 3; ++dd)
            for (int s = 0; s < 2; ++s)
            {
                const int step = (s == 0) ? 1 : -1; // H-1: +1 first
                const int ii = i + ((dd == 0) ? step : 0);
                const int jj = j + ((dd == 1) ? step : 0);
                const int kk = k + ((dd == 2) ? step : 0);

                if (m(ii, jj, kk) != ibm::SOLID)
                {
                    // H-2 (one squaring, then one divide) and H-3 (accumulate).
                    diag -= 1.0 / (dx[dd] * dx[dd]);
                }
                else
                {
                    // v1: `step * ctx.normal[:, d] / ctx.dx[d]`, which parses
                    // `((step*n)/dx)`. Kept in v1's spelling.
                    const amrex::Real scale = step * g.normal(i, j, k, dd) / dx[dd];
                    const amrex::Real sg = scale * w.grad_linear; // H-4: once per arm
                    for (int q = 0; q < ibm::K; ++q)
                        wdon[q] += sg * d.weight[r * ibm::K + q];
                    // S2/R1: the BC datum reaches the row through `constant`
                    // and through nothing else. `linear` and `constant` are two
                    // methods with two signatures, so this is a type-level
                    // split and not a convention.
                    cacc += scale * w.grad_constant;
                }
            }

        // -------------------------------------------------------------------
        // pass 2 — EMIT, in v1's slot order: c, slot 0, slots 1..6, slots 7..14
        //
        // The six markers are re-read and `inv` recomputed rather than buffered:
        // both are bit-identical by construction, and ten registers of
        // accumulator are cheaper than six buffered (index, coefficient) pairs.
        // -------------------------------------------------------------------
        sink.constant(cacc);
        sink.linear(i, j, k, diag);
        for (int dd = 0; dd < 3; ++dd)
            for (int s = 0; s < 2; ++s)
            {
                const int step = (s == 0) ? 1 : -1;
                const int ii = i + ((dd == 0) ? step : 0);
                const int jj = j + ((dd == 1) ? step : 0);
                const int kk = k + ((dd == 2) ? step : 0);
                // S3 / Invariant F, enforced by the branch that reads: a SOLID
                // cell is never named. `WallFrameProbe`'s unconditional arms
                // are non-conformant by design and this must not copy them.
                if (m(ii, jj, kk) != ibm::SOLID) sink.linear(ii, jj, kk, 1.0 / (dx[dd] * dx[dd]));
            }
        // The donors were validated fluid by `preprocess`'s Invariant-F pass,
        // and a dead donor (weight exactly 0.0) points at (i, j, k) itself.
        for (int q = 0; q < ibm::K; ++q)
        {
            const int* dn = d.donor + (r * ibm::K + q) * 3;
            sink.linear(dn[0], dn[1], dn[2], wdon[q]);
        }
    }
};

//! The Maker: it holds what the frame refuses to know (design §4.1) and hands
//! the functor a per-box view of it.
struct MakeWallLaplacianGhostCell
{
    using functor_type = WallLaplacianGhostCell;

    const ibm::CellTypeFab* ct;
    const ibm::IbmGeometryFab* g;
    const ibm::GhostCellData* data;
    ibm::RobinView robin;
    amrex::Real t;
    amrex::GpuArray<amrex::Real, 3> dx;

    WallLaplacianGhostCell operator()(const amrex::MFIter& mfi) const
    {
        return WallLaplacianGhostCell {
            ct->const_array(mfi),
            data->row.const_array(mfi),
            ibm::makeGeometryView(*g, mfi),
            ibm::makeGhostCellView(*data),
            robin,
            t,
            dx
        };
    }

    //! S-5 (B30a-R): the guards the frame has no types for, made once per
    //! sweep, before any launch.
    void validate(const char* fn, int ncomp) const
    {
        ibm::requireGeometryLayout(*g, fn);
        ibm::requireGeometryGhosts(fn, *g, ct->nGrowVect().min());
        ibm::requireRobinComponents(fn, robin.ncomp, ncomp);

        // The row map is resolved by `MFIter` LOCAL INDEX beside `phi`, `out`
        // and the marker, exactly like B30a-R's I-2 guard in the frame — and
        // for the same reason: on different grids the pairing is by position,
        // which reads another box's ranks and then another cell's donors.
        if (data->row.boxArray() != ct->boxArray()
            || data->row.DistributionMap() != ct->DistributionMap())
            throw std::runtime_error(
                std::string(fn)
                + ": the ghostCell data was preprocessed on different grids than this sweep's "
                  "cell_type marker — the sweep pairs them by MFIter local index, so a mismatch "
                  "reads another box's row ranks; rebuild the method data for this level's grids"
            );
    }
};

} // namespace

void registerLaplacianGhostCell(nb::module_& m)
{
    // ----------------------------------------------------------------------
    // THE PRODUCTION ENTRY POINT — the canonical twelve (design §4.4)
    //
    // Q39, ruled at B32: a REGISTERED pair carries all twelve `nb::arg`s in
    // this order, with no defaults, `t` included even where the datum is
    // steady. B36's driver calls every pair by keyword from ONE call site, and
    // a pair that dropped an argument would make that site pair-specific —
    // precisely the coupling the registry exists to remove. Underscore-private
    // hooks are exempt by construction: they are never registered, never
    // resolved and never called by a driver.
    // ----------------------------------------------------------------------
    m.def(
        "wall_laplacian_ghost_cell",
        [](amrex::MultiFab& out,
           const amrex::MultiFab& phi,
           const ibm::CellTypeFab& cell_type,
           const ibm::IbmGeometryFab& geom_ibm,
           const ibm::GhostCellData& method_data,
           const ibm::RobinData& robin,
           const amrex::Geometry& geom,
           double t,
           double coeff,
           int ncomp,
           ibm::WallMode mode,
           double constant_scale)
        {
            ibm::applyWall(
                "wall_laplacian_ghost_cell",
                out,
                phi,
                cell_type,
                MakeWallLaplacianGhostCell {
                    &cell_type, &geom_ibm, &method_data, robin.view(), t, geom.CellSizeArray()
                },
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
        "laplacian x ghostCell over every WALL cell of the level: the interior cross "
        "difference with each arm whose face neighbour is SOLID replaced by the flux through "
        "the surface, closed by robin.H's closure(alpha, beta, gamma(t), d) at the image "
        "point's distance. Overwrite assigns, Add accumulates, Assemble raises. SOLID and "
        "FLUID cells are not written at all — the interior sweep owns FLUID and the pin owns "
        "SOLID. constant_scale = 0 drops exactly the BC datum (the matvec of the affine "
        "operator). Bitwise equal to v1's ghost_cell._closed_flux_rows, row for row."
    );

    // ----------------------------------------------------------------------
    // TEST binding (api §4, §10.6) — the same functor, on the HOST, at ONE
    // cell, against a `RecordSink`. Underscore-private, never registered,
    // never on an evaluate path, and exempt from the twelve by Q39.
    //
    // This is the row-level unit test the port gets back after v1 lost its row
    // objects: `([(i, j, k, a), ...], c)` is exactly what a `BandRows` row
    // carried, recovered from the shipped device code rather than from a numpy
    // builder written beside it.
    // ----------------------------------------------------------------------
    m.def(
        "_wall_row_laplacian_ghost_cell",
        [](const ibm::CellTypeFab& cell_type,
           const ibm::IbmGeometryFab& geom_ibm,
           const ibm::GhostCellData& method_data,
           const ibm::RobinData& robin,
           const amrex::Geometry& geom,
           double t,
           int i,
           int j,
           int k,
           int n) -> nb::tuple
        {
            static constexpr const char* FN = "_wall_row_laplacian_ghost_cell";

            ibm::requireGeometryLayout(geom_ibm, FN);
            if (n < 0 || n >= robin.ncomp())
                throw std::runtime_error(
                    std::string(FN) + ": component " + std::to_string(n)
                    + " is outside the Robin table's " + std::to_string(robin.ncomp())
                );

            const int r = ibm::rowAt(method_data, FN, i, j, k);
            if (r < 0)
                throw std::runtime_error(
                    std::string(FN) + ": cell [" + std::to_string(i) + ", " + std::to_string(j)
                    + ", " + std::to_string(k)
                    + "] is not a WALL cell of this level, so ghostCell built no row there; a "
                      "wall row exists only where the marker is WALL"
                );

            amrex::FArrayBox hostG;
            ibm::stageGeometryBox(FN, geom_ibm, amrex::IntVect(i, j, k), hostG);
            amrex::BaseFab<std::uint8_t> hostM;
            ibm::stageMarkerBox(FN, cell_type, amrex::IntVect(i, j, k), hostM);
            const ibm::HostGhostCell hostD(method_data);

            const ibm::IbmGeometryView gv {hostG.const_array()};
            const int patch = gv.patch(i, j, k);
            if (patch < 0 || patch >= robin.npatch())
                throw std::runtime_error(
                    std::string(FN) + ": the geometry says cell [" + std::to_string(i) + ", "
                    + std::to_string(j) + ", " + std::to_string(k) + "] belongs to patch "
                    + std::to_string(patch) + ", but the Robin table has only "
                    + std::to_string(robin.npatch())
                );

            // The rank is the REAL one, out of the level's row map; only its
            // container is host-resident. A one-cell `Array4` is enough because
            // the functor reads `row` at the cell it is called on and nowhere
            // else.
            const int rank = r;
            const amrex::Array4<const int> rowArr(
                &rank, amrex::Dim3 {i, j, k}, amrex::Dim3 {i + 1, j + 1, k + 1}, 1
            );

            const WallLaplacianGhostCell f {
                hostM.const_array(), rowArr, gv, hostD.view(), robin.view(), t, geom.CellSizeArray()
            };
            ibm::RecordSink rec;
            f(i, j, k, n, rec);

            if (rec.overflow)
                throw std::runtime_error(
                    std::string(FN) + ": the row emitted more than RecordSink::capacity = "
                    + std::to_string(ibm::RecordSink::capacity)
                    + " linear entries and was truncated; grow the sink"
                );

            nb::list entries;
            for (int e = 0; e < rec.count; ++e)
                entries.append(nb::make_tuple(
                    rec.entries[e].i, rec.entries[e].j, rec.entries[e].k, rec.entries[e].a
                ));
            return nb::make_tuple(entries, rec.c);
        },
        nb::arg("cell_type"),
        nb::arg("geom_ibm"),
        nb::arg("method_data"),
        nb::arg("robin"),
        nb::arg("geom"),
        nb::arg("t"),
        nb::arg("i"),
        nb::arg("j"),
        nb::arg("k"),
        nb::arg("n"),
        "TEST ONLY (B32). laplacian x ghostCell's row at one WALL cell, computed on the HOST "
        "against a RecordSink: returns ([(i, j, k, a), ...], c) — the ordered linear entries "
        "and the constant. The order is v1's: the diagonal, then the fluid arms in slot order, "
        "then the eight trilinear donors. Raises if the cell is not a WALL cell."
    );
}
