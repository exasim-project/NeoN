// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

// `grad x ghostCell` — THE THIRD REAL `(operator, method)` PAIR, and the last
// of the ported three (B34).
//
// design §6's one place an operator and a method meet, compiled. The frame
// (`wall_apply.H`), the sinks (`wall_value.H`), the BC transport
// (`robin_data.H`), the wall formula (`robin.H`), the host staging
// (`host_ghost_cell.H`, `wall_stage.H`) and the method's data
// (`ibm/ghost_cell.H`) all already exist and none of them moves here: what this
// file adds is the ASSEMBLY — which cells a row names, in which order, with
// which coefficients.
//
// ===========================================================================
// THE SOURCE OF TRUTH, AND ITS RELATION TO `div_ghost_cell.cpp`
// ===========================================================================
// `src/blockamr/schemes/boundary/ghost_cell.py::_face_balance_rows` — **the
// same function `div` calls** — reached from `GhostCellGrad.rows`
// (`ghost_cell.py:426-453`). THREE substitutions turn `div` into `grad`, and
// there are no others:
//
//        argument        div (B33)                       grad (B34)
//        axes            (0, 1, 2)                       (0,)          <- H-9
//        flux            _band_face_flux(...)            ones == 1.0
//        weight_self     _face_weights(interior, flux)   0.5 * ones
//
// `_face_weights` is not called, `_band_face_flux` is not called, and
// `GhostCellGrad.__init__` stores `interior_scheme` and NEVER reads it. So this
// pair reads no face field, needs no face-value selector, and appends nothing
// to the canonical twelve (see `registerGradGhostCell` below); and since
// `SCHEME_REGISTRY["grad"]` has exactly one entry (`central`, width 1) there is
// no D1-style degrade to compose either.
//
// A face whose neighbour is FLUID keeps the central difference exactly; a face
// whose neighbour is not fluid keeps that SAME formula and substitutes, for the
// neighbour it cannot read, the field the wall condition extrapolates to that
// cell centre — `closure.at(d_G)` at `d_G = s_P + step*dx_0*n̂_0`.
//
// Read together with its consumer (`band_table.cpp:673-694`), a v1 wall row is
// ONE CONSTANT THEN FIFTEEN SLOTS. For `axes = (0,)` only these are ever
// touched, in this exact order:
//
//   emission            slot    value
//   constant(c)          --     sum over WALL x-faces of nb_part*atConstant(d_G)
//   linear(P, .)          0     sum over BOTH x-faces of scale*0.5      <-- H-10
//   linear(P +- e_0, .) 1..2    0.0 + nb_part, for a FLUID face only    <-- H-6
//   linear(donor_q, .)  7..14   sum over WALL x-faces of (nb_part*atLinear)*w_q
//
// Slots 3..6 — the `+-y` and `+-z` neighbours — are NEVER written: `_blank`
// leaves them at `a = 0.0` with the stencil pointing at the TARGET, and v1's
// own liveness rule (a slot pointing at the target is dead) drops them. The
// pair emits nothing for them. `nnz` is still `STRIDE = 15`: the row DECLARES
// fifteen slots and USES eleven. The loop order is `for step in (1, -1)` —
// **+1 first**.
//
// Maximum row size `1 + 2 + 8 = 11`, well inside `RecordSink::capacity = 64`.
//
// ---------------------------------------------------------------------------
// THE TRANSCRIPTION HAZARDS, EACH MARKED AT ITS SITE BELOW
// ---------------------------------------------------------------------------
// H-1  v1 iterates `for step in (1, -1)`. api §5.3's published sketch writes
//      `-1` first, which is a different accumulation order for the donor sums
//      and the constant and a different EMISSION order for the two face slots.
//      Measured pre-build: 9 of 10 configurations — invisible only on G1, where
//      every row has exactly one fluid x-arm so no order is observable.
// H-3' **the diagonal is NOT gated on the face being fluid.** v1's mask there
//      is `ctx.fluid` — a property of the ROW, which the frame has already
//      established by calling us at a WALL cell — and not of the face. Copying
//      the laplacian's fluid-arm gate onto it is caught on 8 of 10, and is
//      INVISIBLE on G2 and G8, the two configurations with no wall arm at all.
// H-4' `const Real sg = nbp * lin;` ONCE PER FACE, then `wdon[q] += sg * w`.
//      The algebraically equal `nbp * (lin * w_q)` is a different number — but
//      only on ONE of the ten configurations, G10, and 194 rows of it. On a
//      dyadic grid `nb_part = +-0.5/dx` is exactly a power of two, so the
//      association is exact and every dyadic configuration is blind to it. G10
//      (a non-dyadic `prob_lo`/`prob_hi`) carries that whole column alone.
// H-5  **the one documented departure from v1's association, inherited from
//      B32/B33.** v1 folds `coeff` into `scale` BEFORE every product
//      (`coeff * step * flux / dx`); the frame multiplies the finished SUM
//      (`wall_apply.H:216`). Measured this session, rows that move between the
//      two placements:
//
//          configuration            coeff 1.0  2.0  0.5   3.0   0.1
//          G1  plane-x Dirichlet (256)      0    0    0     0     0
//          G6  cylinder Mixed    (320)      0    0    0    64    64
//          G10 non-dyadic grid   (352)      0    0    0   352   215
//
//      i.e. identical iff `coeff` is a power of two — and `exp.grad(field)`
//      exposes no `coeff` at all, `Grad.__init__`'s default being `1.0`, so
//      H-5 is INERT on every reachable v1 grad term. RECORDED, not fixed:
//      fixing it edits `wall_apply.H`'s contract for the laplacian and div
//      pairs too. At `coeff = 1.0` the right spelling is `step * fl / dx[dd]`,
//      because v1 parses `((coeff*step)*flux)/dx`.
// H-6  v1's `_blank` allocates `a = np.zeros(...)`, so the face-neighbour slot
//      value it ships is `0.0 + nb_part` and not `nb_part`. For `grad`,
//      `nb_part` is `+-0.5/dx` and never `+-0.0`, so the raw emission is a
//      measured CONTROL here (0 of 10) — the accumulation is transcribed anyway,
//      because v1's SHAPE is what is being ported and not v1's arithmetic on one
//      input. `div` needs it for real: 960 of 3 232 rows.
// H-7  `nbp = sc * (1.0 - ws)` and never `sc - slf`. A measured CONTROL (0 of
//      10) because `ws` is exactly `0.5` here — recorded as a control precisely
//      so a future non-trivial face weight is not assumed safe.
// H-9  **the axis collapse, and the copy-paste defect this file exists to
//      resist.** `div_ghost_cell.cpp`'s functor is `for (int dd = 0; dd < 3;
//      ++dd)`; this one is one axis. See the note at `dd` below for WHY v1 does
//      it — the reason is the row format and not an optimisation. Copying div's
//      axis loop wholesale (`axes-all`) is caught on 10 of 10 configurations and
//      on every one of the 3 136 wall rows; so is the subtler half (`arms-six`),
//      which restricts the accumulation to axis 0 but keeps div's SIX-arm
//      emission loop and so emits `+0.0` at the `+-y`/`+-z` neighbours.
// H-10 **the diagonal is bitwise `+0.0` on every row, and it is still
//      emitted.** For `axes = (0,)` slot 0 accumulates exactly twice:
//      `(0.0 + (coeff*(+1)*1.0/dx_0)*0.5) + ((coeff*(-1)*1.0/dx_0)*0.5)`.
//      `coeff*(-1)` is exactly `-(coeff*1)`, IEEE multiplication and division
//      are sign-symmetric, and `x + (-x)` is `+0.0` in round-to-nearest for
//      every finite `x`. Measured: rows whose diagonal is not bitwise `+0` —
//      **0 of 3 136**. Three consequences, all load-bearing:
//        1. the slot is still emitted (v1's row carries slot 0 with
//           `stencil[0] = target` and `a[0] = +0.0`, and a sweep reading it
//           multiplies `phi(P)` by `+0.0` and adds it); `diag-dropped` is caught
//           on 10/10 and on all 3 136 rows;
//        2. its SIGN is pinned; `diag-neg-zero` is caught on 10/10 and all
//           3 136 rows, so computing it as `-(slf_- + slf_+)` ships a different
//           row;
//        3. the test-side canonicalisation must be STRUCTURAL. B32's `_v1_row`
//           drops a slot whose bits are `0`; for `grad` that rule would discard
//           21 688 of 32 939 live entries (65.8 %) and the diagonal of EVERY
//           row — every one of the 3 136 carries at least one live `+-0.0`.
//
// ===========================================================================
// FLOATING POINT (review.md §4 Q36, ruled at B32; load-bearing again here)
// ===========================================================================
// This TU is on `CMakeLists.txt`'s per-file `--fmad=false` /
// `-Xcompiler=-ffp-contract=off` list, and it MUST be. `robin.H` is a header,
// so its `beta - alpha*distance` AND its `value + d*grad` (`atLinear`,
// `atConstant` — this file is their SECOND caller, `div` being the first) are
// inlined with THIS TU's flags. The new contraction sites this file introduces
// on its own account are `diag += slf`, `arm[s] += nbp`, `wdon[q] += sg * w`,
// `cacc += nbp * atConstant(dG)` and `dG = s_P + step*dx*n̂`. The flag is
// therefore load-bearing for B34's own bitwise bar and not merely inherited.
//
// The list is kept honest mechanically: a row in
// `test_ibm_laplacian_ghost_cell.py` parses that `set_source_files_properties`
// call and asserts that every includer of `robin.H` or `ibm/ghost_cell.H` under
// this tree appears in it. This file enters that contract for free.
//
// ===========================================================================
// Q34 — WHERE THE GEOMETRY IS READ
// ===========================================================================
// At the target cell and nowhere else: `g.patch(i, j, k)`, `g.sdf(i, j, k)` and
// `g.normal(i, j, k, 0)`. There is no face-centred read at all here, so
// `stencil_reach = 1` is honest and this pair's `validate` has no flux clause.
// Not an inspection claim: `test_ibm_wall_functors.py` perturbs the geometry fab
// at a neighbour index and requires the row to be bit-identical (on the
// two-patch geometry, where a neighbour read would also pick up the wrong
// `alpha`/`beta`), and the `normal-nb` mutant is caught on 8 of 10
// configurations.
//
// ===========================================================================
// THE POLE IS NOT GUARDED HERE EITHER (review.md §4 Q46)
// ===========================================================================
// `robin.H`'s `den = beta - alpha*d` reaches exactly zero for `Mixed(f)` with
// `d = (1 - f)/f`, and v1 divides anyway and returns `+-inf`. This pair adds NO
// raise: a raise where v1 returns a number is a behaviour change, and this
// task's whole claim is v1<->v2 bitwise parity. The behaviour — the signs of the
// live donors included — is PINNED by a row, so a later well-meaning guard turns
// a green row red and is read as the behaviour change it is. That row must run
// on a configuration measured to HAVE a wall arm: on G2 and G8 the closure never
// enters the row and the pole would be unaskable.
//
// ===========================================================================
// `ncomp > 1` IS REFUSED (api §9, review.md §4 Q56(c))
// ===========================================================================
// v1's `_check_grad_ncomp` (`ghost_cell.py:456-474`) is the FIRST statement of
// `rows()` and raises:
//
//     grad x ghostCell needs a one-component field, but '<field>' has
//     ncomp = <n>: the band row applies one coefficient list to every
//     component, while the gradient's component n is the difference along axis
//     n. Expressing that needs a per-component row, which the v1 row format
//     (plans/IBM/row-contract.md §2) does not have.
//
// The compiled pair has no field name — it has `ncomp` and its own entry-point
// name — so it raises v1's sentence with the ENTRY POINT in place of the field
// name (the B31 Invariant-F precedent), from `Maker::validate`, before any
// launch. The exception TYPE differs (`NotImplementedError` on v1,
// `std::runtime_error` -> `RuntimeError` on v2) and that gap is owed to B36
// beside B31's; it is cheap to live with, because `NotImplementedError` IS a
// `RuntimeError`, so one `pytest.raises(RuntimeError)` binds both surfaces
// today.
//
// ===========================================================================
// S7 — NO NEW WALL ARITHMETIC
// ===========================================================================
// The only wall formula reached from here is `ibm::closure(...)`, called once
// per (cell, component), and its two `at` reads. `closure`, `atLinear` and
// `atConstant` are CALLED, never re-derived: no `alpha`, no `beta`, no `1/d`
// appears in this file.

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

//! The pair's functor: one wall row of `grad x ghostCell`, at one cell.
//!
//! Captured BY VALUE into every thread, so every member is a view or a scalar
//! (design §4.3). File-local: nothing outside this TU names it, which is
//! design §1.3's rule 3 — changing one pair rebuilds one translation unit.
//!
//! Seven members — the canonical twelve's worth and nothing else: `grad` reads
//! no face flux and no interior scheme.
struct WallGradGhostCell
{
    static constexpr int stencil_reach = 1;

    amrex::Array4<const std::uint8_t> m; //!< the marker, for the face gate
    amrex::Array4<const int> row;        //!< (i,j,k) -> rank, -1 off the wall layer
    ibm::IbmGeometryView g;              //!< read at (i, j, k) ONLY — Q34
    ibm::GhostCellView d;                //!< the method's rows, by rank
    ibm::RobinView robin;                //!< the per-patch (alpha, beta, gamma)
    amrex::Real t;                       //!< the time gamma(t) is read at
    amrex::GpuArray<amrex::Real, 3> dx;  //!< the level's cell size

    template<class Sink>
    AMREX_GPU_HOST_DEVICE void operator()(int i, int j, int k, int n, Sink& sink) const
    {
        // The frame calls a functor at WALL cells only, and every WALL cell has
        // a row by construction of the map (`ibm/ghost_cell.cpp`, pass 1).
        const int r = row(i, j, k);
        const int p = g.patch(i, j, k); // Q34: geometry at (i, j, k) ONLY
        const amrex::Real s_P = g.sdf(i, j, k);

        // The closure ONCE per (cell, component), before the faces — v1
        // computes it once per row and reuses it in both.
        const ibm::WallClosure w =
            ibm::closure(robin.alpha[p], robin.beta[p], robin.gammaAt(p, n, t), d.distance[r]);

        // v1 passes `flux = ones` and `weight_self = 0.5 * ones` into the SAME
        // `_face_balance_rows` that `div` uses (`ghost_cell.py:438-453`). Named
        // here so the transcription reads against v1 line by line; both fold.
        // With `flux` constant, v1's `face` index (`1` for `step = +1`, `0` for
        // `step = -1`) selects nothing — which is why the `face-index` mutant is
        // a measured control here and a real defect for `div`.
        constexpr amrex::Real fl = 1.0;
        constexpr amrex::Real ws = 0.5;
        // H-9: v1's `axes = (0,)`. The row format applies ONE coefficient list
        // to every component (`out(P, n) = sum_k a_k phi(s_k, n)`, row-contract
        // §2), and the gradient's component n is the difference along axis n —
        // which needs a DIFFERENT stencil per component, and the format does not
        // have one. So v1 expresses only `n = 0` and refuses `ncomp > 1` (see
        // `validate`). A reader who "fixes" the missing axes has silently
        // changed what the operator means: `axes-all` is caught on 10 of 10
        // configurations and on all 3 136 rows.
        constexpr int dd = 0;

        amrex::Real diag = 0.0;
        amrex::Real cacc = 0.0;
        amrex::Real wdon[ibm::K] = {};
        // H-6's shape, inherited from B33: v1's slots 1..6 are ZERO-INITIALISED
        // and written once, so the shipped coefficient is `0.0 + nb_part`. For
        // `grad` `nb_part` is never `+-0.0` (it is `+-0.5/dx`), so the raw
        // emission is a measured CONTROL here — the accumulation is transcribed
        // because v1's SHAPE is what is being ported, not v1's arithmetic on one
        // input.
        amrex::Real arm[2] = {};

        // -------------------------------------------------------------------
        // pass 1 — ACCUMULATE, in v1's loop order: +1 then -1
        // -------------------------------------------------------------------
        for (int s = 0; s < 2; ++s)
        {
            const int step = (s == 0) ? 1 : -1; // H-1: +1 first
            const int ii = i + step;

            // H-5: v1 folds `coeff` in here; the frame folds it into the sum.
            // Identical at a power-of-two `coeff`, and `exp.grad` exposes none.
            const amrex::Real sc = step * fl / dx[dd];
            const amrex::Real slf = sc * ws;
            const amrex::Real nbp = sc * (1.0 - ws); // H-7, NOT `sc - slf`

            // H-3': BOTH faces, fluid or wall. v1's mask is the ROW's
            // (`ctx.fluid`), which the frame has already established. This is
            // NOT the laplacian's fluid-arm gate. H-10: the two contributions
            // cancel exactly, so `diag` is bitwise `+0.0` on every row — and is
            // emitted all the same.
            diag += slf;

            if (m(ii, j, k) != ibm::SOLID)
            {
                // H-6: `0.0 + nbp`.
                arm[s] += nbp;
            }
            else
            {
                // The ghost centre's own signed distance from the surface along
                // THIS cell's normal — v1's
                // `ctx.sdf + step * ctx.dx[d] * ctx.normal[:, d]`, which parses
                // `s_P + ((step*dx)*n̂_0)` in both languages. An FMA site of this
                // file's own, covered by `--fmad=false`.
                const amrex::Real dG = s_P + step * dx[dd] * g.normal(i, j, k, dd);
                // S7: the closure read a third way, not a new approximation.
                const amrex::Real lin = w.atLinear(dG);
                const amrex::Real sg = nbp * lin; // H-4': once per face
                for (int q = 0; q < ibm::K; ++q)
                    wdon[q] += sg * d.weight[r * ibm::K + q];
                // S2/R1: the BC datum reaches the row through `constant` and
                // through nothing else. `linear` and `constant` are two methods
                // with two signatures, so this is a type-level split and not a
                // convention.
                cacc += nbp * w.atConstant(dG);
            }
        }

        // -------------------------------------------------------------------
        // pass 2 — EMIT, in v1's slot order: c, slot 0, slots 1..2, slots 7..14
        // -------------------------------------------------------------------
        sink.constant(cacc);
        // H-10: bitwise `+0.0` on 3 136 of 3 136 rows, and NOT optional — v1's
        // row carries the slot and a sweep reading it adds `+0.0 * phi(P)`.
        sink.linear(i, j, k, diag);
        for (int s = 0; s < 2; ++s)
        {
            const int step = (s == 0) ? 1 : -1;
            const int ii = i + step;
            // S3 / Invariant F, enforced by the branch that reads: a SOLID cell
            // is never named. `WallFrameProbe`'s unconditional arms are
            // non-conformant by design and this must not copy them.
            if (m(ii, j, k) != ibm::SOLID) sink.linear(ii, j, k, arm[s]);
        }
        // v1's slots 3..6 (the +-y, +-z neighbours) are NEVER written by
        // `axes = (0,)`: they keep `a = 0.0` with the stencil on the target, so
        // v1's own liveness rule drops them and nothing is emitted for them.
        // Emitting them with `+0.0` is the `arms-six` mutant — 10 of 10, all
        // 3 136 rows.
        //
        // The donors were validated fluid by `preprocess`'s Invariant-F pass,
        // and a dead donor (weight exactly 0.0) points at (i, j, k) itself. All
        // eight are emitted, including on the 1 579 of 3 136 rows where every
        // one of them is `+0.0` because the row has no wall arm at all.
        for (int q = 0; q < ibm::K; ++q)
        {
            const int* dn = d.donor + (r * ibm::K + q) * 3;
            sink.linear(dn[0], dn[1], dn[2], wdon[q]);
        }
    }
};

//! The Maker: it holds what the frame refuses to know (design §4.1) and hands
//! the functor a per-box view of it.
struct MakeWallGradGhostCell
{
    using functor_type = WallGradGhostCell;

    const ibm::CellTypeFab* ct;
    const ibm::IbmGeometryFab* g;
    const ibm::GhostCellData* data;
    ibm::RobinView robin;
    amrex::Real t;
    amrex::GpuArray<amrex::Real, 3> dx;

    WallGradGhostCell operator()(const amrex::MFIter& mfi) const
    {
        return WallGradGhostCell {
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
    //!
    //! There is deliberately no ghost-width check beyond the frame's:
    //! `stencil_reach = 1`, and the only non-target reads are `m(i +- 1, j, k)`
    //! and the donors. There is no face-flux clause either, because there is no
    //! face field — that is `div`'s, and the difference is the whole of §H-9.
    void validate(const char* fn, int ncomp) const
    {
        ibm::requireGeometryLayout(*g, fn);
        ibm::requireGeometryGhosts(fn, *g, ct->nGrowVect().min());
        ibm::requireRobinComponents(fn, robin.ncomp, ncomp);

        // The row map is resolved by `MFIter` LOCAL INDEX beside `phi`, `out`
        // and the marker, exactly like B30a-R's I-2 guard in the frame — and for
        // the same reason: on different grids the pairing is by position, which
        // reads another box's ranks and then another cell's donors.
        if (data->row.boxArray() != ct->boxArray()
            || data->row.DistributionMap() != ct->DistributionMap())
            throw std::runtime_error(
                std::string(fn)
                + ": the ghostCell data was preprocessed on different grids than this sweep's "
                  "cell_type marker — the sweep pairs them by MFIter local index, so a mismatch "
                  "reads another box's row ranks; rebuild the method data for this level's grids"
            );

        // v1's `_check_grad_ncomp` (`ghost_cell.py:456`), transcribed. It is the
        // FIRST statement of v1's `rows()`, so nothing is built before it; here
        // it is the last guard before the first launch, which is the same place
        // in this architecture. v1's sentence, with the ENTRY POINT in place of
        // the field name the compiled pair does not have (api §9).
        if (ncomp != 1)
            throw std::runtime_error(
                std::string(fn)
                + ": grad x ghostCell needs a one-component field, but this sweep has ncomp = "
                + std::to_string(ncomp)
                + ": the band row applies one coefficient list to every component, while the "
                  "gradient's component n is the difference along axis n. Expressing that needs "
                  "a per-component row, which the row format (plans/IBM/row-contract.md §2) does "
                  "not have."
            );
    }
};

} // namespace

void registerGradGhostCell(nb::module_& m)
{
    // ----------------------------------------------------------------------
    // THE PRODUCTION ENTRY POINT — EXACTLY the canonical twelve (design §4.4).
    //
    // Q39, ruled at B32: a REGISTERED pair carries all twelve `nb::arg`s in
    // that order, with no defaults, `t` included even where the datum is
    // steady. Q29(f) makes the twelve a MINIMUM and B33 exercised that for
    // `div`'s three fluxes and its face-value selector — `grad` needs NONE of
    // it, and that is measured rather than preferred: `GhostCellGrad.rows`
    // passes `flux = ones` and `weight_self = 0.5 * ones` and never reads
    // `self.interior`, and `SCHEME_REGISTRY["grad"]` has exactly one entry, so
    // there is no scheme bit to transport even in principle.
    //
    // B32's conformance row asserts the twelve as a PREFIX over every registered
    // `wall_*` attribute, so this pair enters that contract for free;
    // `test_ibm_grad_ghost_cell.py` adds the statement from the other side —
    // that they are ALL of them, `len(params) == 12`.
    // ----------------------------------------------------------------------
    m.def(
        "wall_grad_ghost_cell",
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
                "wall_grad_ghost_cell",
                out,
                phi,
                cell_type,
                MakeWallGradGhostCell {
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
        "grad x ghostCell over every WALL cell of the level: the width-1 central difference "
        "along axis 0, (phi_f^+ - phi_f^-)/dx_0, with the face whose neighbour is SOLID "
        "reading, instead of that neighbour, the field the wall condition extrapolates to its "
        "cell centre — robin.H's closure(alpha, beta, gamma(t), d).at(d_G) at "
        "d_G = sdf + step*dx_0*n_0. ONE axis, whatever the component: the band row applies one "
        "coefficient list to every component while the gradient's component n is the difference "
        "along axis n, so ncomp > 1 is REFUSED before any launch, in v1's own sentence. "
        "Overwrite assigns, Add accumulates, Assemble raises. SOLID and FLUID cells are not "
        "written at all. constant_scale = 0 drops exactly the BC datum. Bitwise equal to v1's "
        "ghost_cell._face_balance_rows at axes=(0,), flux=1, weight_self=0.5, row for row."
    );

    // ----------------------------------------------------------------------
    // TEST binding (api §4, §10.6) — the same functor, on the HOST, at ONE
    // cell, against a `RecordSink`. Underscore-private, never registered, never
    // on an evaluate path, and exempt from the twelve by Q39.
    // ----------------------------------------------------------------------
    m.def(
        "_wall_row_grad_ghost_cell",
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
            static constexpr const char* FN = "_wall_row_grad_ghost_cell";

            ibm::requireGeometryLayout(geom_ibm, FN);
            // The `ncomp > 1` refusal, mirrored off the Robin table's own width
            // because the row hook has no `ncomp` argument. Same sentence as
            // `Maker::validate`, so the two surfaces cannot drift.
            if (robin.ncomp() != 1)
                throw std::runtime_error(
                    std::string(FN)
                    + ": grad x ghostCell needs a one-component field, but this sweep has ncomp = "
                    + std::to_string(robin.ncomp())
                    + ": the band row applies one coefficient list to every component, while the "
                      "gradient's component n is the difference along axis n. Expressing that "
                      "needs a per-component row, which the row format "
                      "(plans/IBM/row-contract.md §2) does not have."
                );
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

            const amrex::IntVect iv(i, j, k);
            amrex::FArrayBox hostG;
            ibm::stageGeometryBox(FN, geom_ibm, iv, hostG);
            amrex::BaseFab<std::uint8_t> hostM;
            ibm::stageMarkerBox(FN, cell_type, iv, hostM);
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

            const WallGradGhostCell f {
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
        "TEST ONLY (B34). grad x ghostCell's row at one WALL cell, computed on the HOST against "
        "a RecordSink: returns ([(i, j, k, a), ...], c) — the ordered linear entries and the "
        "constant. The order is v1's: the diagonal, then the fluid x faces in slot order, then "
        "the eight trilinear donors — at most 1 + 2 + 8 = 11 entries, because grad differences "
        "ONE axis. Raises if the cell is not a WALL cell, or if the Robin table is wider than "
        "one component."
    );
}
