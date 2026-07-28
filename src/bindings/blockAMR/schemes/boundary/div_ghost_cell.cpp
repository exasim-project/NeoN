// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

// `div x ghostCell` — THE SECOND REAL `(operator, method)` PAIR (B33).
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
// `src/blockamr/schemes/boundary/ghost_cell.py::_face_balance_rows`, called
// from `GhostCellDiv.rows` with `axes = (0, 1, 2)`,
// `flux = _band_face_flux(term.coefficient, lev, ctx.band)` and
// `weight_self = _face_weights(self.interior, flux)`. Its row is
// `coeff * sum_d (f_d^+ phi_f^+ - f_d^- phi_f^-) / dx_d`: a face whose
// neighbour is FLUID keeps the interior scheme's own width-1 face value
// exactly, and a face whose neighbour is not fluid keeps that SAME formula and
// substitutes, for the neighbour it cannot read, the field the wall condition
// extrapolates to that cell centre — `closure.at(d_G)` at
// `d_G = s_P + step*dx_d*n̂_d`.
//
// Substituting the neighbour's CELL value and not the surface value `phi_w` is
// the choice that keeps the interior formula's telescoping intact; the literal
// reading of design §9 would halve the gradient at the wall, which v1's own
// module docstring (`ghost_cell.py:50-57`) warns about and which the
// `at-wall-value` mutant models (caught on 10 of 10 configurations).
//
// Read together with its consumer (`band_table.cpp:673-694`), a v1 wall row is
// ONE CONSTANT THEN FIFTEEN SLOTS, in this exact order:
//
//   emission            slot    value
//   constant(c)          --     sum over WALL faces of  nb_part * atConstant(d_G)
//   linear(P, .)          0     sum over ALL SIX faces of scale * weight_self
//   linear(P +- e_d, .) 1..6    0.0 + nb_part, for a FLUID face only   <-- H-6
//   linear(donor_q, .)  7..14   sum over WALL faces of (nb_part*atLinear(d_G))*w_q
//
// and the loop order is `for d in 0,1,2: for step in +1, -1` — **+1 first**,
// with `+1` taking the cell's HIGH face (`face = 1`).
//
// `_band_face_flux` fills `flux[.., d, 0]` from `arr[low]` and `flux[.., d, 1]`
// from `arr[low + e_d]`, reading component 0 of `face_field[lev][d].mf` — i.e.
// exactly `fx(i, j, k)` and `fx(i + 1, j, k)` of `stencil_kernels.cpp`'s
// `divUpwindCell`. Same array, same indices; there is no re-derivation here.
//
// ---------------------------------------------------------------------------
// THE TRANSCRIPTION HAZARDS, EACH MARKED AT ITS SITE BELOW
// ---------------------------------------------------------------------------
// H-1  v1 iterates `for step in (1, -1)`. api §5.3's published sketch writes
//      `-1` first, which is a different accumulation order for the diagonal,
//      the donor sums and the constant, and a different EMISSION order for the
//      six face slots. Measured pre-build: 10 of 10 configurations.
// H-2' the axis loop ascending. Reversed: 10 of 10.
// H-3' **the diagonal is NOT gated on the face being fluid.** v1's mask there
//      is `ctx.fluid` — a property of the ROW, which the frame has already
//      established by calling us at a WALL cell — and not of the face. Copying
//      the laplacian's fluid-arm gate onto it is the single most likely
//      copy-paste defect in this file: `self-gated`, caught 10 of 10,
//      192-448 rows.
// H-4' `const Real sg = nbPart * lin;` ONCE PER FACE, then `wdon[q] += sg * w`.
//      The algebraically equal `nbPart * (lin * w_q)` is a different number:
//      caught on 7 of 10, and INVISIBLE on D1 (weights exactly 0/1), on D2
//      (uniform flux on a dyadic grid) and on D4 (Neumann: `atLinear == 1.0`).
// H-5  **the one documented departure from v1's association, inherited from
//      B32.** v1 folds `coeff` into `scale` BEFORE every product
//      (`coeff * step * flux / dx`); the frame multiplies the final SUM
//      (`wall_apply.H:216`). The two agree bitwise whenever `coeff` is a power
//      of two — `1.0` on every acceptance row, every rung and every D1 row.
//      RECORDED, not fixed: fixing it edits `wall_apply.H`'s contract for every
//      pair. At `coeff = 1.0` the right spelling is `step * fl / dx[dd]`,
//      because v1 parses `((coeff*step)*flux)/dx`.
// H-6  **the one this session found, and the one it pays for.** `_blank`
//      allocates `a = np.zeros(...)`, so v1's face-neighbour slot value is
//      `0.0 + nb_part`, and IEEE says `0.0 + (-0.0) = +0.0`. A functor that
//      emitted `nb_part` RAW would ship `-0.0` where v1 has `+0.0`.
//      `nb_part = scale*(1 - w)` is `-0.0` exactly when the face flux is `+-0.0`
//      and the face-value rule puts the whole weight on the neighbour with
//      `step = -1` — which a rigid-rotation velocity produces by the plane.
//      Measured: 960 of 3 232 wall rows (D3 320, D4 320, D9 320) break without
//      the `arm[]` bank, and the defect is INVISIBLE on the other seven
//      configurations, including every uniform-flux and every skew-flux one.
//      That is why this file accumulates into `arm[slot]` instead of emitting.
//      The same laundering does NOT apply to `diag`, `wdon[]` or `cacc`: those
//      are accumulators on both sides, both starting at `+0.0`.
// H-7  `nbPart = sc * (1.0 - ws)` and never `sc - slf`. A measured CONTROL
//      (0 of 10) because `ws` is exactly one of `{0, 0.5, 1}` here — recorded as
//      a control precisely so a future non-trivial face weight is not assumed
//      safe.
// H-8  `weightSelf`'s comparison is `>=`, never `>`. Also a measured control
//      (0 of 10), and provably one: the two differ only at `f = +-0.0`, where
//      `scale` is `+-0.0` and both `scale*w` and `scale*(1-w)` carry `scale`'s
//      own sign whichever weight is chosen. It is spelled v1's way anyway.
//
// ===========================================================================
// FLOATING POINT (review.md §4 Q36, ruled at B32; load-bearing again here)
// ===========================================================================
// This TU is on `CMakeLists.txt`'s per-file `--fmad=false` /
// `-Xcompiler=-ffp-contract=off` list, and it MUST be. `robin.H` is a header,
// so its `beta - alpha*distance` AND its `value + d*grad` (`atLinear`,
// `atConstant` — this file is their first caller) are inlined with THIS TU's
// flags. The new contraction sites this file introduces on its own account are
// `diag += slf`, `arm[slot] += nbp`, `wdon[q] += sg * w`,
// `cacc += nbp * atConstant(dG)` and `dG = s_P + step*dx*n̂`. The flag is
// therefore load-bearing for B33's own bitwise bar and not merely inherited.
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
// `g.normal(i, j, k, dd)`. Reading `f[dd](i + 1, ...)` is a FACE array at the
// cell's own high face and not a neighbour's geometry, which is why
// `stencil_reach = 1` stays honest and why no ghost width is asked of the flux
// (`validate` says so at its site). Not an inspection claim:
// `test_ibm_wall_functors.py` perturbs the geometry fab at a neighbour index and
// requires the row to be bit-identical, and the `normal-nb` mutant is caught on
// 10 of 10 configurations.
//
// ===========================================================================
// THE POLE IS NOT GUARDED HERE EITHER (review.md §4 Q46)
// ===========================================================================
// `robin.H`'s `den = beta - alpha*d` reaches exactly zero for `Mixed(f)` with
// `d = (1 - f)/f`, and v1 divides anyway and returns `+-inf`. This pair adds NO
// raise: a raise where v1 returns a number is a behaviour change, and this
// task's whole claim is v1<->v2 bitwise parity. The behaviour — the signs of the
// live donors included — is PINNED by a row, so a later well-meaning guard turns
// a green row red and is read as the behaviour change it is.
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
#include "robin.H"
#include "robin_data.H"
#include "wall_apply.H"
#include "wall_stage.H"
#include "wall_value.H"

#include <nanobind/nanobind.h>

#include <AMReX_Array4.H>
#include <AMReX_BaseFab.H>
#include <AMReX_BoxArray.H>
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
#include <vector>

namespace nb = nanobind;

namespace ibm
{

//! Which width-1 face value the INTERIOR scheme uses, as one argument.
//!
//! v1 constructs its boundary scheme with the interior scheme object
//! (`GhostCellDiv.__init__(interior_scheme)`) and reads exactly one bit off it —
//! `type == "Linear"`. The compiled pair cannot see a Python object, so the bit
//! travels as this enum. The mapping (v1 `_face_weights`, plus D1's degrade
//! rule) is:
//!
//!     v1 `Div` scheme     stencil_width   DivFaceValue
//!     linear  (Linear)          1         Central
//!     upwind  (Upwind)          1         Upwind
//!     vanLeer (VanLeer)         2         Upwind   <- the D1 degrade
//!     quick   (QUICK)           2         Upwind   <- the D1 degrade
//!
//! A width-2 scheme degrades to first-order upwind inside `band(w)` and only
//! there, because a wider stencil reaches through the solid where there is
//! nothing valid to read. That degrade is measured, not asserted: D9 (`vanLeer`)
//! and D10 (`quick`) reproduce their width-1 upwind sibling's wall row bitwise.
//!
//! An enum and not a `bool`, and not two registered entry points: the registry
//! key is `(operator, method)` and B36 resolves ONE name per pair, so two names
//! would put scheme-specific dispatch back at the driver — the coupling Q39
//! exists to remove. **B36 owns the call-site mapping**; B33 ships no helper for
//! it, because a helper with no caller is speculative code.
enum class DivFaceValue : int
{
    Central = 0,
    Upwind = 1
};

} // namespace ibm

namespace
{

//! The pair's functor: one wall row of `div x ghostCell`, at one cell.
//!
//! Captured BY VALUE into every thread, so every member is a view or a scalar
//! (design §4.3). File-local: nothing outside this TU names it, which is
//! design §1.3's rule 3 — changing one pair rebuilds one translation unit.
struct WallDivGhostCell
{
    static constexpr int stencil_reach = 1;

    amrex::Array4<const std::uint8_t> m; //!< the marker, for the face gate
    amrex::Array4<const int> row;        //!< (i,j,k) -> rank, -1 off the wall layer
    amrex::GpuArray<amrex::Array4<const amrex::Real>, 3> f; //!< the three face fluxes
    ibm::IbmGeometryView g;                                 //!< read at (i, j, k) ONLY — Q34
    ibm::GhostCellView d;                                   //!< the method's rows, by rank
    ibm::RobinView robin;                                   //!< the per-patch (alpha, beta, gamma)
    ibm::DivFaceValue face_value;                           //!< Central | Upwind
    amrex::Real t;                                          //!< the time gamma(t) is read at
    amrex::GpuArray<amrex::Real, 3> dx;                     //!< the level's cell size

    //! v1's `flux[:, d, face]`: `face = 1` for `step = +1` is the cell's own
    //! HIGH face, `face = 0` for `step = -1` its LOW one. Identical indices to
    //! `stencil_kernels.cpp`'s `fx(i, j, k)` / `fx(i + 1, j, k)`.
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
    faceFlux(int dd, int i, int j, int k, int step) const noexcept
    {
        const int up = (step == 1) ? 1 : 0;
        return f[dd](i + ((dd == 0) ? up : 0), j + ((dd == 1) ? up : 0), k + ((dd == 2) ? up : 0));
    }

    //! v1's `_face_weights`: the interior scheme's width-1 weight on the
    //! TARGET's own value at this face. `Central` keeps the central average;
    //! `Upwind` takes the cell the flux comes from. The flux is stored in the
    //! `+d` orientation on both faces, so `f >= 0` means the LOW face's upstream
    //! cell is the neighbour and the HIGH face's is the target.
    //!
    //! H-8: the comparison is `>=`, never `>` — v1's spelling, and a measured
    //! control (the two differ only at `f = +-0.0`, where every product is
    //! `+-0.0` with `scale`'s own sign either way).
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
    weightSelf(amrex::Real flux, int step) const noexcept
    {
        if (face_value == ibm::DivFaceValue::Central) return 0.5;
        const bool positive = (flux >= 0.0);
        return (step == 1) ? (positive ? 1.0 : 0.0) : (positive ? 0.0 : 1.0);
    }

    template<class Sink>
    AMREX_GPU_HOST_DEVICE void operator()(int i, int j, int k, int n, Sink& sink) const
    {
        // The frame calls a functor at WALL cells only, and every WALL cell has
        // a row by construction of the map (`ibm/ghost_cell.cpp`, pass 1).
        const int r = row(i, j, k);
        const int p = g.patch(i, j, k); // Q34: geometry at (i, j, k) ONLY
        const amrex::Real s_P = g.sdf(i, j, k);

        // The closure ONCE per (cell, component), before the faces — v1
        // computes it once per row and reuses it in all six.
        const ibm::WallClosure w =
            ibm::closure(robin.alpha[p], robin.beta[p], robin.gammaAt(p, n, t), d.distance[r]);

        amrex::Real diag = 0.0;
        amrex::Real cacc = 0.0;
        amrex::Real wdon[ibm::K] = {};
        // H-6: v1's slots 1..6 are ZERO-INITIALISED and written once, so the
        // shipped coefficient is `0.0 + nb_part` and not `nb_part`. Emitting the
        // raw value ships `-0.0` where v1 has `+0.0` — 960 of 3 232 wall rows.
        amrex::Real arm[6] = {};

        // -------------------------------------------------------------------
        // pass 1 — ACCUMULATE, in v1's loop order: d ascending, +1 then -1
        // -------------------------------------------------------------------
        for (int dd = 0; dd < 3; ++dd)
            for (int s = 0; s < 2; ++s)
            {
                const int step = (s == 0) ? 1 : -1; // H-1: +1 first
                const int slot = 2 * dd + s;        // v1's `_slot(d, step) - 1`
                const int ii = i + ((dd == 0) ? step : 0);
                const int jj = j + ((dd == 1) ? step : 0);
                const int kk = k + ((dd == 2) ? step : 0);

                const amrex::Real fl = faceFlux(dd, i, j, k, step);
                const amrex::Real ws = weightSelf(fl, step);
                // H-5: v1 folds `coeff` in here; the frame folds it into the
                // sum. Identical at a power-of-two `coeff`, which is every row
                // in scope. v1 parses `((coeff*step)*flux)/dx`.
                const amrex::Real sc = step * fl / dx[dd];
                const amrex::Real slf = sc * ws;
                const amrex::Real nbp = sc * (1.0 - ws); // H-7, NOT `sc - slf`

                // H-3': EVERY face, fluid or wall. v1's mask is the ROW's
                // (`ctx.fluid`), which the frame has already established. This
                // is NOT the laplacian's fluid-arm gate.
                diag += slf;

                if (m(ii, jj, kk) != ibm::SOLID)
                {
                    // H-6: `0.0 + nbp`, which is `+0.0` where `nbp` is `-0.0`.
                    arm[slot] += nbp;
                }
                else
                {
                    // The ghost centre's own signed distance from the surface
                    // along THIS cell's normal — v1's
                    // `ctx.sdf + step * ctx.dx[d] * ctx.normal[:, d]`, which
                    // parses `s_P + ((step*dx)*n̂_d)` in both languages. An FMA
                    // site of this file's own, covered by `--fmad=false`.
                    const amrex::Real dG = s_P + step * dx[dd] * g.normal(i, j, k, dd);
                    // S7: the closure read a third way, not a new approximation.
                    const amrex::Real lin = w.atLinear(dG);
                    const amrex::Real sg = nbp * lin; // H-4': once per face
                    for (int q = 0; q < ibm::K; ++q)
                        wdon[q] += sg * d.weight[r * ibm::K + q];
                    // S2/R1: the BC datum reaches the row through `constant` and
                    // through nothing else. `linear` and `constant` are two
                    // methods with two signatures, so this is a type-level split
                    // and not a convention.
                    cacc += nbp * w.atConstant(dG);
                }
            }

        // -------------------------------------------------------------------
        // pass 2 — EMIT, in v1's slot order: c, slot 0, slots 1..6, slots 7..14
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
                // cell is never named. `WallFrameProbe`'s unconditional arms are
                // non-conformant by design and this must not copy them.
                if (m(ii, jj, kk) != ibm::SOLID) sink.linear(ii, jj, kk, arm[2 * dd + s]);
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
struct MakeWallDivGhostCell
{
    using functor_type = WallDivGhostCell;

    const ibm::CellTypeFab* ct;
    const ibm::IbmGeometryFab* g;
    const ibm::GhostCellData* data;
    const amrex::MultiFab* flux[3];
    ibm::RobinView robin;
    ibm::DivFaceValue face_value;
    amrex::Real t;
    amrex::GpuArray<amrex::Real, 3> dx;

    WallDivGhostCell operator()(const amrex::MFIter& mfi) const
    {
        return WallDivGhostCell {
            ct->const_array(mfi),
            data->row.const_array(mfi),
            {flux[0]->const_array(mfi), flux[1]->const_array(mfi), flux[2]->const_array(mfi)},
            ibm::makeGeometryView(*g, mfi),
            ibm::makeGhostCellView(*data),
            robin,
            face_value,
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

        // div-specific, and the same defect class: the three face fabs are
        // resolved by `MFIter` LOCAL INDEX beside phi/out/ct too, so they must
        // be the marker's own BoxArray converted to face centring in their own
        // direction, on the marker's DistributionMapping.
        //
        // There is deliberately NO ghost-width check on the flux: the functor
        // reads only the cell's own two faces in each direction, and a face
        // fab's valid box contains both of them for every cell of `validbox`.
        for (int dd = 0; dd < 3; ++dd)
            if (flux[dd]->boxArray()
                    != amrex::convert(ct->boxArray(), amrex::IntVect::TheDimensionVector(dd))
                || flux[dd]->DistributionMap() != ct->DistributionMap())
                throw std::runtime_error(
                    std::string(fn) + ": the face flux in direction " + std::to_string(dd)
                    + " is not on this level's grids — a div wall row reads it beside phi, out and "
                      "the cell_type marker by MFIter local index, so it must be the marker's "
                      "BoxArray converted to face centring in that direction, on the marker's "
                      "DistributionMapping; rebuild the face fluxes for this level's grids"
                );
    }
};

//! A host-resident copy of `GhostCellData`'s flat arrays, so the functor above
//! can be called ON THE HOST at one cell (api §10.6).
//!
//! The four arrays are `Gpu::DeviceVector`s — device memory, not managed — so
//! unlike `RobinView` they cannot simply be pointed at from the host. The values
//! are the real ones and the code that reads them is the same
//! `AMREX_GPU_HOST_DEVICE` member the kernel calls; only the residence changes.
//!
//! **A SECOND COPY of `laplacian_ghost_cell.cpp`'s type, deliberately** (api
//! §10.4 triggers at "a private helper that a second file copies"). This is that
//! second file, so the note is owed and the lift is not: B34 is the third copy
//! and the right moment to move it — into `wall_stage.H`, beside the other
//! staging helpers. Recorded as owed at B34.
struct HostGhostCell
{
    std::vector<amrex::Real> image_point;
    std::vector<int> donor;
    std::vector<amrex::Real> weight;
    std::vector<amrex::Real> distance;
    int nrows = 0;

    explicit HostGhostCell(const ibm::GhostCellData& d)
        : image_point(d.image_point.size()), donor(d.donor.size()), weight(d.weight.size()),
          distance(d.distance.size()), nrows(d.nrows)
    {
        if (d.nrows == 0) return;
        amrex::Gpu::copy(
            amrex::Gpu::deviceToHost, d.image_point.begin(), d.image_point.end(), image_point.data()
        );
        amrex::Gpu::copy(amrex::Gpu::deviceToHost, d.donor.begin(), d.donor.end(), donor.data());
        amrex::Gpu::copy(amrex::Gpu::deviceToHost, d.weight.begin(), d.weight.end(), weight.data());
        amrex::Gpu::copy(
            amrex::Gpu::deviceToHost, d.distance.begin(), d.distance.end(), distance.data()
        );
        amrex::Gpu::streamSynchronize();
    }

    ibm::GhostCellView view() const
    {
        return ibm::GhostCellView {
            image_point.data(), donor.data(), weight.data(), distance.data(), nrows
        };
    }
};

} // namespace

void registerDivGhostCell(nb::module_& m)
{
    // ----------------------------------------------------------------------
    // The face-value selector (review.md §4 Q52(b), B-2 approved).
    //
    // Bound in the pair's own TU rather than in `wall_frame.cpp`: `WallMode`'s
    // precedent is that the TU which owns the concept binds it, and keeping
    // `wall_frame.cpp` untouched is worth more than the symmetry.
    // ----------------------------------------------------------------------
    nb::enum_<ibm::DivFaceValue>(m, "DivFaceValue", nb::is_arithmetic())
        .value("Central", ibm::DivFaceValue::Central)
        .value("Upwind", ibm::DivFaceValue::Upwind);

    // ----------------------------------------------------------------------
    // THE PRODUCTION ENTRY POINT — the canonical twelve (design §4.4) PLUS the
    // four `div` needs (review.md §4 Q52(a)/(b), extending Q29(f)).
    //
    // Q39, ruled at B32: a REGISTERED pair carries all twelve `nb::arg`s in
    // that order, with no defaults. Q29(f) makes the twelve a MINIMUM, and
    // `div`'s row is a FACE balance: it needs the three face-flux MultiFabs and
    // it needs to know which face-value rule the interior scheme uses. Those are
    // arguments 13..16, in the order `div_*_acc(out, phi, fx, fy, fz, ...)`
    // already takes the three fluxes.
    //
    // B32's conformance row asserts the twelve as a PREFIX over every registered
    // `wall_*` attribute, so this pair enters that contract for free and the row
    // grows in strength at no cost.
    // ----------------------------------------------------------------------
    m.def(
        "wall_div_ghost_cell",
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
           double constant_scale,
           const amrex::MultiFab& flux_x,
           const amrex::MultiFab& flux_y,
           const amrex::MultiFab& flux_z,
           ibm::DivFaceValue face_value)
        {
            ibm::applyWall(
                "wall_div_ghost_cell",
                out,
                phi,
                cell_type,
                MakeWallDivGhostCell {
                    &cell_type,
                    &geom_ibm,
                    &method_data,
                    {&flux_x, &flux_y, &flux_z},
                    robin.view(),
                    face_value,
                    t,
                    geom.CellSizeArray()
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
        nb::arg("flux_x"),
        nb::arg("flux_y"),
        nb::arg("flux_z"),
        nb::arg("face_value"),
        "div x ghostCell over every WALL cell of the level: the width-1 face balance "
        "sum_d (f_d^+ phi_f^+ - f_d^- phi_f^-)/dx_d with each face whose neighbour is SOLID "
        "reading, instead of that neighbour, the field the wall condition extrapolates to its "
        "cell centre — robin.H's closure(alpha, beta, gamma(t), d).at(d_G) at "
        "d_G = sdf + step*dx_d*n_d. face_value is the INTERIOR scheme's width-1 face rule: "
        "linear -> Central, upwind -> Upwind, and vanLeer/quick -> Upwind, which is the D1 "
        "degrade (a width-2 stencil reaches through the solid inside the band). Overwrite "
        "assigns, Add accumulates, Assemble raises. SOLID and FLUID cells are not written at "
        "all. constant_scale = 0 drops exactly the BC datum. Bitwise equal to v1's "
        "ghost_cell._face_balance_rows, row for row."
    );

    // ----------------------------------------------------------------------
    // TEST binding (api §4, §10.6) — the same functor, on the HOST, at ONE
    // cell, against a `RecordSink`. Underscore-private, never registered, never
    // on an evaluate path, and exempt from the twelve by Q39.
    // ----------------------------------------------------------------------
    m.def(
        "_wall_row_div_ghost_cell",
        [](const ibm::CellTypeFab& cell_type,
           const ibm::IbmGeometryFab& geom_ibm,
           const ibm::GhostCellData& method_data,
           const ibm::RobinData& robin,
           const amrex::Geometry& geom,
           double t,
           const amrex::MultiFab& flux_x,
           const amrex::MultiFab& flux_y,
           const amrex::MultiFab& flux_z,
           ibm::DivFaceValue face_value,
           int i,
           int j,
           int k,
           int n) -> nb::tuple
        {
            static constexpr const char* FN = "_wall_row_div_ghost_cell";

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

            const amrex::IntVect iv(i, j, k);
            amrex::FArrayBox hostG;
            ibm::stageGeometryBox(FN, geom_ibm, iv, hostG);
            amrex::BaseFab<std::uint8_t> hostM;
            ibm::stageMarkerBox(FN, cell_type, iv, hostM);
            const amrex::MultiFab* src[3] = {&flux_x, &flux_y, &flux_z};
            amrex::FArrayBox hostF[3];
            for (int dd = 0; dd < 3; ++dd)
                ibm::stageFaceBox(FN, *src[dd], dd, iv, hostF[dd]);
            const HostGhostCell hostD(method_data);

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

            const WallDivGhostCell f {
                hostM.const_array(),
                rowArr,
                {hostF[0].const_array(), hostF[1].const_array(), hostF[2].const_array()},
                gv,
                hostD.view(),
                robin.view(),
                face_value,
                t,
                geom.CellSizeArray()
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
        nb::arg("flux_x"),
        nb::arg("flux_y"),
        nb::arg("flux_z"),
        nb::arg("face_value"),
        nb::arg("i"),
        nb::arg("j"),
        nb::arg("k"),
        nb::arg("n"),
        "TEST ONLY (B33). div x ghostCell's row at one WALL cell, computed on the HOST against a "
        "RecordSink: returns ([(i, j, k, a), ...], c) — the ordered linear entries and the "
        "constant. The order is v1's: the diagonal, then the fluid faces in slot order, then the "
        "eight trilinear donors. Raises if the cell is not a WALL cell."
    );
}
