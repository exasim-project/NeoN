// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

// The wall sweep's FRAME (design §4.1, api §5.1) — B30a.
//
// Written once and shared by every `(operator, method)` pair: the `MFIter`
// loop, the guards, the marker branch, the modes, the launch. A pair supplies a
// Maker and nothing else.
//
// ---------------------------------------------------------------------------
// WHAT THIS HEADER DOES NOT INCLUDE, AND WHY THAT IS THE POINT
// ---------------------------------------------------------------------------
// Not `robin_data.H`, not `geometry_view.H`, not any method's data header, and
// above all not `robin.H` (B30b, the row under revision — review.md §4
// Q28/Q30). The frame sees eight arguments of the canonical twelve
// (design §4.4); the Maker absorbs the other four, opaquely. That split IS the
// formula-independence claim, stated mechanically: nothing B44's successor can
// do to the closure reaches this file, because this file cannot name it.
//
// ---------------------------------------------------------------------------
// THE MAKER CONCEPT
// ---------------------------------------------------------------------------
//     Maker::functor_type                                       // the pair's functor
//     functor_type Maker::operator()(const amrex::MFIter&) const   // built per box, by value
//     functor_type::stencil_reach                               // static constexpr int
//     functor_type::operator()(i, j, k, n, Sink&) const         // AMREX_GPU_HOST_DEVICE,
//                                                               // template<class Sink>
//
// The functor is captured BY VALUE into every thread, so its members are views
// and scalars only — never an owning container, never a `std::function`
// (design §4.3, register pressure).
//
// ---------------------------------------------------------------------------
// THE GUARDS (api §8, design §4.4)
// ---------------------------------------------------------------------------
// v1 needed five host-side agreement checks because the row table was an
// independent object that could disagree with the MultiFab it indexed:
// staleness, local box count, targets inside the valid box, CSR monotonicity,
// stencils inside the fab box. The marker is a `FabArray` on the same
// `BoxArray` and `DistributionMapping`, rebuilt by the same invalidation, so
// four of those five cannot arise *once the fabs really are on one set of
// grids* — and that they are is no longer assumed: it is CHECKED HERE, at the
// top of the sweep (B30a-R's I-2). What survives beyond it is the one that was
// never about the table — a stencil must fit inside the ghost region — and it
// is now declared by the functor as `stencil_reach` instead of by a registry.
//
// The frame makes for itself every check it has the arguments to make: the
// ghost widths, `out != phi`, and the refusal of `WallMode::Assemble`. The one
// remaining guard, that the Robin table carries one gamma per field component,
// needs a type this header refuses to know, so it is `requireRobinComponents`
// in `robin_data.H` and a pair's binding calls it.
//
// Read that as a per-box statement, not as a call-level guarantee: the two
// `requireGhostWidth` calls sit INSIDE the `MFIter` loop (D5, so the guard
// narrates itself the way `geometry_view.H` and api §8 already say it does), so
// on a rank that owns no local box neither of them runs — and neither does any
// launch. Only the `mode` and `&out != &phi` guards are outside the loop and
// therefore fire on every rank, every call.

#include "wall_value.H"

#include "../../ibm/cell_type.H"

#include <AMReX_Box.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_MFIter.H>
#include <AMReX_MultiFab.H>
#include <AMReX_REAL.H>

#include <stdexcept>
#include <string>

namespace ibm
{

//! The declared reach must fit in the ghost region of everything the launch
//! reads off-centre — the FIELD side of the ghost contract, per box, per
//! launch.
//!
//! Deliberately named apart from `requireGeometryGhosts` (`ibm/geometry_view.H`,
//! api §8): that one polices the GEOMETRY fab against the marker it is paired
//! with, this one a field-shaped fab against a functor's declared stencil
//! reach. Two subjects, two checks, one message shape — the width it needs and
//! the width it has.
//!
//! What it does NOT check is geometry and method-data width, and that is a
//! conformance statement rather than an omission: the frame's contract is that
//! a functor reads `geom_ibm` and `method_data` **at the target cell only**
//! (api §5.3 does, and review.md §4 Q34 keeps it that way until someone
//! decides what a ghost `normal` means). A functor that reads geometry at a
//! neighbour is outside its own declared reach and needs a third array here.
template<class FA>
inline void requireGhostWidth(const char* fn, const char* what, const FA& fa, int reach)
{
    const int has = fa.nGrowVect().min();
    if (has >= reach) return;
    throw std::runtime_error(
        std::string(fn) + ": the functor declares stencil_reach = " + std::to_string(reach)
        + " but " + what + " has ngrow = " + std::to_string(has)
        + "; a wall stencil must fit inside the ghost region — grow the field or use a narrower "
          "wall scheme"
    );
}

//! The sweep. api §5.1, with the guards where the arguments to make them are.
//!
//! `fn` is the pair's own entry-point name (`wall_laplacian_ghost_cell`, ...),
//! not this template's: api §9 requires a failure to name the offending object,
//! and "applyWall" names nothing a caller can see.
template<class Maker>
void applyWall(
    const char* fn,
    amrex::MultiFab& out,
    const amrex::MultiFab& phi,
    const CellTypeFab& ct,
    Maker maker,
    amrex::Real coeff,
    int ncomp,
    WallMode mode,
    amrex::Real constant_scale
)
{
    using functor_type = typename Maker::functor_type;
    static_assert(
        functor_type::stencil_reach >= 0,
        "a wall functor declares `static constexpr int stencil_reach` — the number of cells its "
        "row reaches outside the cell it is called on. It is what the ghost-width guard checks, "
        "so there is no default and no way to opt out."
    );

    // S4: the frame owns the mode, and a mode it cannot honour says so rather
    // than behaving like the nearest one it can. `AssembleSink` is the implicit
    // track's, and the implicit track is not built (api §5.2).
    if (mode == WallMode::Assemble)
        throw std::runtime_error(
            std::string(fn)
            + ": WallMode.Assemble is the implicit track's seam and is declared, not implemented "
              "— there is no AssembleSink in this build. Use Overwrite for the first term of an "
              "equation and Add for the rest."
        );

    // design §4.4: `out` and `phi` are different MultiFabs, which is what makes
    // the sweep free of atomics and of any ordering constraint between cells.
    // In place, a wall row would read cells another thread had already written.
    if (&out == &phi)
        throw std::runtime_error(
            std::string(fn)
            + ": the wall sweep reads phi and writes out, so they must be different MultiFabs — "
              "in place, a row would read values another cell's row had already overwritten"
        );

    // B30a-R, I-2: the loop below resolves `out`, `phi` and `ct` by MFIter
    // LOCAL INDEX, so three fabs on different grids are paired by position and
    // not by box. In a `-DNDEBUG` build a different box count reads another
    // box's cells and a different extent runs past the end of a fab (measured:
    // exit 11). Precedent for the check and its shape: `ibm/cell_type.cpp`'s
    // `requireMarkerGhostWidth`.
    //
    // It can only cover the three fabs THIS function holds. `geom_ibm` and the
    // method's data live inside the Maker, which the frame cannot see, so the
    // rest of the agreement is B30a-R's S-5 (a `Maker::validate(fn, ncomp)`
    // hook) and still owed. Do not read this guard as complete.
    if (out.boxArray() != phi.boxArray() || ct.boxArray() != phi.boxArray()
        || out.DistributionMap() != phi.DistributionMap()
        || ct.DistributionMap() != phi.DistributionMap())
        throw std::runtime_error(
            std::string(fn)
            + ": out, phi and the cell_type marker must share one BoxArray and one "
              "DistributionMapping — this sweep pairs them by MFIter local index, so a mismatch "
              "reads another box's cells (or, at different extents, past the end of a fab); "
              "rebuild them on this level's grids"
        );

    const int reach = functor_type::stencil_reach;
    const bool add = (mode == WallMode::Add);

    for (amrex::MFIter mfi(phi); mfi.isValid(); ++mfi)
    {
        requireGhostWidth(fn, "the field", phi, reach);
        requireGhostWidth(fn, "the cell_type marker", ct, reach);

        auto const& src = phi.const_array(mfi);
        auto const& dst = out.array(mfi);
        auto const& m = ct.const_array(mfi);
        const auto f = maker(mfi); // the pair's functor, built per box, by value

        amrex::ParallelFor(
            mfi.validbox(),
            ncomp,
            [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
            {
                // SOLID: no-op — the pin owns that value (design §7, Q3), and
                // v1's zero row there is the one behavioural difference the
                // port records rather than reproduces (plan §8, OPEN-C).
                // FLUID: the interior sweep already did it.
                if (m(i, j, k) != ibm::WALL) return;

                ApplySink sink {src, n, constant_scale};
                f(i, j, k, n, sink); // <- the ONLY line that varies by pair
                const amrex::Real value = coeff * sink.value();
                if (add)
                {
                    dst(i, j, k, n) += value;
                }
                else
                {
                    dst(i, j, k, n) = value;
                }
            }
        );
    }
}

} // namespace ibm
