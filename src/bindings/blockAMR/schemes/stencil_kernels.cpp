// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

// C++ baseline stencil kernels for performance comparison with JAX dispatch.
// Implements VanLeer divergence + laplacian forward Euler step directly
// on AMReX MultiFabs using ParallelFor (GPU-accelerated when available).

#include <nanobind/nanobind.h>

#include <AMReX.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Geometry.H>

#include "../bindings.hpp"
#include "../ibm/cell_type.H"

#include <cstdint>
#include <stdexcept>
#include <string>

namespace nb = nanobind;

namespace
{

// Harmonic-mean VanLeer correction: ψ(r)*Δ with one division, no abs.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
vanleerCorr(amrex::Real d_up, amrex::Real d_down)
{
    amrex::Real prod = d_up * d_down;
    return (prod > 0.0) ? 2.0 * prod / (d_up + d_down) : 0.0;
}

// ---------------------------------------------------------------------------
// Comp-indexed cell-level scheme helpers for the composable accumulate kernels.
// Formulas mirror the JAX cell kernels in cell_kernels.py exactly (per-axis
// division, x→y→z accumulation order) so the cpp backend matches the jax path.
// ---------------------------------------------------------------------------

using Arr4c = amrex::Array4<const amrex::Real>;

AMREX_GPU_DEVICE AMREX_FORCE_INLINE amrex::Real divUpwindCell(
    Arr4c const& phi,
    Arr4c const& fx,
    Arr4c const& fy,
    Arr4c const& fz,
    int i,
    int j,
    int k,
    int n,
    amrex::Real dx,
    amrex::Real dy,
    amrex::Real dz
)
{
    amrex::Real s0 = phi(i, j, k, n);
    amrex::Real total = 0.0;
    {
        amrex::Real fl = fx(i, j, k), fr = fx(i + 1, j, k);
        amrex::Real Fl = fl * ((fl >= 0.0) ? phi(i - 1, j, k, n) : s0);
        amrex::Real Fr = fr * ((fr >= 0.0) ? s0 : phi(i + 1, j, k, n));
        total += (Fr - Fl) / dx;
    }
    {
        amrex::Real fl = fy(i, j, k), fr = fy(i, j + 1, k);
        amrex::Real Fl = fl * ((fl >= 0.0) ? phi(i, j - 1, k, n) : s0);
        amrex::Real Fr = fr * ((fr >= 0.0) ? s0 : phi(i, j + 1, k, n));
        total += (Fr - Fl) / dy;
    }
    {
        amrex::Real fl = fz(i, j, k), fr = fz(i, j, k + 1);
        amrex::Real Fl = fl * ((fl >= 0.0) ? phi(i, j, k - 1, n) : s0);
        amrex::Real Fr = fr * ((fr >= 0.0) ? s0 : phi(i, j, k + 1, n));
        total += (Fr - Fl) / dz;
    }
    return total;
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE amrex::Real divLinearCell(
    Arr4c const& phi,
    Arr4c const& fx,
    Arr4c const& fy,
    Arr4c const& fz,
    int i,
    int j,
    int k,
    int n,
    amrex::Real dx,
    amrex::Real dy,
    amrex::Real dz
)
{
    amrex::Real s0 = phi(i, j, k, n);
    amrex::Real total = 0.0;
    {
        amrex::Real fl = fx(i, j, k), fr = fx(i + 1, j, k);
        amrex::Real Fl = fl * 0.5 * (phi(i - 1, j, k, n) + s0);
        amrex::Real Fr = fr * 0.5 * (s0 + phi(i + 1, j, k, n));
        total += (Fr - Fl) / dx;
    }
    {
        amrex::Real fl = fy(i, j, k), fr = fy(i, j + 1, k);
        amrex::Real Fl = fl * 0.5 * (phi(i, j - 1, k, n) + s0);
        amrex::Real Fr = fr * 0.5 * (s0 + phi(i, j + 1, k, n));
        total += (Fr - Fl) / dy;
    }
    {
        amrex::Real fl = fz(i, j, k), fr = fz(i, j, k + 1);
        amrex::Real Fl = fl * 0.5 * (phi(i, j, k - 1, n) + s0);
        amrex::Real Fr = fr * 0.5 * (s0 + phi(i, j, k + 1, n));
        total += (Fr - Fl) / dz;
    }
    return total;
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE amrex::Real divVanLeerCell(
    Arr4c const& phi,
    Arr4c const& fx,
    Arr4c const& fy,
    Arr4c const& fz,
    int i,
    int j,
    int k,
    int n,
    amrex::Real dx,
    amrex::Real dy,
    amrex::Real dz
)
{
    amrex::Real total = 0.0;
    {
        amrex::Real fl = fx(i, j, k), fr = fx(i + 1, j, k);
        amrex::Real sm2 = phi(i - 2, j, k, n), sm1 = phi(i - 1, j, k, n), s0 = phi(i, j, k, n),
                    sp1 = phi(i + 1, j, k, n), sp2 = phi(i + 2, j, k, n);
        amrex::Real dl = s0 - sm1;
        amrex::Real pl = (fl >= 0.0) ? sm1 + 0.5 * vanleerCorr(sm1 - sm2, dl)
                                     : s0 - 0.5 * vanleerCorr(sp1 - s0, dl);
        amrex::Real dr = sp1 - s0;
        amrex::Real pr = (fr >= 0.0) ? s0 + 0.5 * vanleerCorr(s0 - sm1, dr)
                                     : sp1 - 0.5 * vanleerCorr(sp2 - sp1, dr);
        total += (fr * pr - fl * pl) / dx;
    }
    {
        amrex::Real fl = fy(i, j, k), fr = fy(i, j + 1, k);
        amrex::Real sm2 = phi(i, j - 2, k, n), sm1 = phi(i, j - 1, k, n), s0 = phi(i, j, k, n),
                    sp1 = phi(i, j + 1, k, n), sp2 = phi(i, j + 2, k, n);
        amrex::Real dl = s0 - sm1;
        amrex::Real pl = (fl >= 0.0) ? sm1 + 0.5 * vanleerCorr(sm1 - sm2, dl)
                                     : s0 - 0.5 * vanleerCorr(sp1 - s0, dl);
        amrex::Real dr = sp1 - s0;
        amrex::Real pr = (fr >= 0.0) ? s0 + 0.5 * vanleerCorr(s0 - sm1, dr)
                                     : sp1 - 0.5 * vanleerCorr(sp2 - sp1, dr);
        total += (fr * pr - fl * pl) / dy;
    }
    {
        amrex::Real fl = fz(i, j, k), fr = fz(i, j, k + 1);
        amrex::Real sm2 = phi(i, j, k - 2, n), sm1 = phi(i, j, k - 1, n), s0 = phi(i, j, k, n),
                    sp1 = phi(i, j, k + 1, n), sp2 = phi(i, j, k + 2, n);
        amrex::Real dl = s0 - sm1;
        amrex::Real pl = (fl >= 0.0) ? sm1 + 0.5 * vanleerCorr(sm1 - sm2, dl)
                                     : s0 - 0.5 * vanleerCorr(sp1 - s0, dl);
        amrex::Real dr = sp1 - s0;
        amrex::Real pr = (fr >= 0.0) ? s0 + 0.5 * vanleerCorr(s0 - sm1, dr)
                                     : sp1 - 0.5 * vanleerCorr(sp2 - sp1, dr);
        total += (fr * pr - fl * pl) / dz;
    }
    return total;
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE amrex::Real divQuickCell(
    Arr4c const& phi,
    Arr4c const& fx,
    Arr4c const& fy,
    Arr4c const& fz,
    int i,
    int j,
    int k,
    int n,
    amrex::Real dx,
    amrex::Real dy,
    amrex::Real dz
)
{
    amrex::Real total = 0.0;
    {
        amrex::Real fl = fx(i, j, k), fr = fx(i + 1, j, k);
        amrex::Real sm2 = phi(i - 2, j, k, n), sm1 = phi(i - 1, j, k, n), s0 = phi(i, j, k, n),
                    sp1 = phi(i + 1, j, k, n), sp2 = phi(i + 2, j, k, n);
        amrex::Real pl = (fl >= 0.0) ? 0.375 * s0 + 0.75 * sm1 - 0.125 * sm2
                                     : 0.375 * sm1 + 0.75 * s0 - 0.125 * sp1;
        amrex::Real pr = (fr >= 0.0) ? 0.375 * sp1 + 0.75 * s0 - 0.125 * sm1
                                     : 0.375 * s0 + 0.75 * sp1 - 0.125 * sp2;
        total += (fr * pr - fl * pl) / dx;
    }
    {
        amrex::Real fl = fy(i, j, k), fr = fy(i, j + 1, k);
        amrex::Real sm2 = phi(i, j - 2, k, n), sm1 = phi(i, j - 1, k, n), s0 = phi(i, j, k, n),
                    sp1 = phi(i, j + 1, k, n), sp2 = phi(i, j + 2, k, n);
        amrex::Real pl = (fl >= 0.0) ? 0.375 * s0 + 0.75 * sm1 - 0.125 * sm2
                                     : 0.375 * sm1 + 0.75 * s0 - 0.125 * sp1;
        amrex::Real pr = (fr >= 0.0) ? 0.375 * sp1 + 0.75 * s0 - 0.125 * sm1
                                     : 0.375 * s0 + 0.75 * sp1 - 0.125 * sp2;
        total += (fr * pr - fl * pl) / dy;
    }
    {
        amrex::Real fl = fz(i, j, k), fr = fz(i, j, k + 1);
        amrex::Real sm2 = phi(i, j, k - 2, n), sm1 = phi(i, j, k - 1, n), s0 = phi(i, j, k, n),
                    sp1 = phi(i, j, k + 1, n), sp2 = phi(i, j, k + 2, n);
        amrex::Real pl = (fl >= 0.0) ? 0.375 * s0 + 0.75 * sm1 - 0.125 * sm2
                                     : 0.375 * sm1 + 0.75 * s0 - 0.125 * sp1;
        amrex::Real pr = (fr >= 0.0) ? 0.375 * sp1 + 0.75 * s0 - 0.125 * sm1
                                     : 0.375 * s0 + 0.75 * sp1 - 0.125 * sp2;
        total += (fr * pr - fl * pl) / dz;
    }
    return total;
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE amrex::Real laplacianCell(
    Arr4c const& phi, int i, int j, int k, int n, amrex::Real dx, amrex::Real dy, amrex::Real dz
)
{
    amrex::Real c = phi(i, j, k, n);
    return (phi(i + 1, j, k, n) - 2.0 * c + phi(i - 1, j, k, n)) / (dx * dx)
         + (phi(i, j + 1, k, n) - 2.0 * c + phi(i, j - 1, k, n)) / (dy * dy)
         + (phi(i, j, k + 1, n) - 2.0 * c + phi(i, j, k - 1, n)) / (dz * dz);
}

// ---------------------------------------------------------------------------
// W1 — the immersed-boundary degrade (design §5, review.md §4 Q42(a)).
//
// A width-w > 1 interior scheme falls back to its width-1 formula at any cell
// whose stencil would read a SOLID cell. The test is PER CELL, on the stencil's
// own offsets — NOT per face. A per-face degrade mixes an upwind face (a cell
// centre) with a van Leer face (a face centre) and returns 1.5*B instead of B
// on a linear field at the cell two out from a plane wall, which breaks D1.
// Both arms of the branch below are calls to the parents' own device functions,
// so no formula is written twice and "bitwise the parent wherever nothing
// reaches SOLID" is a property of the code rather than a bet on the compiler.
// ---------------------------------------------------------------------------

using CtArr4c = amrex::Array4<const std::uint8_t>;

// The twelve axis cells a width-2 div stencil reads. There are no diagonals in
// it, which is why the "exact for any stencil shape" claim holds by inspection.
//
// Spelled as a SOLID comparison on each offset, not as `m(i, j, k) == ibm::WALL`
// at the centre: the two agree only under `classifyDefault`, and the short form
// would couple the scheme to the method's marker rule (design §5 — the marker
// is geometry; the stencil is the scheme's business). The +-1 offsets are in
// fact redundant for a FLUID centre — by the definition of WALL no FLUID cell
// has a SOLID face neighbour — so they fire only at WALL cells, which the wall
// sweep overwrites anyway. They stay because the rule is "the stencil's own
// offsets"; this note is here so a later reader does not remove them as dead.
AMREX_GPU_DEVICE AMREX_FORCE_INLINE bool solidWithinTwo(CtArr4c const& m, int i, int j, int k)
{
    return m(i - 2, j, k) == ibm::SOLID || m(i - 1, j, k) == ibm::SOLID
        || m(i + 1, j, k) == ibm::SOLID || m(i + 2, j, k) == ibm::SOLID
        || m(i, j - 2, k) == ibm::SOLID || m(i, j - 1, k) == ibm::SOLID
        || m(i, j + 1, k) == ibm::SOLID || m(i, j + 2, k) == ibm::SOLID
        || m(i, j, k - 2) == ibm::SOLID || m(i, j, k - 1) == ibm::SOLID
        || m(i, j, k + 1) == ibm::SOLID || m(i, j, k + 2) == ibm::SOLID;
}

// The ghost contract of the degrade, on the field and on the marker. The message
// shape is the compiled surface's standard (api §8) — the width it needs and the
// width it has — but the sentence is this caller's own: `wall_apply.H`'s
// `requireGhostWidth` says "the functor declares stencil_reach = N", and an
// interior kernel has no functor. Shipping a guard whose message narrates a call
// pattern the caller does not have is the defect class found at B29-R I-1 and
// B31-R I-2, so the guard is duplicated rather than reused.
//
// Called OUTSIDE the MFIter loop, unlike `applyWall`'s (B30a D5): the width is a
// property of the fab, an interior kernel has no per-box narration to add, and
// out here the check fires on every rank on every call even when a rank owns no
// local box. `Array4`'s own index assert is compiled out of a release build, so
// this is the only thing between a narrow marker and an illegal address.
template<class FA>
void requireStencilGhosts(const char* fn, const char* what, const FA& fa, int reach)
{
    const int has = fa.nGrowVect().min();
    if (has >= reach) return;
    throw std::runtime_error(
        std::string(fn)
        + ": the W1 degrade tests the marker on the stencil's own offsets, so it "
          "reads "
        + std::to_string(reach) + " cells outside the valid box, but " + what
        + " has ngrow = " + std::to_string(has) + "; grow the field and the marker to "
        + std::to_string(reach) + ", or use a width-1 div scheme"
    );
}

// `ct.const_array(mfi)` with an MFIter over a foreign BoxArray or a foreign
// DistributionMapping indexes another box's memory: a segfault when the counts
// differ (measured at B30a-R, I-2) and a plausible wrong answer when the counts
// agree and the extents do not. Precedent: `ibm/cell_type.cpp`'s own BoxArray
// guards on `classify_default` and `pin_solid`.
void requireSameLayout(const char* fn, const amrex::MultiFab& phi_mf, const ibm::CellTypeFab& ct)
{
    if (phi_mf.boxArray() == ct.boxArray() && phi_mf.DistributionMap() == ct.DistributionMap())
        return;
    throw std::runtime_error(
        std::string(fn)
        + ": the marker must share the field's BoxArray and DistributionMapping — the marker is a "
          "field with the same decomposition as the fields it accompanies; the field's BoxArray "
          "has "
        + std::to_string(phi_mf.boxArray().size()) + " boxes and the marker's has "
        + std::to_string(ct.boxArray().size())
    );
}

} // namespace


// ---------------------------------------------------------------------------
// Composable accumulate kernels (out +=) + generic axpy — the cpp backend
// composes these into a scratch source MultiFab (see plan 03 §4).
// ---------------------------------------------------------------------------

// Accumulate coeff * div_scheme(phi) into out, one term per launch, ncomp-general.
static void divUpwindAcc(
    amrex::MultiFab& out_mf,
    const amrex::MultiFab& phi_mf,
    const amrex::MultiFab& fx_mf,
    const amrex::MultiFab& fy_mf,
    const amrex::MultiFab& fz_mf,
    const amrex::Geometry& geom,
    amrex::Real coeff,
    int ncomp
)
{
    const auto dx = geom.CellSizeArray();
    const amrex::Real dhx = dx[0], dhy = dx[1], dhz = dx[2];
    for (amrex::MFIter mfi(phi_mf); mfi.isValid(); ++mfi)
    {
        const amrex::Box& bx = mfi.validbox();
        auto const& phi = phi_mf.const_array(mfi);
        auto const& out = out_mf.array(mfi);
        auto const& fx = fx_mf.const_array(mfi);
        auto const& fy = fy_mf.const_array(mfi);
        auto const& fz = fz_mf.const_array(mfi);
        amrex::ParallelFor(
            bx,
            ncomp,
            [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) {
                out(i, j, k, n) +=
                    coeff * divUpwindCell(phi, fx, fy, fz, i, j, k, n, dhx, dhy, dhz);
            }
        );
    }
}

static void divLinearAcc(
    amrex::MultiFab& out_mf,
    const amrex::MultiFab& phi_mf,
    const amrex::MultiFab& fx_mf,
    const amrex::MultiFab& fy_mf,
    const amrex::MultiFab& fz_mf,
    const amrex::Geometry& geom,
    amrex::Real coeff,
    int ncomp
)
{
    const auto dx = geom.CellSizeArray();
    const amrex::Real dhx = dx[0], dhy = dx[1], dhz = dx[2];
    for (amrex::MFIter mfi(phi_mf); mfi.isValid(); ++mfi)
    {
        const amrex::Box& bx = mfi.validbox();
        auto const& phi = phi_mf.const_array(mfi);
        auto const& out = out_mf.array(mfi);
        auto const& fx = fx_mf.const_array(mfi);
        auto const& fy = fy_mf.const_array(mfi);
        auto const& fz = fz_mf.const_array(mfi);
        amrex::ParallelFor(
            bx,
            ncomp,
            [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) {
                out(i, j, k, n) +=
                    coeff * divLinearCell(phi, fx, fy, fz, i, j, k, n, dhx, dhy, dhz);
            }
        );
    }
}

static void divVanLeerAcc(
    amrex::MultiFab& out_mf,
    const amrex::MultiFab& phi_mf,
    const amrex::MultiFab& fx_mf,
    const amrex::MultiFab& fy_mf,
    const amrex::MultiFab& fz_mf,
    const amrex::Geometry& geom,
    amrex::Real coeff,
    int ncomp
)
{
    const auto dx = geom.CellSizeArray();
    const amrex::Real dhx = dx[0], dhy = dx[1], dhz = dx[2];
    for (amrex::MFIter mfi(phi_mf); mfi.isValid(); ++mfi)
    {
        const amrex::Box& bx = mfi.validbox();
        auto const& phi = phi_mf.const_array(mfi);
        auto const& out = out_mf.array(mfi);
        auto const& fx = fx_mf.const_array(mfi);
        auto const& fy = fy_mf.const_array(mfi);
        auto const& fz = fz_mf.const_array(mfi);
        amrex::ParallelFor(
            bx,
            ncomp,
            [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) {
                out(i, j, k, n) +=
                    coeff * divVanLeerCell(phi, fx, fy, fz, i, j, k, n, dhx, dhy, dhz);
            }
        );
    }
}

static void divQuickAcc(
    amrex::MultiFab& out_mf,
    const amrex::MultiFab& phi_mf,
    const amrex::MultiFab& fx_mf,
    const amrex::MultiFab& fy_mf,
    const amrex::MultiFab& fz_mf,
    const amrex::Geometry& geom,
    amrex::Real coeff,
    int ncomp
)
{
    const auto dx = geom.CellSizeArray();
    const amrex::Real dhx = dx[0], dhy = dx[1], dhz = dx[2];
    for (amrex::MFIter mfi(phi_mf); mfi.isValid(); ++mfi)
    {
        const amrex::Box& bx = mfi.validbox();
        auto const& phi = phi_mf.const_array(mfi);
        auto const& out = out_mf.array(mfi);
        auto const& fx = fx_mf.const_array(mfi);
        auto const& fy = fy_mf.const_array(mfi);
        auto const& fz = fz_mf.const_array(mfi);
        amrex::ParallelFor(
            bx,
            ncomp,
            [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
            { out(i, j, k, n) += coeff * divQuickCell(phi, fx, fy, fz, i, j, k, n, dhx, dhy, dhz); }
        );
    }
}

// ---------------------------------------------------------------------------
// The W1 siblings (design §5, Q42(a)). Each is its parent above with three
// additions and nothing else: the marker argument, the three host-side guards,
// and the one `if` in the lambda. The accumulate statement in the `else` arm is
// the parent's, token for token — hoisting `coeff *` out, computing into a temp
// or reordering the call would make "bitwise the parent" a claim about the
// compiler instead of a claim about the code.
//
// No early-out at a SOLID centre cell: the parents do not skip, v2 leaves the
// interior sweep's value at SOLID cells (the wall sweep writes WALL only), and
// the all-SOLID row that pins the fallback needs every cell written.
// ---------------------------------------------------------------------------

static void divVanLeerIbmAcc(
    amrex::MultiFab& out_mf,
    const amrex::MultiFab& phi_mf,
    const ibm::CellTypeFab& ct,
    const amrex::MultiFab& fx_mf,
    const amrex::MultiFab& fy_mf,
    const amrex::MultiFab& fz_mf,
    const amrex::Geometry& geom,
    amrex::Real coeff,
    int ncomp
)
{
    requireStencilGhosts("div_vanleer_acc_ibm", "the field", phi_mf, 2);
    requireStencilGhosts("div_vanleer_acc_ibm", "the cell_type marker", ct, 2);
    requireSameLayout("div_vanleer_acc_ibm", phi_mf, ct);

    const auto dx = geom.CellSizeArray();
    const amrex::Real dhx = dx[0], dhy = dx[1], dhz = dx[2];
    for (amrex::MFIter mfi(phi_mf); mfi.isValid(); ++mfi)
    {
        const amrex::Box& bx = mfi.validbox();
        auto const& phi = phi_mf.const_array(mfi);
        auto const& out = out_mf.array(mfi);
        auto const& m = ct.const_array(mfi);
        auto const& fx = fx_mf.const_array(mfi);
        auto const& fy = fy_mf.const_array(mfi);
        auto const& fz = fz_mf.const_array(mfi);
        amrex::ParallelFor(
            bx,
            ncomp,
            [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
            {
                if (solidWithinTwo(m, i, j, k))
                    out(i, j, k, n) +=
                        coeff * divUpwindCell(phi, fx, fy, fz, i, j, k, n, dhx, dhy, dhz);
                else
                    out(i, j, k, n) +=
                        coeff * divVanLeerCell(phi, fx, fy, fz, i, j, k, n, dhx, dhy, dhz);
            }
        );
    }
}

static void divQuickIbmAcc(
    amrex::MultiFab& out_mf,
    const amrex::MultiFab& phi_mf,
    const ibm::CellTypeFab& ct,
    const amrex::MultiFab& fx_mf,
    const amrex::MultiFab& fy_mf,
    const amrex::MultiFab& fz_mf,
    const amrex::Geometry& geom,
    amrex::Real coeff,
    int ncomp
)
{
    requireStencilGhosts("div_quick_acc_ibm", "the field", phi_mf, 2);
    requireStencilGhosts("div_quick_acc_ibm", "the cell_type marker", ct, 2);
    requireSameLayout("div_quick_acc_ibm", phi_mf, ct);

    const auto dx = geom.CellSizeArray();
    const amrex::Real dhx = dx[0], dhy = dx[1], dhz = dx[2];
    for (amrex::MFIter mfi(phi_mf); mfi.isValid(); ++mfi)
    {
        const amrex::Box& bx = mfi.validbox();
        auto const& phi = phi_mf.const_array(mfi);
        auto const& out = out_mf.array(mfi);
        auto const& m = ct.const_array(mfi);
        auto const& fx = fx_mf.const_array(mfi);
        auto const& fy = fy_mf.const_array(mfi);
        auto const& fz = fz_mf.const_array(mfi);
        amrex::ParallelFor(
            bx,
            ncomp,
            [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
            {
                if (solidWithinTwo(m, i, j, k))
                    out(i, j, k, n) +=
                        coeff * divUpwindCell(phi, fx, fy, fz, i, j, k, n, dhx, dhy, dhz);
                else
                    out(i, j, k, n) +=
                        coeff * divQuickCell(phi, fx, fy, fz, i, j, k, n, dhx, dhy, dhz);
            }
        );
    }
}

// Accumulate coeff * laplacian(phi) into out (constant gamma folded into coeff).
static void laplacianAcc(
    amrex::MultiFab& out_mf,
    const amrex::MultiFab& phi_mf,
    const amrex::Geometry& geom,
    amrex::Real coeff,
    int ncomp
)
{
    const auto dx = geom.CellSizeArray();
    const amrex::Real dhx = dx[0], dhy = dx[1], dhz = dx[2];
    for (amrex::MFIter mfi(phi_mf); mfi.isValid(); ++mfi)
    {
        const amrex::Box& bx = mfi.validbox();
        auto const& phi = phi_mf.const_array(mfi);
        auto const& out = out_mf.array(mfi);
        amrex::ParallelFor(
            bx,
            ncomp,
            [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
            { out(i, j, k, n) += coeff * laplacianCell(phi, i, j, k, n, dhx, dhy, dhz); }
        );
    }
}

// Accumulate coeff * grad(phi) into out: scalar phi (comp 0) → 3-component vector,
// central difference (phi_r - phi_l)/(2 dh) per spatial direction (cf. grad.py).
//
// `out` may carry FEWER than three components, and then only the leading ones are
// written (B36). `evaluate(Equation(exp.grad(T)))` on a scalar T sizes the backend's
// scratch source by the SOLVED field's component count — one — so the unguarded
// version wrote two components past the end of every fab, silently, on the one path
// that reaches it. `nc` is uniform over the launch, so the three branches cost a
// predicated store and nothing diverges; a three-component `out` (every other caller)
// is bitwise what it was.
static void gradAcc(
    amrex::MultiFab& out_mf,
    const amrex::MultiFab& phi_mf,
    const amrex::Geometry& geom,
    amrex::Real coeff
)
{
    const auto dx = geom.CellSizeArray();
    const amrex::Real dhx = dx[0], dhy = dx[1], dhz = dx[2];
    const int nc = out_mf.nComp();
    for (amrex::MFIter mfi(phi_mf); mfi.isValid(); ++mfi)
    {
        const amrex::Box& bx = mfi.validbox();
        auto const& phi = phi_mf.const_array(mfi);
        auto const& out = out_mf.array(mfi);
        amrex::ParallelFor(
            bx,
            [=] AMREX_GPU_DEVICE(int i, int j, int k)
            {
                if (nc > 0)
                    out(i, j, k, 0) +=
                        coeff * (phi(i + 1, j, k, 0) - phi(i - 1, j, k, 0)) / (2.0 * dhx);
                if (nc > 1)
                    out(i, j, k, 1) +=
                        coeff * (phi(i, j + 1, k, 0) - phi(i, j - 1, k, 0)) / (2.0 * dhy);
                if (nc > 2)
                    out(i, j, k, 2) +=
                        coeff * (phi(i, j, k + 1, 0) - phi(i, j, k - 1, 0)) / (2.0 * dhz);
            }
        );
    }
}

// Accumulate coeff * phi into out (pointwise source term).
static void
sourceAcc(amrex::MultiFab& out_mf, const amrex::MultiFab& phi_mf, amrex::Real coeff, int ncomp)
{
    for (amrex::MFIter mfi(phi_mf); mfi.isValid(); ++mfi)
    {
        const amrex::Box& bx = mfi.validbox();
        auto const& phi = phi_mf.const_array(mfi);
        auto const& out = out_mf.array(mfi);
        amrex::ParallelFor(
            bx,
            ncomp,
            [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
            { out(i, j, k, n) += coeff * phi(i, j, k, n); }
        );
    }
}

// Generic forward-Euler axpy: phi -= dt_over_coeff * src.
static void eulerUpdate(
    amrex::MultiFab& phi_mf, const amrex::MultiFab& src_mf, amrex::Real dt_over_coeff, int ncomp
)
{
    for (amrex::MFIter mfi(phi_mf); mfi.isValid(); ++mfi)
    {
        const amrex::Box& bx = mfi.validbox();
        auto const& phi = phi_mf.array(mfi);
        auto const& src = src_mf.const_array(mfi);
        amrex::ParallelFor(
            bx,
            ncomp,
            [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
            { phi(i, j, k, n) -= dt_over_coeff * src(i, j, k, n); }
        );
    }
}


// Forward Euler step: phi_new = phi - dt * (div(F,phi) - laplacian(nu,phi))
// VanLeer divergence + central-difference laplacian, ncomp components.
static void eulerStepVanLeerLap(
    amrex::MultiFab& phi_mf,
    const amrex::MultiFab& fx_mf,
    const amrex::MultiFab& fy_mf,
    const amrex::MultiFab& fz_mf,
    const amrex::Geometry& geom,
    amrex::Real dt,
    amrex::Real nu,
    int ncomp
)
{
    const auto dx = geom.CellSizeArray();
    const amrex::Real idx = 1.0 / dx[0];
    const amrex::Real idy = 1.0 / dx[1];
    const amrex::Real idz = 1.0 / dx[2];
    const amrex::Real idx2 = idx * idx;
    const amrex::Real idy2 = idy * idy;
    const amrex::Real idz2 = idz * idz;

    for (amrex::MFIter mfi(phi_mf); mfi.isValid(); ++mfi)
    {
        const amrex::Box& bx = mfi.validbox();
        auto const& phi = phi_mf.array(mfi);
        auto const& fx = fx_mf.const_array(mfi);
        auto const& fy = fy_mf.const_array(mfi);
        auto const& fz = fz_mf.const_array(mfi);

        for (int comp = 0; comp < ncomp; ++comp)
        {
            amrex::ParallelFor(
                bx,
                [=] AMREX_GPU_DEVICE(int i, int j, int k)
                {
                    // --- VanLeer divergence ---
                    amrex::Real divF = 0.0;

                    // x-direction
                    {
                        amrex::Real fl = fx(i, j, k);
                        amrex::Real fr = fx(i + 1, j, k);
                        amrex::Real sm2 = phi(i - 2, j, k, comp);
                        amrex::Real sm1 = phi(i - 1, j, k, comp);
                        amrex::Real s0 = phi(i, j, k, comp);
                        amrex::Real sp1 = phi(i + 1, j, k, comp);
                        amrex::Real sp2 = phi(i + 2, j, k, comp);

                        amrex::Real dl = s0 - sm1;
                        amrex::Real phi_l = (fl >= 0.0) ? sm1 + 0.5 * vanleerCorr(sm1 - sm2, dl)
                                                        : s0 - 0.5 * vanleerCorr(sp1 - s0, dl);
                        amrex::Real dr = sp1 - s0;
                        amrex::Real phi_r = (fr >= 0.0) ? s0 + 0.5 * vanleerCorr(s0 - sm1, dr)
                                                        : sp1 - 0.5 * vanleerCorr(sp2 - sp1, dr);
                        divF += (fr * phi_r - fl * phi_l) * idx;
                    }
                    // y-direction
                    {
                        amrex::Real fl = fy(i, j, k);
                        amrex::Real fr = fy(i, j + 1, k);
                        amrex::Real sm2 = phi(i, j - 2, k, comp);
                        amrex::Real sm1 = phi(i, j - 1, k, comp);
                        amrex::Real s0 = phi(i, j, k, comp);
                        amrex::Real sp1 = phi(i, j + 1, k, comp);
                        amrex::Real sp2 = phi(i, j + 2, k, comp);

                        amrex::Real dl = s0 - sm1;
                        amrex::Real phi_l = (fl >= 0.0) ? sm1 + 0.5 * vanleerCorr(sm1 - sm2, dl)
                                                        : s0 - 0.5 * vanleerCorr(sp1 - s0, dl);
                        amrex::Real dr = sp1 - s0;
                        amrex::Real phi_r = (fr >= 0.0) ? s0 + 0.5 * vanleerCorr(s0 - sm1, dr)
                                                        : sp1 - 0.5 * vanleerCorr(sp2 - sp1, dr);
                        divF += (fr * phi_r - fl * phi_l) * idy;
                    }
                    // z-direction
                    {
                        amrex::Real fl = fz(i, j, k);
                        amrex::Real fr = fz(i, j, k + 1);
                        amrex::Real sm2 = phi(i, j, k - 2, comp);
                        amrex::Real sm1 = phi(i, j, k - 1, comp);
                        amrex::Real s0 = phi(i, j, k, comp);
                        amrex::Real sp1 = phi(i, j, k + 1, comp);
                        amrex::Real sp2 = phi(i, j, k + 2, comp);

                        amrex::Real dl = s0 - sm1;
                        amrex::Real phi_l = (fl >= 0.0) ? sm1 + 0.5 * vanleerCorr(sm1 - sm2, dl)
                                                        : s0 - 0.5 * vanleerCorr(sp1 - s0, dl);
                        amrex::Real dr = sp1 - s0;
                        amrex::Real phi_r = (fr >= 0.0) ? s0 + 0.5 * vanleerCorr(s0 - sm1, dr)
                                                        : sp1 - 0.5 * vanleerCorr(sp2 - sp1, dr);
                        divF += (fr * phi_r - fl * phi_l) * idz;
                    }

                    // --- Central-difference laplacian ---
                    amrex::Real s0 = phi(i, j, k, comp);
                    amrex::Real lap =
                        (phi(i + 1, j, k, comp) - 2.0 * s0 + phi(i - 1, j, k, comp)) * idx2
                        + (phi(i, j + 1, k, comp) - 2.0 * s0 + phi(i, j - 1, k, comp)) * idy2
                        + (phi(i, j, k + 1, comp) - 2.0 * s0 + phi(i, j, k - 1, comp)) * idz2;

                    // Forward Euler update
                    phi(i, j, k, comp) -= dt * (divF - nu * lap);
                }
            );
        }
    }
}


// Linear divergence + laplacian forward Euler step.
static void eulerStepLinearLap(
    amrex::MultiFab& phi_mf,
    const amrex::MultiFab& fx_mf,
    const amrex::MultiFab& fy_mf,
    const amrex::MultiFab& fz_mf,
    const amrex::Geometry& geom,
    amrex::Real dt,
    amrex::Real nu,
    int ncomp
)
{
    const auto dx = geom.CellSizeArray();
    const amrex::Real idx = 1.0 / dx[0];
    const amrex::Real idy = 1.0 / dx[1];
    const amrex::Real idz = 1.0 / dx[2];
    const amrex::Real idx2 = idx * idx;
    const amrex::Real idy2 = idy * idy;
    const amrex::Real idz2 = idz * idz;

    for (amrex::MFIter mfi(phi_mf); mfi.isValid(); ++mfi)
    {
        const amrex::Box& bx = mfi.validbox();
        auto const& phi = phi_mf.array(mfi);
        auto const& fx = fx_mf.const_array(mfi);
        auto const& fy = fy_mf.const_array(mfi);
        auto const& fz = fz_mf.const_array(mfi);

        for (int comp = 0; comp < ncomp; ++comp)
        {
            amrex::ParallelFor(
                bx,
                [=] AMREX_GPU_DEVICE(int i, int j, int k)
                {
                    amrex::Real divF = 0.0;

                    // x-direction: linear F = f * 0.5*(u_l + u_r)
                    {
                        amrex::Real fl = fx(i, j, k);
                        amrex::Real fr = fx(i + 1, j, k);
                        amrex::Real u_l = phi(i - 1, j, k, comp);
                        amrex::Real u_c = phi(i, j, k, comp);
                        amrex::Real u_r = phi(i + 1, j, k, comp);
                        divF += (fr * 0.5 * (u_c + u_r) - fl * 0.5 * (u_l + u_c)) * idx;
                    }
                    // y-direction
                    {
                        amrex::Real fl = fy(i, j, k);
                        amrex::Real fr = fy(i, j + 1, k);
                        amrex::Real u_l = phi(i, j - 1, k, comp);
                        amrex::Real u_c = phi(i, j, k, comp);
                        amrex::Real u_r = phi(i, j + 1, k, comp);
                        divF += (fr * 0.5 * (u_c + u_r) - fl * 0.5 * (u_l + u_c)) * idy;
                    }
                    // z-direction
                    {
                        amrex::Real fl = fz(i, j, k);
                        amrex::Real fr = fz(i, j, k + 1);
                        amrex::Real u_l = phi(i, j, k - 1, comp);
                        amrex::Real u_c = phi(i, j, k, comp);
                        amrex::Real u_r = phi(i, j, k + 1, comp);
                        divF += (fr * 0.5 * (u_c + u_r) - fl * 0.5 * (u_l + u_c)) * idz;
                    }

                    amrex::Real s0 = phi(i, j, k, comp);
                    amrex::Real lap =
                        (phi(i + 1, j, k, comp) - 2.0 * s0 + phi(i - 1, j, k, comp)) * idx2
                        + (phi(i, j + 1, k, comp) - 2.0 * s0 + phi(i, j - 1, k, comp)) * idy2
                        + (phi(i, j, k + 1, comp) - 2.0 * s0 + phi(i, j, k - 1, comp)) * idz2;

                    phi(i, j, k, comp) -= dt * (divF - nu * lap);
                }
            );
        }
    }
}


// Upwind divergence + laplacian forward Euler step.
static void eulerStepUpwindLap(
    amrex::MultiFab& phi_mf,
    const amrex::MultiFab& fx_mf,
    const amrex::MultiFab& fy_mf,
    const amrex::MultiFab& fz_mf,
    const amrex::Geometry& geom,
    amrex::Real dt,
    amrex::Real nu,
    int ncomp
)
{
    const auto dx = geom.CellSizeArray();
    const amrex::Real idx = 1.0 / dx[0];
    const amrex::Real idy = 1.0 / dx[1];
    const amrex::Real idz = 1.0 / dx[2];
    const amrex::Real idx2 = idx * idx;
    const amrex::Real idy2 = idy * idy;
    const amrex::Real idz2 = idz * idz;

    for (amrex::MFIter mfi(phi_mf); mfi.isValid(); ++mfi)
    {
        const amrex::Box& bx = mfi.validbox();
        auto const& phi = phi_mf.array(mfi);
        auto const& fx = fx_mf.const_array(mfi);
        auto const& fy = fy_mf.const_array(mfi);
        auto const& fz = fz_mf.const_array(mfi);

        for (int comp = 0; comp < ncomp; ++comp)
        {
            amrex::ParallelFor(
                bx,
                [=] AMREX_GPU_DEVICE(int i, int j, int k)
                {
                    amrex::Real divF = 0.0;
                    amrex::Real s0 = phi(i, j, k, comp);

                    // x-direction: upwind
                    {
                        amrex::Real fl = fx(i, j, k);
                        amrex::Real fr = fx(i + 1, j, k);
                        amrex::Real Fl = fl * ((fl >= 0.0) ? phi(i - 1, j, k, comp) : s0);
                        amrex::Real Fr = fr * ((fr >= 0.0) ? s0 : phi(i + 1, j, k, comp));
                        divF += (Fr - Fl) * idx;
                    }
                    // y-direction
                    {
                        amrex::Real fl = fy(i, j, k);
                        amrex::Real fr = fy(i, j + 1, k);
                        amrex::Real Fl = fl * ((fl >= 0.0) ? phi(i, j - 1, k, comp) : s0);
                        amrex::Real Fr = fr * ((fr >= 0.0) ? s0 : phi(i, j + 1, k, comp));
                        divF += (Fr - Fl) * idy;
                    }
                    // z-direction
                    {
                        amrex::Real fl = fz(i, j, k);
                        amrex::Real fr = fz(i, j, k + 1);
                        amrex::Real Fl = fl * ((fl >= 0.0) ? phi(i, j, k - 1, comp) : s0);
                        amrex::Real Fr = fr * ((fr >= 0.0) ? s0 : phi(i, j, k + 1, comp));
                        divF += (Fr - Fl) * idz;
                    }

                    amrex::Real lap =
                        (phi(i + 1, j, k, comp) - 2.0 * s0 + phi(i - 1, j, k, comp)) * idx2
                        + (phi(i, j + 1, k, comp) - 2.0 * s0 + phi(i, j - 1, k, comp)) * idy2
                        + (phi(i, j, k + 1, comp) - 2.0 * s0 + phi(i, j, k - 1, comp)) * idz2;

                    phi(i, j, k, comp) -= dt * (divF - nu * lap);
                }
            );
        }
    }
}


// QUICK divergence + laplacian forward Euler step.
static void eulerStepQuickLap(
    amrex::MultiFab& phi_mf,
    const amrex::MultiFab& fx_mf,
    const amrex::MultiFab& fy_mf,
    const amrex::MultiFab& fz_mf,
    const amrex::Geometry& geom,
    amrex::Real dt,
    amrex::Real nu,
    int ncomp
)
{
    const auto dx = geom.CellSizeArray();
    const amrex::Real idx = 1.0 / dx[0];
    const amrex::Real idy = 1.0 / dx[1];
    const amrex::Real idz = 1.0 / dx[2];
    const amrex::Real idx2 = idx * idx;
    const amrex::Real idy2 = idy * idy;
    const amrex::Real idz2 = idz * idz;

    for (amrex::MFIter mfi(phi_mf); mfi.isValid(); ++mfi)
    {
        const amrex::Box& bx = mfi.validbox();
        auto const& phi = phi_mf.array(mfi);
        auto const& fx = fx_mf.const_array(mfi);
        auto const& fy = fy_mf.const_array(mfi);
        auto const& fz = fz_mf.const_array(mfi);

        for (int comp = 0; comp < ncomp; ++comp)
        {
            amrex::ParallelFor(
                bx,
                [=] AMREX_GPU_DEVICE(int i, int j, int k)
                {
                    amrex::Real divF = 0.0;

                    // x-direction: QUICK (3/8 downstream + 6/8 upwind - 1/8 far-upwind)
                    {
                        amrex::Real fl = fx(i, j, k);
                        amrex::Real fr = fx(i + 1, j, k);
                        amrex::Real sm2 = phi(i - 2, j, k, comp);
                        amrex::Real sm1 = phi(i - 1, j, k, comp);
                        amrex::Real s0 = phi(i, j, k, comp);
                        amrex::Real sp1 = phi(i + 1, j, k, comp);
                        amrex::Real sp2 = phi(i + 2, j, k, comp);

                        amrex::Real phi_l = (fl >= 0.0) ? 0.375 * s0 + 0.75 * sm1 - 0.125 * sm2
                                                        : 0.375 * sm1 + 0.75 * s0 - 0.125 * sp1;
                        amrex::Real phi_r = (fr >= 0.0) ? 0.375 * sp1 + 0.75 * s0 - 0.125 * sm1
                                                        : 0.375 * s0 + 0.75 * sp1 - 0.125 * sp2;
                        divF += (fr * phi_r - fl * phi_l) * idx;
                    }
                    // y-direction
                    {
                        amrex::Real fl = fy(i, j, k);
                        amrex::Real fr = fy(i, j + 1, k);
                        amrex::Real sm2 = phi(i, j - 2, k, comp);
                        amrex::Real sm1 = phi(i, j - 1, k, comp);
                        amrex::Real s0 = phi(i, j, k, comp);
                        amrex::Real sp1 = phi(i, j + 1, k, comp);
                        amrex::Real sp2 = phi(i, j + 2, k, comp);

                        amrex::Real phi_l = (fl >= 0.0) ? 0.375 * s0 + 0.75 * sm1 - 0.125 * sm2
                                                        : 0.375 * sm1 + 0.75 * s0 - 0.125 * sp1;
                        amrex::Real phi_r = (fr >= 0.0) ? 0.375 * sp1 + 0.75 * s0 - 0.125 * sm1
                                                        : 0.375 * s0 + 0.75 * sp1 - 0.125 * sp2;
                        divF += (fr * phi_r - fl * phi_l) * idy;
                    }
                    // z-direction
                    {
                        amrex::Real fl = fz(i, j, k);
                        amrex::Real fr = fz(i, j, k + 1);
                        amrex::Real sm2 = phi(i, j, k - 2, comp);
                        amrex::Real sm1 = phi(i, j, k - 1, comp);
                        amrex::Real s0 = phi(i, j, k, comp);
                        amrex::Real sp1 = phi(i, j, k + 1, comp);
                        amrex::Real sp2 = phi(i, j, k + 2, comp);

                        amrex::Real phi_l = (fl >= 0.0) ? 0.375 * s0 + 0.75 * sm1 - 0.125 * sm2
                                                        : 0.375 * sm1 + 0.75 * s0 - 0.125 * sp1;
                        amrex::Real phi_r = (fr >= 0.0) ? 0.375 * sp1 + 0.75 * s0 - 0.125 * sm1
                                                        : 0.375 * s0 + 0.75 * sp1 - 0.125 * sp2;
                        divF += (fr * phi_r - fl * phi_l) * idz;
                    }

                    amrex::Real s0 = phi(i, j, k, comp);
                    amrex::Real lap =
                        (phi(i + 1, j, k, comp) - 2.0 * s0 + phi(i - 1, j, k, comp)) * idx2
                        + (phi(i, j + 1, k, comp) - 2.0 * s0 + phi(i, j - 1, k, comp)) * idy2
                        + (phi(i, j, k + 1, comp) - 2.0 * s0 + phi(i, j, k - 1, comp)) * idz2;

                    phi(i, j, k, comp) -= dt * (divF - nu * lap);
                }
            );
        }
    }
}


// ---------------------------------------------------------------------------
// Divergence-only source kernels (no time step, write to separate output)
// ---------------------------------------------------------------------------

// Upwind divergence source term.
static void divUpwind(
    amrex::MultiFab& out_mf,
    const amrex::MultiFab& phi_mf,
    const amrex::MultiFab& fx_mf,
    const amrex::MultiFab& fy_mf,
    const amrex::MultiFab& fz_mf,
    const amrex::Geometry& geom
)
{
    const auto dx = geom.CellSizeArray();
    const amrex::Real idx = 1.0 / dx[0];
    const amrex::Real idy = 1.0 / dx[1];
    const amrex::Real idz = 1.0 / dx[2];

    for (amrex::MFIter mfi(phi_mf); mfi.isValid(); ++mfi)
    {
        const amrex::Box& bx = mfi.validbox();
        auto const& phi = phi_mf.const_array(mfi);
        auto const& out = out_mf.array(mfi);
        auto const& fx = fx_mf.const_array(mfi);
        auto const& fy = fy_mf.const_array(mfi);
        auto const& fz = fz_mf.const_array(mfi);

        amrex::ParallelFor(
            bx,
            [=] AMREX_GPU_DEVICE(int i, int j, int k)
            {
                amrex::Real s0 = phi(i, j, k);
                amrex::Real fl_x = fx(i, j, k), fr_x = fx(i + 1, j, k);
                amrex::Real fl_y = fy(i, j, k), fr_y = fy(i, j + 1, k);
                amrex::Real fl_z = fz(i, j, k), fr_z = fz(i, j, k + 1);
                out(i, j, k) = (fr_x * ((fr_x >= 0) ? s0 : phi(i + 1, j, k))
                                - fl_x * ((fl_x >= 0) ? phi(i - 1, j, k) : s0))
                                 * idx
                             + (fr_y * ((fr_y >= 0) ? s0 : phi(i, j + 1, k))
                                - fl_y * ((fl_y >= 0) ? phi(i, j - 1, k) : s0))
                                   * idy
                             + (fr_z * ((fr_z >= 0) ? s0 : phi(i, j, k + 1))
                                - fl_z * ((fl_z >= 0) ? phi(i, j, k - 1) : s0))
                                   * idz;
            }
        );
    }
}

// Linear divergence source term.
static void divLinear(
    amrex::MultiFab& out_mf,
    const amrex::MultiFab& phi_mf,
    const amrex::MultiFab& fx_mf,
    const amrex::MultiFab& fy_mf,
    const amrex::MultiFab& fz_mf,
    const amrex::Geometry& geom
)
{
    const auto dx = geom.CellSizeArray();
    const amrex::Real idx = 1.0 / dx[0];
    const amrex::Real idy = 1.0 / dx[1];
    const amrex::Real idz = 1.0 / dx[2];

    for (amrex::MFIter mfi(phi_mf); mfi.isValid(); ++mfi)
    {
        const amrex::Box& bx = mfi.validbox();
        auto const& phi = phi_mf.const_array(mfi);
        auto const& out = out_mf.array(mfi);
        auto const& fx = fx_mf.const_array(mfi);
        auto const& fy = fy_mf.const_array(mfi);
        auto const& fz = fz_mf.const_array(mfi);

        amrex::ParallelFor(
            bx,
            [=] AMREX_GPU_DEVICE(int i, int j, int k)
            {
                amrex::Real s0 = phi(i, j, k);
                out(i, j, k) = (fx(i + 1, j, k) * 0.5 * (s0 + phi(i + 1, j, k))
                                - fx(i, j, k) * 0.5 * (phi(i - 1, j, k) + s0))
                                 * idx
                             + (fy(i, j + 1, k) * 0.5 * (s0 + phi(i, j + 1, k))
                                - fy(i, j, k) * 0.5 * (phi(i, j - 1, k) + s0))
                                   * idy
                             + (fz(i, j, k + 1) * 0.5 * (s0 + phi(i, j, k + 1))
                                - fz(i, j, k) * 0.5 * (phi(i, j, k - 1) + s0))
                                   * idz;
            }
        );
    }
}

// VanLeer divergence source term.
static void divVanLeer(
    amrex::MultiFab& out_mf,
    const amrex::MultiFab& phi_mf,
    const amrex::MultiFab& fx_mf,
    const amrex::MultiFab& fy_mf,
    const amrex::MultiFab& fz_mf,
    const amrex::Geometry& geom
)
{
    const auto dx = geom.CellSizeArray();
    const amrex::Real idx = 1.0 / dx[0];
    const amrex::Real idy = 1.0 / dx[1];
    const amrex::Real idz = 1.0 / dx[2];

    for (amrex::MFIter mfi(phi_mf); mfi.isValid(); ++mfi)
    {
        const amrex::Box& bx = mfi.validbox();
        auto const& phi = phi_mf.const_array(mfi);
        auto const& out = out_mf.array(mfi);
        auto const& fx = fx_mf.const_array(mfi);
        auto const& fy = fy_mf.const_array(mfi);
        auto const& fz = fz_mf.const_array(mfi);

        amrex::ParallelFor(
            bx,
            [=] AMREX_GPU_DEVICE(int i, int j, int k)
            {
                amrex::Real divF = 0.0;
                // x
                {
                    amrex::Real fl = fx(i, j, k), fr = fx(i + 1, j, k);
                    amrex::Real sm1 = phi(i - 1, j, k), s0 = phi(i, j, k), sp1 = phi(i + 1, j, k);
                    amrex::Real sm2 = phi(i - 2, j, k), sp2 = phi(i + 2, j, k);
                    amrex::Real dl = s0 - sm1;
                    amrex::Real pl = (fl >= 0) ? sm1 + 0.5 * vanleerCorr(sm1 - sm2, dl)
                                               : s0 - 0.5 * vanleerCorr(sp1 - s0, dl);
                    amrex::Real dr = sp1 - s0;
                    amrex::Real pr = (fr >= 0) ? s0 + 0.5 * vanleerCorr(s0 - sm1, dr)
                                               : sp1 - 0.5 * vanleerCorr(sp2 - sp1, dr);
                    divF += (fr * pr - fl * pl) * idx;
                }
                // y
                {
                    amrex::Real fl = fy(i, j, k), fr = fy(i, j + 1, k);
                    amrex::Real sm1 = phi(i, j - 1, k), s0 = phi(i, j, k), sp1 = phi(i, j + 1, k);
                    amrex::Real sm2 = phi(i, j - 2, k), sp2 = phi(i, j + 2, k);
                    amrex::Real dl = s0 - sm1;
                    amrex::Real pl = (fl >= 0) ? sm1 + 0.5 * vanleerCorr(sm1 - sm2, dl)
                                               : s0 - 0.5 * vanleerCorr(sp1 - s0, dl);
                    amrex::Real dr = sp1 - s0;
                    amrex::Real pr = (fr >= 0) ? s0 + 0.5 * vanleerCorr(s0 - sm1, dr)
                                               : sp1 - 0.5 * vanleerCorr(sp2 - sp1, dr);
                    divF += (fr * pr - fl * pl) * idy;
                }
                // z
                {
                    amrex::Real fl = fz(i, j, k), fr = fz(i, j, k + 1);
                    amrex::Real sm1 = phi(i, j, k - 1), s0 = phi(i, j, k), sp1 = phi(i, j, k + 1);
                    amrex::Real sm2 = phi(i, j, k - 2), sp2 = phi(i, j, k + 2);
                    amrex::Real dl = s0 - sm1;
                    amrex::Real pl = (fl >= 0) ? sm1 + 0.5 * vanleerCorr(sm1 - sm2, dl)
                                               : s0 - 0.5 * vanleerCorr(sp1 - s0, dl);
                    amrex::Real dr = sp1 - s0;
                    amrex::Real pr = (fr >= 0) ? s0 + 0.5 * vanleerCorr(s0 - sm1, dr)
                                               : sp1 - 0.5 * vanleerCorr(sp2 - sp1, dr);
                    divF += (fr * pr - fl * pl) * idz;
                }
                out(i, j, k) = divF;
            }
        );
    }
}

// QUICK divergence source term.
static void divQuick(
    amrex::MultiFab& out_mf,
    const amrex::MultiFab& phi_mf,
    const amrex::MultiFab& fx_mf,
    const amrex::MultiFab& fy_mf,
    const amrex::MultiFab& fz_mf,
    const amrex::Geometry& geom
)
{
    const auto dx = geom.CellSizeArray();
    const amrex::Real idx = 1.0 / dx[0];
    const amrex::Real idy = 1.0 / dx[1];
    const amrex::Real idz = 1.0 / dx[2];

    for (amrex::MFIter mfi(phi_mf); mfi.isValid(); ++mfi)
    {
        const amrex::Box& bx = mfi.validbox();
        auto const& phi = phi_mf.const_array(mfi);
        auto const& out = out_mf.array(mfi);
        auto const& fx = fx_mf.const_array(mfi);
        auto const& fy = fy_mf.const_array(mfi);
        auto const& fz = fz_mf.const_array(mfi);

        amrex::ParallelFor(
            bx,
            [=] AMREX_GPU_DEVICE(int i, int j, int k)
            {
                amrex::Real divF = 0.0;
                // x
                {
                    amrex::Real fl = fx(i, j, k), fr = fx(i + 1, j, k);
                    amrex::Real sm2 = phi(i - 2, j, k), sm1 = phi(i - 1, j, k), s0 = phi(i, j, k);
                    amrex::Real sp1 = phi(i + 1, j, k), sp2 = phi(i + 2, j, k);
                    amrex::Real pl = (fl >= 0) ? 0.375 * s0 + 0.75 * sm1 - 0.125 * sm2
                                               : 0.375 * sm1 + 0.75 * s0 - 0.125 * sp1;
                    amrex::Real pr = (fr >= 0) ? 0.375 * sp1 + 0.75 * s0 - 0.125 * sm1
                                               : 0.375 * s0 + 0.75 * sp1 - 0.125 * sp2;
                    divF += (fr * pr - fl * pl) * idx;
                }
                // y
                {
                    amrex::Real fl = fy(i, j, k), fr = fy(i, j + 1, k);
                    amrex::Real sm2 = phi(i, j - 2, k), sm1 = phi(i, j - 1, k), s0 = phi(i, j, k);
                    amrex::Real sp1 = phi(i, j + 1, k), sp2 = phi(i, j + 2, k);
                    amrex::Real pl = (fl >= 0) ? 0.375 * s0 + 0.75 * sm1 - 0.125 * sm2
                                               : 0.375 * sm1 + 0.75 * s0 - 0.125 * sp1;
                    amrex::Real pr = (fr >= 0) ? 0.375 * sp1 + 0.75 * s0 - 0.125 * sm1
                                               : 0.375 * s0 + 0.75 * sp1 - 0.125 * sp2;
                    divF += (fr * pr - fl * pl) * idy;
                }
                // z
                {
                    amrex::Real fl = fz(i, j, k), fr = fz(i, j, k + 1);
                    amrex::Real sm2 = phi(i, j, k - 2), sm1 = phi(i, j, k - 1), s0 = phi(i, j, k);
                    amrex::Real sp1 = phi(i, j, k + 1), sp2 = phi(i, j, k + 2);
                    amrex::Real pl = (fl >= 0) ? 0.375 * s0 + 0.75 * sm1 - 0.125 * sm2
                                               : 0.375 * sm1 + 0.75 * s0 - 0.125 * sp1;
                    amrex::Real pr = (fr >= 0) ? 0.375 * sp1 + 0.75 * s0 - 0.125 * sm1
                                               : 0.375 * s0 + 0.75 * sp1 - 0.125 * sp2;
                    divF += (fr * pr - fl * pl) * idz;
                }
                out(i, j, k) = divF;
            }
        );
    }
}


// Build precomputed stencil offsets for all boxes in a MultiFab.
// Returns (base, fx_off, fy_off, fz_off) as Python lists of ints.
static nb::tuple buildStencilOffsets(
    const amrex::MultiFab& mf,
    const amrex::MultiFab& fx_mf,
    const amrex::MultiFab& fy_mf,
    const amrex::MultiFab& fz_mf,
    int ng
)
{
    nb::list base_list, fx_list, fy_list, fz_list;

    int cell_global_off = 0;
    int fx_global_off = 0;
    int fy_global_off = 0;
    int fz_global_off = 0;

    for (amrex::MFIter mfi(mf); mfi.isValid(); ++mfi)
    {
        const auto& bx = mfi.validbox();
        const auto lo = bx.smallEnd();
        const auto hi = bx.bigEnd();
        int nx = hi[0] - lo[0] + 1;
        int ny = hi[1] - lo[1] + 1;
        int nz = hi[2] - lo[2] + 1;
        int Nx_g = nx + 2 * ng;
        int Ny_g = ny + 2 * ng;

        for (int iz = 0; iz < nz; ++iz)
        {
            for (int it = 0; it < ny; ++it)
            {
                for (int ix = 0; ix < nx; ++ix)
                {
                    int i = ng + ix;
                    int j = ng + it;
                    int k = ng + iz;

                    base_list.append(cell_global_off + i + Nx_g * j + Nx_g * Ny_g * k);
                    fx_list.append(fx_global_off + ix + (nx + 1) * it + (nx + 1) * ny * iz);
                    fy_list.append(fy_global_off + ix + nx * it + nx * (ny + 1) * iz);
                    fz_list.append(fz_global_off + ix + nx * it + nx * ny * iz);
                }
            }
        }

        cell_global_off += Nx_g * Ny_g * (nz + 2 * ng);
        fx_global_off += (nx + 1) * ny * nz;
        fy_global_off += nx * (ny + 1) * nz;
        fz_global_off += nx * ny * (nz + 1);
    }

    return nb::make_tuple(base_list, fx_list, fy_list, fz_list);
}


void registerStencilKernels(nb::module_& m)
{
    m.def(
        "euler_step_vanleer_lap",
        [](amrex::MultiFab& phi,
           const amrex::MultiFab& fx,
           const amrex::MultiFab& fy,
           const amrex::MultiFab& fz,
           const amrex::Geometry& geom,
           double dt,
           double nu,
           int ncomp) { eulerStepVanLeerLap(phi, fx, fy, fz, geom, dt, nu, ncomp); },
        nb::arg("phi"),
        nb::arg("fx"),
        nb::arg("fy"),
        nb::arg("fz"),
        nb::arg("geom"),
        nb::arg("dt"),
        nb::arg("nu"),
        nb::arg("ncomp") = 1,
        "Forward Euler step with VanLeer div + laplacian (C++ baseline)."
    );

    m.def(
        "euler_step_linear_lap",
        [](amrex::MultiFab& phi,
           const amrex::MultiFab& fx,
           const amrex::MultiFab& fy,
           const amrex::MultiFab& fz,
           const amrex::Geometry& geom,
           double dt,
           double nu,
           int ncomp) { eulerStepLinearLap(phi, fx, fy, fz, geom, dt, nu, ncomp); },
        nb::arg("phi"),
        nb::arg("fx"),
        nb::arg("fy"),
        nb::arg("fz"),
        nb::arg("geom"),
        nb::arg("dt"),
        nb::arg("nu"),
        nb::arg("ncomp") = 1,
        "Forward Euler step with Linear div + laplacian (C++ baseline)."
    );

    m.def(
        "euler_step_upwind_lap",
        [](amrex::MultiFab& phi,
           const amrex::MultiFab& fx,
           const amrex::MultiFab& fy,
           const amrex::MultiFab& fz,
           const amrex::Geometry& geom,
           double dt,
           double nu,
           int ncomp) { eulerStepUpwindLap(phi, fx, fy, fz, geom, dt, nu, ncomp); },
        nb::arg("phi"),
        nb::arg("fx"),
        nb::arg("fy"),
        nb::arg("fz"),
        nb::arg("geom"),
        nb::arg("dt"),
        nb::arg("nu"),
        nb::arg("ncomp") = 1,
        "Forward Euler step with Upwind div + laplacian (C++ baseline)."
    );

    m.def(
        "euler_step_quick_lap",
        [](amrex::MultiFab& phi,
           const amrex::MultiFab& fx,
           const amrex::MultiFab& fy,
           const amrex::MultiFab& fz,
           const amrex::Geometry& geom,
           double dt,
           double nu,
           int ncomp) { eulerStepQuickLap(phi, fx, fy, fz, geom, dt, nu, ncomp); },
        nb::arg("phi"),
        nb::arg("fx"),
        nb::arg("fy"),
        nb::arg("fz"),
        nb::arg("geom"),
        nb::arg("dt"),
        nb::arg("nu"),
        nb::arg("ncomp") = 1,
        "Forward Euler step with QUICK div + laplacian (C++ baseline)."
    );

    // --- Divergence-only source kernels ---

    m.def(
        "div_upwind",
        [](amrex::MultiFab& out,
           const amrex::MultiFab& phi,
           const amrex::MultiFab& fx,
           const amrex::MultiFab& fy,
           const amrex::MultiFab& fz,
           const amrex::Geometry& geom) { divUpwind(out, phi, fx, fy, fz, geom); },
        nb::arg("out"),
        nb::arg("phi"),
        nb::arg("fx"),
        nb::arg("fy"),
        nb::arg("fz"),
        nb::arg("geom"),
        "Upwind divergence source term (C++ baseline)."
    );

    m.def(
        "div_linear",
        [](amrex::MultiFab& out,
           const amrex::MultiFab& phi,
           const amrex::MultiFab& fx,
           const amrex::MultiFab& fy,
           const amrex::MultiFab& fz,
           const amrex::Geometry& geom) { divLinear(out, phi, fx, fy, fz, geom); },
        nb::arg("out"),
        nb::arg("phi"),
        nb::arg("fx"),
        nb::arg("fy"),
        nb::arg("fz"),
        nb::arg("geom"),
        "Linear divergence source term (C++ baseline)."
    );

    m.def(
        "div_vanleer",
        [](amrex::MultiFab& out,
           const amrex::MultiFab& phi,
           const amrex::MultiFab& fx,
           const amrex::MultiFab& fy,
           const amrex::MultiFab& fz,
           const amrex::Geometry& geom) { divVanLeer(out, phi, fx, fy, fz, geom); },
        nb::arg("out"),
        nb::arg("phi"),
        nb::arg("fx"),
        nb::arg("fy"),
        nb::arg("fz"),
        nb::arg("geom"),
        "VanLeer divergence source term (C++ baseline)."
    );

    m.def(
        "div_quick",
        [](amrex::MultiFab& out,
           const amrex::MultiFab& phi,
           const amrex::MultiFab& fx,
           const amrex::MultiFab& fy,
           const amrex::MultiFab& fz,
           const amrex::Geometry& geom) { divQuick(out, phi, fx, fy, fz, geom); },
        nb::arg("out"),
        nb::arg("phi"),
        nb::arg("fx"),
        nb::arg("fy"),
        nb::arg("fz"),
        nb::arg("geom"),
        "QUICK divergence source term (C++ baseline)."
    );

    m.def(
        "build_stencil_offsets",
        &buildStencilOffsets,
        nb::arg("mf"),
        nb::arg("fx"),
        nb::arg("fy"),
        nb::arg("fz"),
        nb::arg("ng"),
        "Build precomputed stencil offsets (base, fx, fy, fz) as int32 arrays."
    );

    // ---- Simple kernels for benchmarking Pallas vs C++ ----

    m.def(
        "laplacian",
        [](amrex::MultiFab& out_mf, const amrex::MultiFab& phi_mf, const amrex::Geometry& geom)
        {
            const auto dx = geom.CellSizeArray();
            const amrex::Real idx2 = 1.0 / (dx[0] * dx[0]);
            const amrex::Real idy2 = 1.0 / (dx[1] * dx[1]);
            const amrex::Real idz2 = 1.0 / (dx[2] * dx[2]);

            for (amrex::MFIter mfi(phi_mf); mfi.isValid(); ++mfi)
            {
                const amrex::Box& bx = mfi.validbox();
                auto const& phi = phi_mf.const_array(mfi);
                auto const& out = out_mf.array(mfi);

                amrex::ParallelFor(
                    bx,
                    [=] AMREX_GPU_DEVICE(int i, int j, int k)
                    {
                        amrex::Real c = phi(i, j, k);
                        out(i, j, k) = (phi(i + 1, j, k) - 2.0 * c + phi(i - 1, j, k)) * idx2
                                     + (phi(i, j + 1, k) - 2.0 * c + phi(i, j - 1, k)) * idy2
                                     + (phi(i, j, k + 1) - 2.0 * c + phi(i, j, k - 1)) * idz2;
                    }
                );
            }
        },
        nb::arg("out"),
        nb::arg("phi"),
        nb::arg("geom"),
        "Pure laplacian: out(i,j,k) = lap(phi)."
    );

    m.def(
        "write_cell_idx",
        [](amrex::MultiFab& out_mf)
        {
            for (amrex::MFIter mfi(out_mf); mfi.isValid(); ++mfi)
            {
                const amrex::Box& bx = mfi.validbox();
                auto const& out = out_mf.array(mfi);
                const auto lo = bx.smallEnd();
                const int nx = bx.length(0);
                const int ny = bx.length(1);

                amrex::ParallelFor(
                    bx,
                    [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                        out(i, j, k) = static_cast<amrex::Real>(
                            (i - lo[0]) + nx * (j - lo[1]) + nx * ny * (k - lo[2])
                        );
                    }
                );
            }
        },
        nb::arg("out"),
        "Write flat cell index: out(i,j,k) = (i-lo) + nx*(j-lo) + nx*ny*(k-lo)."
    );

    // ---- Multi-component Laplacian ----

    m.def(
        "laplacian_ncomp",
        [](amrex::MultiFab& out_mf,
           const amrex::MultiFab& phi_mf,
           const amrex::Geometry& geom,
           int ncomp)
        {
            const auto dx = geom.CellSizeArray();
            const amrex::Real idx2 = 1.0 / (dx[0] * dx[0]);
            const amrex::Real idy2 = 1.0 / (dx[1] * dx[1]);
            const amrex::Real idz2 = 1.0 / (dx[2] * dx[2]);

            for (amrex::MFIter mfi(phi_mf); mfi.isValid(); ++mfi)
            {
                const amrex::Box& bx = mfi.validbox();
                auto const& phi = phi_mf.const_array(mfi);
                auto const& out = out_mf.array(mfi);

                amrex::ParallelFor(
                    bx,
                    ncomp,
                    [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
                    {
                        amrex::Real c = phi(i, j, k, n);
                        out(i, j, k, n) =
                            (phi(i + 1, j, k, n) - 2.0 * c + phi(i - 1, j, k, n)) * idx2
                            + (phi(i, j + 1, k, n) - 2.0 * c + phi(i, j - 1, k, n)) * idy2
                            + (phi(i, j, k + 1, n) - 2.0 * c + phi(i, j, k - 1, n)) * idz2;
                    }
                );
            }
        },
        nb::arg("out"),
        nb::arg("phi"),
        nb::arg("geom"),
        nb::arg("ncomp") = 1,
        "Multi-component laplacian: out(i,j,k,n) = lap(phi, n)."
    );

    // ---- Multi-component first-order upwind divergence ----

    m.def(
        "upwind_div_ncomp",
        [](amrex::MultiFab& out_mf,
           const amrex::MultiFab& phi_mf,
           const amrex::MultiFab& fx_mf,
           const amrex::MultiFab& fy_mf,
           const amrex::MultiFab& fz_mf,
           const amrex::Geometry& geom,
           int ncomp)
        {
            const auto dx = geom.CellSizeArray();
            const amrex::Real idx = 1.0 / dx[0];
            const amrex::Real idy = 1.0 / dx[1];
            const amrex::Real idz = 1.0 / dx[2];

            for (amrex::MFIter mfi(phi_mf); mfi.isValid(); ++mfi)
            {
                const amrex::Box& bx = mfi.validbox();
                auto const& phi = phi_mf.const_array(mfi);
                auto const& out = out_mf.array(mfi);
                auto const& fx = fx_mf.const_array(mfi);
                auto const& fy = fy_mf.const_array(mfi);
                auto const& fz = fz_mf.const_array(mfi);

                amrex::ParallelFor(
                    bx,
                    ncomp,
                    [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
                    {
                        // x-direction
                        amrex::Real fl_x = fx(i, j, k);
                        amrex::Real fr_x = fx(i + 1, j, k);
                        amrex::Real Fl_x =
                            fl_x * (fl_x >= 0 ? phi(i - 1, j, k, n) : phi(i, j, k, n));
                        amrex::Real Fr_x =
                            fr_x * (fr_x >= 0 ? phi(i, j, k, n) : phi(i + 1, j, k, n));

                        // y-direction
                        amrex::Real fl_y = fy(i, j, k);
                        amrex::Real fr_y = fy(i, j + 1, k);
                        amrex::Real Fl_y =
                            fl_y * (fl_y >= 0 ? phi(i, j - 1, k, n) : phi(i, j, k, n));
                        amrex::Real Fr_y =
                            fr_y * (fr_y >= 0 ? phi(i, j, k, n) : phi(i, j + 1, k, n));

                        // z-direction
                        amrex::Real fl_z = fz(i, j, k);
                        amrex::Real fr_z = fz(i, j, k + 1);
                        amrex::Real Fl_z =
                            fl_z * (fl_z >= 0 ? phi(i, j, k - 1, n) : phi(i, j, k, n));
                        amrex::Real Fr_z =
                            fr_z * (fr_z >= 0 ? phi(i, j, k, n) : phi(i, j, k + 1, n));

                        out(i, j, k, n) =
                            (Fr_x - Fl_x) * idx + (Fr_y - Fl_y) * idy + (Fr_z - Fl_z) * idz;
                    }
                );
            }
        },
        nb::arg("out"),
        nb::arg("phi"),
        nb::arg("fx"),
        nb::arg("fy"),
        nb::arg("fz"),
        nb::arg("geom"),
        nb::arg("ncomp") = 1,
        "Multi-component first-order upwind divergence."
    );

    // ---- Composable accumulate kernels + generic axpy (plan 03 §4) ----

    m.def(
        "div_upwind_acc",
        [](amrex::MultiFab& out,
           const amrex::MultiFab& phi,
           const amrex::MultiFab& fx,
           const amrex::MultiFab& fy,
           const amrex::MultiFab& fz,
           const amrex::Geometry& geom,
           double coeff,
           int ncomp) { divUpwindAcc(out, phi, fx, fy, fz, geom, coeff, ncomp); },
        nb::arg("out"),
        nb::arg("phi"),
        nb::arg("fx"),
        nb::arg("fy"),
        nb::arg("fz"),
        nb::arg("geom"),
        nb::arg("coeff") = 1.0,
        nb::arg("ncomp") = 1,
        "Accumulate coeff*div_upwind(phi) into out (out +=), ncomp-general."
    );

    m.def(
        "div_linear_acc",
        [](amrex::MultiFab& out,
           const amrex::MultiFab& phi,
           const amrex::MultiFab& fx,
           const amrex::MultiFab& fy,
           const amrex::MultiFab& fz,
           const amrex::Geometry& geom,
           double coeff,
           int ncomp) { divLinearAcc(out, phi, fx, fy, fz, geom, coeff, ncomp); },
        nb::arg("out"),
        nb::arg("phi"),
        nb::arg("fx"),
        nb::arg("fy"),
        nb::arg("fz"),
        nb::arg("geom"),
        nb::arg("coeff") = 1.0,
        nb::arg("ncomp") = 1,
        "Accumulate coeff*div_linear(phi) into out (out +=), ncomp-general."
    );

    m.def(
        "div_vanleer_acc",
        [](amrex::MultiFab& out,
           const amrex::MultiFab& phi,
           const amrex::MultiFab& fx,
           const amrex::MultiFab& fy,
           const amrex::MultiFab& fz,
           const amrex::Geometry& geom,
           double coeff,
           int ncomp) { divVanLeerAcc(out, phi, fx, fy, fz, geom, coeff, ncomp); },
        nb::arg("out"),
        nb::arg("phi"),
        nb::arg("fx"),
        nb::arg("fy"),
        nb::arg("fz"),
        nb::arg("geom"),
        nb::arg("coeff") = 1.0,
        nb::arg("ncomp") = 1,
        "Accumulate coeff*div_vanleer(phi) into out (out +=), ncomp-general."
    );

    m.def(
        "div_quick_acc",
        [](amrex::MultiFab& out,
           const amrex::MultiFab& phi,
           const amrex::MultiFab& fx,
           const amrex::MultiFab& fy,
           const amrex::MultiFab& fz,
           const amrex::Geometry& geom,
           double coeff,
           int ncomp) { divQuickAcc(out, phi, fx, fy, fz, geom, coeff, ncomp); },
        nb::arg("out"),
        nb::arg("phi"),
        nb::arg("fx"),
        nb::arg("fy"),
        nb::arg("fz"),
        nb::arg("geom"),
        nb::arg("coeff") = 1.0,
        nb::arg("ncomp") = 1,
        "Accumulate coeff*div_quick(phi) into out (out +=), ncomp-general."
    );

    // ---- The W1 siblings (design §5, Q42(a)) ----
    //
    // The same two kernels with the marker as an extra argument. Nothing routes
    // through them on any evaluate path: the scheme dispatch, the marker's
    // production allocation and the noIbm / absent-key routing are B36's. Those
    // two paths carry no marker at all and must keep using the plain kernels,
    // which the guards below would otherwise refuse.

    m.def(
        "div_vanleer_acc_ibm",
        [](amrex::MultiFab& out,
           const amrex::MultiFab& phi,
           const ibm::CellTypeFab& cell_type,
           const amrex::MultiFab& fx,
           const amrex::MultiFab& fy,
           const amrex::MultiFab& fz,
           const amrex::Geometry& geom,
           double coeff,
           int ncomp) { divVanLeerIbmAcc(out, phi, cell_type, fx, fy, fz, geom, coeff, ncomp); },
        nb::arg("out"),
        nb::arg("phi"),
        nb::arg("cell_type"),
        nb::arg("fx"),
        nb::arg("fy"),
        nb::arg("fz"),
        nb::arg("geom"),
        nb::arg("coeff") = 1.0,
        nb::arg("ncomp") = 1,
        "Accumulate coeff*div_vanleer(phi) into out (out +=), degraded to div_upwind at every "
        "cell whose width-2 stencil reads a SOLID marker (W1)."
    );

    m.def(
        "div_quick_acc_ibm",
        [](amrex::MultiFab& out,
           const amrex::MultiFab& phi,
           const ibm::CellTypeFab& cell_type,
           const amrex::MultiFab& fx,
           const amrex::MultiFab& fy,
           const amrex::MultiFab& fz,
           const amrex::Geometry& geom,
           double coeff,
           int ncomp) { divQuickIbmAcc(out, phi, cell_type, fx, fy, fz, geom, coeff, ncomp); },
        nb::arg("out"),
        nb::arg("phi"),
        nb::arg("cell_type"),
        nb::arg("fx"),
        nb::arg("fy"),
        nb::arg("fz"),
        nb::arg("geom"),
        nb::arg("coeff") = 1.0,
        nb::arg("ncomp") = 1,
        "Accumulate coeff*div_quick(phi) into out (out +=), degraded to div_upwind at every "
        "cell whose width-2 stencil reads a SOLID marker (W1)."
    );

    m.def(
        "laplacian_acc",
        [](amrex::MultiFab& out,
           const amrex::MultiFab& phi,
           const amrex::Geometry& geom,
           double coeff,
           int ncomp) { laplacianAcc(out, phi, geom, coeff, ncomp); },
        nb::arg("out"),
        nb::arg("phi"),
        nb::arg("geom"),
        nb::arg("coeff") = 1.0,
        nb::arg("ncomp") = 1,
        "Accumulate coeff*laplacian(phi) into out (constant gamma folded into coeff)."
    );

    m.def(
        "grad_acc",
        [](amrex::MultiFab& out,
           const amrex::MultiFab& phi,
           const amrex::Geometry& geom,
           double coeff) { gradAcc(out, phi, geom, coeff); },
        nb::arg("out"),
        nb::arg("phi"),
        nb::arg("geom"),
        nb::arg("coeff") = 1.0,
        "Accumulate coeff*grad(phi) (scalar->3-vector, central difference) into out."
    );

    m.def(
        "source_acc",
        [](amrex::MultiFab& out, const amrex::MultiFab& phi, double coeff, int ncomp)
        { sourceAcc(out, phi, coeff, ncomp); },
        nb::arg("out"),
        nb::arg("phi"),
        nb::arg("coeff") = 1.0,
        nb::arg("ncomp") = 1,
        "Accumulate coeff*phi into out (pointwise source term)."
    );

    m.def(
        "euler_update",
        [](amrex::MultiFab& phi, const amrex::MultiFab& src, double dt_over_coeff, int ncomp)
        { eulerUpdate(phi, src, dt_over_coeff, ncomp); },
        nb::arg("phi"),
        nb::arg("src"),
        nb::arg("dt_over_coeff"),
        nb::arg("ncomp") = 1,
        "Generic forward-Euler axpy: phi -= dt_over_coeff*src."
    );
}
