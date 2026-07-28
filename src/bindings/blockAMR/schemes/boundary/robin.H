// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

// THE wall formula (design §1.3's "C1 hole") — B30b.
//
// `closure(alpha, beta, gamma, d)` closes `alpha phi_w + beta dphi/dn|_w =
// gamma` against ONE field value taken at signed distance `d` from the surface
// along the outward normal, with a linear profile between the two:
//
//     phi(d) = phi_w + d dphi/dn|_w
//
// Eliminating one unknown at a time, with `den = beta - alpha d`,
//
//     dphi/dn|_w = (gamma - alpha phi_i) / den
//     phi_w      = (beta phi_i - d gamma) / den
//
// so both readings are affine in the same single field value (design §4, R1),
// and BOTH BC families drop out of the general form exactly. Dirichlet
// `(1, 0, value)` gives `phi_w = value` and the one-sided difference
// `(phi_i - value)/d`; Neumann `(0, 1, g)` gives `dphi/dn = g` and
// `phi_w = phi_i - d g`. **There is no branch on the BC type anywhere in this
// file, and there must never be one** (review.md §4 Q29(g)): the BC dependence
// IS the arithmetic.
//
// ---------------------------------------------------------------------------
// VERBATIM — DO NOT IMPROVE THIS FUNCTION (review.md §4 Q41, user decision)
// ---------------------------------------------------------------------------
// This is a transcription of v1's `src/blockamr/schemes/boundary/ghost_cell.py`
// `wall_closure`, character for character, in v1's operation order and v1's
// field order. It is not a design, and one of its two arms is KNOWN to be
// inaccurate:
//
//   The `beta != 0` arm imposes its flux over the STAIRCASE blocked-face area.
//   B44 measured that error to six significant digits on six meshes and refuted
//   two cell-local repairs; the obstruction is structural (the exact weight is
//   the face aperture, which the current fluid-centred row ownership cannot
//   recover — B44-R measured 21-48 % coverage). Q41 (user decision, 2026-07-28)
//   ports it VERBATIM: the port's bar is v1<->v2 parity, the three Neumann
//   accuracy rows keep their `B18_NEUMANN_WALL_ACCURACY` strict-xfail markers,
//   and the `cutCell` fallback is scoped AFTER G2. Changing one bit here moves
//   recorded numbers and breaks the only claim this port makes.
//
// B44's bitwise-proven `w = beta / (beta + alpha d)` vehicle — the place a
// post-G2 fallback has to stand, inside this function's body and without
// disturbing the alpha branch — is inherited as DOCUMENTATION ONLY. No field,
// no argument and no line of code expresses it here.
//
// ---------------------------------------------------------------------------
// THE TWO BRANCHES, AND THE TWO SIGNED ZEROS THAT ARE LOAD-BEARING
// ---------------------------------------------------------------------------
//   Dirichlet (1, 0):  den = -d      value_linear = 0.0 / -d = -0.0
//                                    value_constant ~ gamma, NOT exactly gamma
//                                    grad_linear = 1/d      grad_constant = -(gamma/d)
//   Neumann   (0, 1):  den = 1.0     value_linear = 1.0     value_constant = -(d gamma)
//                                    grad_linear = -0.0 / 1 = -0.0
//                                    grad_constant = gamma, exactly
//
// `value_linear` is `-0.0` on EVERY Dirichlet row and `grad_linear` is `-0.0`
// on every Neumann row. The second is the whole of B44's structural finding —
// at `alpha = 0` the `a`-write vanishes identically and the row keeps only its
// constant. Both are observable through `RecordSink::linear`, which stores what
// it is handed, and both are pinned by a row in
// `test/blockamr/test_ibm_robin_closure.py`. Do not compute a negation by
// subtraction (`0.0 - alpha`): that returns `+0.0` where v1 returns `-0.0`.
//
// ---------------------------------------------------------------------------
// THE POLE IS NOT GUARDED, DELIBERATELY (review.md §4 Q43(c))
// ---------------------------------------------------------------------------
// `den = beta - alpha d` reaches exactly zero for `Mixed(f)` with
// `d = (1 - f)/f` — the pole B44 flagged, and part of why the
// `w = beta/(beta + alpha d)` vehicle exists at all. v1 divides anyway and
// returns `+-inf`; nothing in v1 warns, checks or documents it. B30b adds NO
// raise and NO clamp: a raise where v1 returns a number is a behavioural change
// in the one session whose entire claim is that nothing changed. The `+-inf` is
// PINNED by a test row, so a later well-meaning guard turns a green row red and
// gets read as the behaviour change it is. Guarding it is a post-G2 question,
// beside the fallback.
//
// ---------------------------------------------------------------------------
// FLOATING POINT — THIS HEADER'S ARITHMETIC INHERITS THE INCLUDING TU'S FLAGS
// ---------------------------------------------------------------------------
// The parity bar is BITWISE against numpy, and numpy cannot fuse: each of `-`,
// `*`, `/` is a separate ufunc over a materialised array, so every intermediate
// is rounded. `beta - alpha * distance` contracted into an fma is ONE rounding
// where numpy does two, and it differs in the last bit exactly where
// `beta ~ alpha d` (measured: the `Mixed(0.6)` and near-pole `Mixed(0.95)`
// configurations of the test file, 6 of 6 reads each).
//
// This is a HEADER: its arithmetic is inlined into whichever TU includes it,
// WITH THAT TU'S FLAGS. Therefore —
//
//     ANY TU THAT INCLUDES THIS HEADER AND CLAIMS PARITY WITH v1 MUST COMPILE
//     WITH `-ffp-contract=off` / `--fmad=false`.
//
// Three TUs include it: `schemes/boundary/robin_closure.cpp` (B30b),
// `schemes/boundary/laplacian_ghost_cell.cpp` (B32, the `laplacian x ghostCell`
// pair) and `schemes/boundary/div_ghost_cell.cpp` (B33, the `div x ghostCell`
// pair). All three are on `src/bindings/blockAMR/CMakeLists.txt`'s per-file
// flags list, next to `ibm/ghost_cell.cpp`.
//
// **Q36 was RULED at B32: the per-file opt-in STAYS**, and the directory-level
// IEEE posture B30b recommended is refuted in part. A directory posture is
// really directory-wide flags plus an exclusion list, and the two lists fail in
// opposite directions: `schemes/stencil_kernels.cpp` must NOT take the flags
// (Q42 — they move A1/B41/A4/A8's recorded accuracy digits) and
// `schemes/boundary/wall_frame.cpp` shipped without them by its own recorded
// decision, its `GammaExpr::Harmonic` being a real contraction site. Forgetting
// an entry on an OPT-IN list turns that TU's own bitwise rows red — loud and
// immediate. Forgetting one on an opt-out list moves a tolerance row's digits
// and nothing notices.
//
// The opt-in list's known weakness — one CMakeLists edit away from being
// silently incomplete — is closed MECHANICALLY rather than by inverting it:
// `test_ibm_laplacian_ghost_cell.py`'s conformance row parses that
// `set_source_files_properties` call and asserts that every `.cpp` under
// `src/bindings/blockAMR/` including this header is on the list. B33, B34 and
// B48 therefore discover a missing flag as a red row, not as a last-bit
// mystery.
//
// ---------------------------------------------------------------------------
// WHAT IS NOT HERE
// ---------------------------------------------------------------------------
// No geometry read (review.md §4 Q34: alpha, beta, gamma and d all ARRIVE as
// arguments; `d` is the image point's distance from `GhostCellData`, which is
// the caller's business and B32's seam). No method's data type, no view, no
// fab, no operator, no `WALL_SCHEMES` entry. This file names nothing of ours,
// which is what keeps `robin.H` a method-agnostic name.
//
// In production `d > 0` always (`sdf + step`, both positive) while `at()`'s
// argument is routinely NEGATIVE — a ghost centre inside the body. Neither is
// asserted: an assert would be new behaviour.

#include <AMReX_Extension.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_REAL.H>

#include <type_traits>

namespace ibm
{

static_assert(
    std::is_same_v<amrex::Real, double>,
    "the closure's bar is bitwise parity with v1's float64 numpy `wall_closure`; a "
    "single-precision amrex::Real would pass assert_allclose while failing bit equality. "
    "Build with NeoN_DEFINE_DP_SCALAR ON."
);

//! `phi_w` and `dphi/dn|_w`, each affine in ONE field value (design §4, R1).
//!
//! Field order is v1's dataclass order and a reorder is a silent wrong answer,
//! so it is pinned by a row: value_linear, value_constant, grad_linear,
//! grad_constant.
//!
//! v1 carries `value_linear` / `grad_linear` as one number per ROW and the two
//! constants per (row, component), because only gamma is ever per-component.
//! v2's `closure()` is called per (cell, component) and recomputes all four
//! from identical inputs, so the bits are identical — the shapes differ, the
//! arithmetic does not.
struct WallClosure
{
    amrex::Real value_linear;
    amrex::Real value_constant;
    amrex::Real grad_linear;
    amrex::Real grad_constant;

    //! The closure read a THIRD way: the field at signed distance `d` from the
    //! surface along the normal (v1 `WallClosure.at`), `phi(d) = phi_w + d
    //! dphi/dn|_w`. The same object read again, NOT a new approximation.
    //! `schemes/boundary/div_ghost_cell.cpp` (B33) is the first caller of both,
    //! at `d_G = s_P + step*dx_d*n̂_d`; `grad` (B34) is the second. The
    //! laplacian calls neither (W5, design §6, api §5.3).
    //!
    //! `value_linear + d * grad_linear` is itself a contraction site — see the
    //! floating-point note in the file header.
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real atLinear(amrex::Real d) const noexcept
    {
        return value_linear + d * grad_linear;
    }

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real atConstant(amrex::Real d) const noexcept
    {
        return value_constant + d * grad_constant;
    }
};

//! v1's `wall_closure`, transcribed. Every operation, and the order of them, is
//! the numpy: C++ parses `-distance * gamma / den` as `((-distance) * gamma) /
//! den` and `-alpha / den` as `(-alpha) / den`, which is the parse numpy takes.
//!
//! Rewrites that are algebraically exact are still NOT made — `-(alpha/den)`,
//! `(gamma * -distance)/den`, `-(alpha*distance - beta)`, and above all
//! `-distance * grad_constant` (which saves a divide and is exact only in
//! exact arithmetic — measured to move bits). "Provably equal" is a claim
//! someone has to re-derive at every reading; verbatim is not.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE WallClosure
closure(amrex::Real alpha, amrex::Real beta, amrex::Real gamma, amrex::Real distance) noexcept
{
    const amrex::Real den = beta - alpha * distance;
    return WallClosure {
        beta / den,              // value_linear
        -distance * gamma / den, // value_constant
        -alpha / den,            // grad_linear
        gamma / den              // grad_constant
    };
}

} // namespace ibm
