# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""Named gaps between the IBM API we want and the one we have.

The IBM suite is written against the *intended* public API, ahead of the
implementation — so a large part of it is red by construction. Each red test
carries one of the markers below instead of being deleted, weakened, or
quietly skipped, which makes the suite double as the backlog: ``pytest -rx``
prints exactly what is missing and why.

Every marker is ``strict=True`` on purpose. When the implementation lands the
test starts passing, strict-xfail turns that pass into a failure, and whoever
landed it is forced to come here and remove the marker. A non-strict xfail
would let a finished feature sit unnoticed behind a stale marker forever.

Two rules for using these:

* **Reference not-yet-existing names inside the test body, never at module
  scope.** ``xfail`` cannot rescue a collection error — an import of a symbol
  that does not exist yet takes the whole file down with it. Import it locally
  in the function so the ``ImportError``/``AttributeError`` lands inside the
  xfail.
* **If the gap aborts the process rather than raising** (an AMReX ``Abort``,
  a segfault), use ``pytest.mark.skip`` instead and say so in the reason —
  a dead interpreter reports nothing at all.
"""

import pytest


def _gap(task, reason):
    """A strict xfail naming the task that closes it."""
    return pytest.mark.xfail(reason=f"{task}: {reason}", strict=True, raises=None)


# -- measured refutations, awaiting the gate's judgement ---------------------

B18_NEUMANN_WALL_ACCURACY = _gap(
    "B18",
    "measured refutation (B16, 2026-07-27, cpp/CUDA, quiet GPU): the "
    "FixedGradient(1/R) wall on T = ln r does not converge. L-inf over the six "
    "meshes 32..80 is non-monotone and *rises* from n=48 on — wall "
    "1.137e-2, 1.225e-2, 1.561e-2, 1.290e-2 — fitting a wall order of 1.073 and "
    "an interior order of 0.851, against the same-solution FixedValue row's "
    "1.768/1.439 on identical meshes, body and driver. Not a transient — "
    "measured: re-driving the n=64 point to T_END=1.2 leaves wall 1.560674e-02 "
    "and interior 1.271948e-02 unchanged to every printed digit, so this is the "
    "steady state. Nothing is patched around "
    "it and MIN_ORDER is untouched (O3/O4) — whether a Neumann wall this "
    "inaccurate condemns the wall formula is the accuracy gate's judgement, B18 "
    "(G1; verification §9: if the discriminating quantities do not converge, "
    "cutCell is a restart of the numerics, not an extension)",
)

# -- decided, not yet built -------------------------------------------------

B26_STEADY_VALIDATION_MEASUREMENT = _gap(
    "B26",
    "**not a missing measurement — a missing capability.** B26 ran "
    "(2026-07-28) and measured what solve() can drive: the A1 annulus field "
    "row, band 1.768 / bulk 1.439 over the six meshes, recorded in "
    "plans/IBM/tasks.md §1. A1 is therefore no longer among this marker's "
    "users. The two rows still here cannot be posed at all through the public "
    "API, so no session can measure them until each gap is built: **A2** needs "
    "a field-independent (Su) drive — the row spells it exp.body_force(f, U) "
    "and `grep -r body_force src/blockamr/` returns nothing, so the row raises "
    "at term construction before any wall arithmetic (the same shape of "
    "blocker as B41_EXPLICIT_SOURCE_TERM; whether it folds into B41 is open — "
    "B41 wires a compiled source_acc kernel, A2 wants a callable per-component "
    "drive). **A3** needs a *spatial* surface datum for the rotating wall "
    "u = omega x r, and that is **not** what B42 built: B42 (2026-07-28) made "
    "FixedValue accept a callable of the evaluation time, spelled f(x, y, z, t) "
    "and evaluated at the wall foot points, which is what A4/A6 need; A3's row "
    "spells its datum f(x, y, z) — three arguments — so it now raises a "
    "TypeError on the arity instead of inside broadcast_gamma, and is no closer "
    "to measurable. Whether A3 is respelled or the datum is widened is decided "
    "by the session that next schedules A3 (review.md §4 Q25 OP-1)",
)

B27_UNSTEADY_VALIDATION_MEASUREMENT = _gap(
    "B27",
    "**the measurement is taken** — B27 ran on 2026-07-28 (cpp/CUDA) and every "
    "number is in plans/IBM/tasks.md §1, *Measured, B27*: A4's phase error is "
    "recorded (a common lag of -0.0246 rad = -3.69 timesteps, identical across "
    "Euler/RK2/RK4, inside PHASE_ATOL = 0.08) and A6 at alpha = 6 is recorded "
    "green. **One user is left: A6 at alpha = 3**, and what it carries is not an "
    "unrecorded number but a measured miss that no one may tune away (O3/O4). It "
    "overshoots the amplitude at the outermost station pair — 0.618431 against "
    "an exact 0.557549, +10.9 % against AMP_RTOL = 0.05 — while its phase error "
    "there (0.068 rad) is still inside PHASE_ATOL; the interior seven stations "
    "pass. **B47 diagnosed it (2026-07-28): mesh under-resolution, not a formula "
    "deficit.** Holding alpha = 3 and resolving delta with 7 and then 10 cells "
    "instead of SCALE_CELLS = 5 (mesh scaled, no test edited, no constant moved) "
    "takes the outermost amplitude error 10.920 % -> 5.018 % -> 2.342 % and the "
    "phase error 0.0677 -> 0.0362 -> 0.0183 rad — both monotone, both at "
    "O(dx^2) (fitted 2.22 and 1.89 over the 5 -> 10 factor of two), which is the "
    "(dx/delta)^2 scaling this file's own Resolution paragraph derives. So the "
    "row is red **waiting on a finer-mesh budget decision**, not on the wall "
    "formula: it clears AMP_RTOL at SCALE_CELLS = 10 and sits on the bound at 7, "
    "and SCALE_CELLS is shared with A4/A5 at a cost of SCALE_CELLS^2 steps on "
    "n^3 cells. Nothing here bears on B44/W6 — A6's walls are FixedValue. "
    "Numbers in plans/IBM/tasks.md §1, *Measured, B47*. A4's three rows and A6 "
    "at alpha = 6 lost this marker at B42 and stay "
    "green. **Both A8 rows lost it at B46** (2026-07-28) and are green: their "
    "bulk-*exactness* assertion was the mis-posed half, not the measurement, and "
    "the gate re-posed it as a characterization (*Judged, B18* item 5, review.md "
    "§4 Q27a) — the band drift the row was built to characterize was always 50x "
    "inside DRIFT_FRACTION_A8, and the bulk now asserts the measured 2.3941e-04 "
    "(RK2) / 2.3777e-04 (RK4) instead of bitwise zero",
)

# -- missing implementation -------------------------------------------------

T6_DIRECT_FORCING_ROWS = _gap(
    "T6", "directForcing still runs the jnp-mask path instead of wall rows"
)

T17_MOVING_BODIES = _gap("T17", "moving bodies and fresh cells are not implemented")

T18_FORCES = _gap("T18", "per-patch force and torque diagnostics are not implemented")

T19_FLUX_ROWS = _gap("T19", "flux rows / cutCell are gated on the T16 accuracy gate")

RECONSTRUCTION_ORDER = _gap(
    "T14",
    "reconstruction is trilinear (linear-exact) over one solid layer, so the "
    "ghost value of a *quadratic* field carries O(dx^2) and the exact discrete "
    "cancellation this row asserts breaks in the band (B11: 2.87e-3 over 160 of "
    "4096 cells at n = 32, against 5.1e-15 away from the wall); quadratic/MLS is "
    "not implemented. This is a *pointwise reconstruction* claim and nothing "
    "more. The clause 'so the wall is first order' was retired at B18/Q24 "
    "(2026-07-28): the steady solution error at the wall is measured at 1.885 "
    "(r2-value, pure wall), 1.944 (r4-value) and 1.768 (ln r) — ~second order, "
    "not first — and the local reconstruction order does not set it",
)

PENALIZATION = _gap("method", "the penalization method is not implemented")

CUT_CELL = _gap("method", "cutCell validates its schema and refuses to run")
