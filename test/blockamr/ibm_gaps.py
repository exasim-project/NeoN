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

WALL_ORDER_CLAIM = _gap(
    "Q24",
    "measured refutation of a *design claim*, not of the contract (B41, "
    "2026-07-28, cpp/CUDA): the steady band L-inf on T = r² with FixedValue(R²) "
    "— the pure-wall row, whose bulk operator is exact — fits an observed order "
    "of **1.885** over the six meshes 32..80, and T = r⁴ fits 1.944. Both are "
    "above WALL_ORDER_SECOND = 1.8, so the wall does *not* behave as the "
    "'trilinear reconstruction is linear-exact, therefore the wall is first "
    "order' argument this row states; its sibling "
    "test_observed_order_at_the_wall_is_second_order_with_higher_order_reconstruction "
    "x-passes on the same numbers, on the *trilinear* reconstruction, with no "
    "quadratic/MLS anywhere in the tree. The pair was built to be mutually "
    "exclusive, so both halves moving is a finding about the claim. Nothing was "
    "patched around it: WALL_ORDER_SECOND, MIN_ORDER, T_END, DT_SAFETY, "
    "RESOLUTIONS and the masks are untouched (O3/O4). Which half survives — and "
    "whether 'first order at the wall' was ever this method's ceiling — is a "
    "contract question escalated to plans/IBM/review.md §3 and decided with the "
    "accuracy gate B18",
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
    "solve() applies solution['ibm'] since B15, and the callable wall datum "
    "since B42 (2026-07-28), so every unsteady study now runs; the rows still "
    "marked are the ones that still miss their accuracy bounds — A6 at "
    "alpha = 3 and both A8 rows. A4's three rows and A6 at alpha = 6 x-passed "
    "at B42 and lost this marker there, with no number recorded: B27 is the "
    "Phase-2 session that measures and records the unsteady results",
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
    "wall is first order; quadratic/MLS is not implemented",
)

PENALIZATION = _gap("method", "the penalization method is not implemented")

CUT_CELL = _gap("method", "cutCell validates its schema and refuses to run")
