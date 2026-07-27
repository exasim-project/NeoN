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

B41_EXPLICIT_SOURCE_TERM = _gap(
    "B41",
    "the DSL has no explicit (Su) source term: exp.source(coeff, phi) is the "
    "implicit (Sp) form, Source carries no scheme and no cpp kernel (the cpp "
    "backend raises for it, pinned by a green test), and the band driver "
    "resolves a boundary scheme per spatial term. A sourced manufactured "
    "solution therefore cannot be stated, let alone driven — these rows fail at "
    "term construction, before any wall arithmetic. B41 wires the already "
    "compiled source_acc kernel through an explicit source term; the wall orders "
    "of the r²/r⁴ rows are then a short measurement session (decision Q15). The "
    "ln r half of the same §4 table is measured and recorded next door in "
    "test_ibm_solution_error.py (B16) — Dirichlet green, the Neumann interior "
    "row a recorded refutation awaiting the gate (B18)",
)

B26_STEADY_VALIDATION_MEASUREMENT = _gap(
    "B26",
    "solve() applies solution['ibm'] since B15, but the steady validation "
    "studies (A2/A3) still miss their accuracy bounds; B26 is the Phase-2 "
    "session that measures and records the steady validation orders (after "
    "the B16/D2 accuracy contract)",
)

B27_UNSTEADY_VALIDATION_MEASUREMENT = _gap(
    "B27",
    "solve() applies solution['ibm'] since B15, but the unsteady validation "
    "studies (A4/A6/A8) still miss their accuracy bounds; B27 is the Phase-2 "
    "session that measures and records the unsteady (Stokes A4) results",
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
