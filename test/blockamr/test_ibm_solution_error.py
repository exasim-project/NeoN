# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""The wall accuracy contract: the error of the converged **solution**.

Companion to ``test_ibm_rungs.py`` (the exact, single-``evaluate`` ladder). This
file carries the one thing that ladder cannot express: how accurate the wall
treatment is, as opposed to whether it is consistent.

**The contract is a solution error, not an operator residual.** Trilinear
reconstruction puts an ``O(dx^2)`` error into the ghost value; the laplacian
divides by ``dx^2``; so the pointwise wall-region *residual* of one ``evaluate``
is ``O(1)`` **by construction**. That is not a defect and there is no mesh fine
enough to make it small — asserting a bound like ``< 6/n`` on it asserts a wish.
What does converge is the solution of the steady problem, because the ``O(1)``
wall residual acts on ``O(n)`` cells out of ``O(n^3)`` and is integrated away by
the elliptic solve. So the probe is ``solve()`` driven to steady state, and the
metric is ``L-inf(T - T_exact)``, reported at the wall and in the interior
separately.

The case is A1 of the verification plan §9 (concentric conduction) in its
sharpest form: ``T = ln r`` has ``laplacian(T) = 0`` identically, so the entire
steady residual is wall error and there is no interior truncation term to
subtract off. The outer boundary is the analytic value in the ghost band — an
exact Dirichlet condition on the box — and the inner boundary is the immersed
datum on ``r = R``. ``ln r`` is the unique harmonic function matching both, so
the exact solution is known in closed form. Both wall data of the §4 table are
driven: ``FixedValue(ln R)`` and ``FixedGradient(1/R)``, which are different
algebra on the same solution.

**Where this file stands.** It is the **D2** spec — the convergence contract in
its smallest honest form — and its Dirichlet rows are green, unmarked, since
``solve()`` began applying ``solution["ibm"]`` (B15; decision Q10 in
``plans/IBM/review.md`` §4). **B16** then landed the tabulated study on this
file: the ``ln r`` half of verification §4's manufactured-solution table, both
wall data, six meshes, wall and interior norms fitted separately, plus the
recorded order table of :data:`RECORDED_ORDERS`. The ``r²``/``r⁴`` half of the
table lives next door in ``test_ibm_convergence.py`` and is still red — not for
a wall-accuracy reason, but because a sourced manufactured solution needs an
explicit (Su) source term the Python DSL does not have yet (**B41**; decision
Q15 in ``plans/IBM/review.md`` §4).

**B16 refuted the Neumann row, and the refutation is recorded, not repaired.**
``FixedValue(ln R)`` converges cleanly (wall 1.768, interior 1.439);
``FixedGradient(1/R)`` — the *same* solution, mesh, body and driver, only the
wall algebra differs — does not converge at all: its ``L∞`` is non-monotone —
falling only to ``n = 48``, rising through 56 and 64, dipping again at 80 —
and the least-squares fits are 1.073 at the wall and
0.851 in the interior. The interior row is therefore left failing under a
strict xfail naming the accuracy gate (``B18_NEUMANN_WALL_ACCURACY``), and
:data:`MIN_ORDER` is **not** lowered, the case is **not** dropped and neither
:data:`T_END` nor :data:`DT_SAFETY` is retuned to flatter the fit (O3/O4). The
wall row's 1.073 is above the floor, but it is a fit through a sequence that
does not converge, so it should be read as part of the refutation rather than
as a pass. Whether this condemns the wall formula is B18's judgement.

Forward Euler is the pseudo-time driver, though ``RungeKutta2``/``RungeKutta4``
now exist too: it reaches steady state here and the transient is not the object
of study. The price is an explicit-diffusion timestep limit,
``dt < dx^2 / (2 * sum_d 1/dx_d^2)``, which is why the mesh is built with
``dx == dy == dz`` (an anisotropic cell would set the limit by the thin
direction alone) and why ``DT_SAFETY`` sits well inside it.

Tier: **nightly** (verification plan §10; decision Q16). Two cases x six meshes
of forward-Euler pseudo-time is 133 632 solve steps per case — measured 618 s
for the file on the ``cpp`` backend on the CUDA build, which is well outside
§10's ~10 minute pre-merge budget for the whole rung tier. Hence the
module-level ``slow`` marker.
"""

from typing import Callable, NamedTuple

import numpy as np
import pytest

import blockamr
from blockamr.dsl import Equation, exp, solve
from blockamr.field import CellField
from blockamr.ibm import Cylinder, FixedGradient, FixedValue
from blockamr.mesh import Mesh

from .ibm_gaps import B18_NEUMANN_WALL_ACCURACY

pytestmark = pytest.mark.slow

BACKEND = "cpp"

R = 0.25  # cylinder radius — large enough that the coarsest mesh resolves it
CENTRE = (0.5, 0.5)
AXIS = 2
NZ = 4  # thin in the cylinder axis; T is z-invariant

ALPHA = 1.0  # laplacian coefficient (thermal diffusivity)

# The refinement set of the order study. Three is the minimum an order can be
# fitted from; six because the L-inf error of a sharp-interface method on a cut
# geometry is *non-monotone* in n — how the circle slices the cells it crosses
# changes with the mesh, and a three-point fit of this data moves by a whole
# order depending which three (measured: interior 1.08 to 1.91). Six points make
# the least-squares fit stable enough to be a contract.
RESOLUTIONS = (32, 40, 48, 56, 64, 80)

# The asserted floor on the observed order, wall and interior alike. Not 2,
# though the reconstruction is linear-exact and the interior scheme is second
# order: with a reference stepper that does apply the wall condition, this case
# measures ~1.6 at the wall and ~1.4 in the interior, and the pairwise rates
# scatter from -0.7 to 4.1 across neighbouring resolutions for the geometric
# reason above. First order is what the method actually owes here; anything
# tighter would be the same kind of wish as the O(1) residual bound this file
# replaces.
MIN_ORDER = 1.0

# Forward Euler on 3-D diffusion is stable for dt <= 1/(2*alpha*sum_d 1/dx_d^2);
# with dx == dy == dz that is dx^2/(6*alpha). A safety factor of 2 leaves room
# for the row amplification the wall reconstruction adds on top of it.
DT_SAFETY = 12.0

# Long enough for the diffusive transient across the unit box to have died: the
# state at 0.6 and at 1.2 agree to ~1e-5, so this is the steady state and not a
# snapshot of one.
T_END = 0.6

# The wall region is the fluid shell the wall treatment owns; two cells is the
# reach of the laplacian stencil plus the layer it reconstructs. Everything else
# is interior.
BAND_CELLS = 2.0


# ---------------------------------------------------------------------------
# The ``ln r`` rows of the verification plan §4 table
# ---------------------------------------------------------------------------


class _Case(NamedTuple):
    """One row of the §4 table: the solution and its immersed wall datum.

    No manufactured source appears here, and that is the property that selects
    these two rows for this file: ``laplacian(ln r) == 0`` identically, so the
    steady problem is Laplace's and the whole residual is wall error. The §4
    rows whose Laplacian is nonzero pose a *Poisson* problem and need an
    explicit source term (``test_ibm_convergence.py``, B41).
    """

    exact: Callable  # T_exact(X, Y, Z)
    bc: object  # the immersed wall condition on r = R


def _ln_r(X, Y, Z):
    """``T = ln r`` about the cylinder axis. ``laplacian(T) == 0`` identically."""
    return np.log(np.hypot(X - CENTRE[0], Y - CENTRE[1]))


#: The two wall data verification plan §4 tabulates for ``ln r``.
#: ``FixedGradient`` is ``dT/dn`` with ``n̂`` the body's *outward* (into-fluid)
#: normal, so on ``r = R`` it is ``dT/dr|_R = 1/R``. The Neumann row stays
#: well posed because the outer box carries the exact Dirichlet datum in its
#: ghost band, which fixes the additive constant a pure Neumann problem leaves
#: free.
CASES = {
    "lnr-value": _Case(_ln_r, FixedValue(np.log(R))),
    "lnr-gradient": _Case(_ln_r, FixedGradient(1.0 / R)),
}

#: The tabulated study of B16 (``plans/IBM/tasks.md`` §1): the fitted order per
#: ``(case, region)`` over :data:`RESOLUTIONS`, measured 2026-07-27 on the
#: ``cpp`` backend on a quiet GPU. This is a *characterization*, not the
#: contract — the contract is ``order > MIN_ORDER`` and it is asserted
#: separately. To re-measure when the wall formula deliberately changes: set
#: this to ``None``, run the test, read the table out of the assertion message,
#: paste it back with the new date and the new ledger ID.
#:
#: **The ``lnr-gradient`` entries are a recorded refutation, not a target.**
#: The Neumann row's ``L∞`` is non-monotone — falling only to ``n = 48``, then
#: rising through 56 and 64 before dipping at 80 — so neither 1.073 nor 0.851
#: describes a converging sequence; they are here so that a change of mechanism
#: moves a *recorded* number rather than nothing.
#: The judgement is B18's (``B18_NEUMANN_WALL_ACCURACY``).
RECORDED_ORDERS = {
    ("lnr-gradient", "interior"): 0.851,
    ("lnr-gradient", "wall"): 1.073,
    ("lnr-value", "interior"): 1.439,
    ("lnr-value", "wall"): 1.768,
}

#: The magnitude guard the order fits cannot provide: a slope is scale-invariant,
#: so multiplying every error by ten would leave :data:`RECORDED_ORDERS` — and
#: every fit-based assertion — unchanged. The fact that carries the refutation
#: is a magnitude: the Neumann wall error is ~10x the Dirichlet one on the same
#: finest mesh. One number per case, the ``n = 80`` wall ``L∞`` as measured by
#: B16 (``plans/IBM/tasks.md`` §1), asserted to a generous relative tolerance —
#: wide enough for another machine, far too narrow for an order-of-magnitude
#: drift. Re-record together with :data:`RECORDED_ORDERS`.
RECORDED_WALL_LINF_80 = {
    "lnr-gradient": 1.289804e-02,
    "lnr-value": 1.335653e-03,
}

#: Relative tolerance on each entry of :data:`RECORDED_WALL_LINF_80`.
MAGNITUDE_RTOL = 0.25

#: Absolute tolerance on the fitted order, per entry of :data:`RECORDED_ORDERS`.
#: Wide enough that a re-run on another machine does not fail on the last digit,
#: narrow enough that a change of reconstruction (first order -> second) cannot
#: hide inside it.
ORDER_TOLERANCE = 0.3


# ---------------------------------------------------------------------------
# Helpers — mesh, analytic fills, assembly, region masks
# ---------------------------------------------------------------------------


def _make_mesh(n):
    """``n x n x NZ`` cells with **cubic** cells, periodic in z only.

    ``z`` spans ``NZ/n`` so ``dz == dx``: the explicit diffusion limit is set by
    the smallest cell dimension, and a quasi-2-D box with ``dz >> dx`` would make
    the timestep a property of the padding direction rather than of the study.
    Periodic in z because ``T`` is z-invariant, non-periodic in x/y because
    ``ln r`` is not — those halos carry the analytic Dirichlet datum instead.
    """
    box = blockamr.Box([0, 0, 0], [n - 1, n - 1, NZ - 1])
    rb = blockamr.RealBox([0.0, 0.0, 0.0], [1.0, 1.0, NZ / n])
    geom = blockamr.Geometry(box, rb, 0, [0, 0, 1])
    ba = blockamr.BoxArray(box)
    ba.max_size(max(n, NZ))
    dm = blockamr.DistributionMapping(ba)
    mesh = Mesh(ba, dm, geom)
    mesh.bodies = {"cyl": Cylinder(centre=CENTRE, radius=R, axis=AXIS)}
    return mesh


def _coords(mesh, lo, shape):
    """Cell-centre coordinate meshgrid for a box starting at index ``lo``."""
    geom = mesh.geom(0)
    dx = geom.cell_size()
    plo = geom.prob_lo()
    axes = [
        np.array([plo[d] + (lo[d] + i + 0.5) * dx[d] for i in range(shape[d])]) for d in range(3)
    ]
    return np.meshgrid(*axes, indexing="ij")


def _seed(field, mesh, func):
    """Fill valid cells *and* the ghost band from the exact solution.

    Valid cells: the initial condition (the exact solution is also the fixed
    point, so a converged run must return to it). Solid cells are seeded too —
    the IBM must reconstruct from its own BC, never lean on what it finds inside
    the body.

    Ghost band: the outer Dirichlet boundary. ``fill_boundary`` fills the
    periodic z halo and leaves the domain-exterior x/y ghosts alone, so this
    analytic seed survives every ``fill_patch`` of every step.
    """
    mf = field.mf[0]
    for mfi in blockamr.MFIterator(mf):
        arr = mf.copy_to_host(mfi)
        X, Y, Z = _coords(mesh, mfi.valid_box().small_end(), arr.shape[:3])
        arr[:, :, :, 0] = func(X, Y, Z)
        mf.copy_from(mfi, arr)
    field.fill_patch(0, 0.0)

    ng = mf.n_grow()
    for mfi in blockamr.MFIterator(mf):
        arr = mf.copy_grown_to_host(mfi)
        lo = [c - ng for c in mfi.valid_box().small_end()]
        X, Y, Z = _coords(mesh, lo, arr.shape[:3])
        arr[:, :, :, 0] = func(X, Y, Z)
        mf.copy_grown_from(mfi, arr)


def _assemble(field, n):
    """Stitch the field's valid cells into one global ``(n, n, NZ)`` array."""
    out = np.full((n, n, NZ), np.nan)
    mf = field.mf[0]
    for mfi in blockamr.MFIterator(mf):
        lo = mfi.valid_box().small_end()
        arr = np.asarray(mf.copy_to_host(mfi))
        arr = arr.reshape(arr.shape[:3])
        out[
            lo[0] : lo[0] + arr.shape[0],
            lo[1] : lo[1] + arr.shape[1],
            lo[2] : lo[2] + arr.shape[2],
        ] = arr
    assert not np.isnan(out).any(), "box decomposition did not cover the domain"
    return out


def _regions(mesh, n):
    """``(wall_region, interior)`` fluid masks, derived test-side from the body.

    With no access to the implementation's classification this is an
    *independent* oracle, and the plan (§4, §10) prefers it to asking the code
    which cells it believes are near the wall.
    """
    X, Y, _Z = _coords(mesh, (0, 0, 0), (n, n, NZ))
    r = np.hypot(X - CENTRE[0], Y - CENTRE[1])
    dx = float(mesh.geom(0).cell_size()[0])
    fluid = r > R
    wall_region = fluid & (r < R + BAND_CELLS * dx)
    return wall_region, fluid & ~wall_region


# ---------------------------------------------------------------------------
# The steady drive and the fit
# ---------------------------------------------------------------------------

#: ``(case, n) -> (wall Linf, interior Linf, global L2)``. The solves are shared
#: by every assertion below rather than repeated per test.
_ERRORS: dict[tuple[str, int], tuple[float, float, float]] = {}


def _steady_errors(case_name, n):
    """Drive ``dT/dt = alpha laplacian(T)`` to steady state and return the norms.

    The equation is the transient form of the steady problem the case actually
    poses; forward Euler is the pseudo-time driver, not the object of study, so
    the answer must be independent of ``dt`` and of ``T_END`` — it is, to ~1e-5.
    """
    key = (case_name, n)
    if key in _ERRORS:
        return _ERRORS[key]

    case = CASES[case_name]
    mesh = _make_mesh(n)
    T = CellField(mesh, ncomp=1, ngrow=1, name="T", ibm_bc={"cyl": case.bc})
    _seed(T, mesh, case.exact)

    dx = float(mesh.geom(0).cell_size()[0])
    dt = dx * dx / (DT_SAFETY * ALPHA)
    eqn = Equation(exp.ddt(T) - exp.laplacian(ALPHA, T), schemes={"ddt": "Euler"})
    for step in range(round(T_END / dt)):
        solve(eqn, dt=dt, t=step * dt, solution={"ibm": "ghostCell", "backend": BACKEND})

    err = np.abs(_assemble(T, n) - case.exact(*_coords(mesh, (0, 0, 0), (n, n, NZ))))
    wall_region, interior = _regions(mesh, n)
    _ERRORS[key] = (
        float(err[wall_region].max()),
        float(err[interior].max()),
        float(np.sqrt((err[wall_region | interior] ** 2).mean())),
    )
    return _ERRORS[key]


def _observed_order(errors):
    """Least-squares ``p`` in ``err ~ C dx^p`` over :data:`RESOLUTIONS`."""
    dx = 1.0 / np.array(RESOLUTIONS, dtype=float)
    slope, _intercept = np.polyfit(np.log(dx), np.log(np.array(errors, dtype=float)), 1)
    return float(slope)


def _rows(case_name):
    """The three norms of one case at every resolution, one line each.

    The global ``L2`` appears **here and nowhere else**. A single global ``L2``
    mixes the ``O(dx)`` wall region with the ``O(dx^2)`` interior over a cell
    count that favours the interior, and converges at a flattering rate
    describing neither — so it is reported and never asserted (verification plan
    §4 and the §10 anti-patterns).
    """
    return "\n".join(
        f"    n={n:3d}  wall Linf={w:.6e}  interior Linf={i:.6e}  (global L2={g:.6e})"
        for n, (w, i, g) in ((n, _steady_errors(case_name, n)) for n in RESOLUTIONS)
    )


def _report(case_name, label, order):
    """Failure message of one contract assertion: the fit and the six rows."""
    return f"{case_name} {label}: observed order {order:.3f}\n{_rows(case_name)}"


def _order_table():
    """``(case, region) -> fitted order`` for every case and both regions."""
    return {
        (case_name, region): _observed_order(
            [_steady_errors(case_name, n)[index] for n in RESOLUTIONS]
        )
        for case_name in sorted(CASES)
        for index, region in ((0, "wall"), (1, "interior"))
    }


def _table_report(table):
    """The whole study, in a form that can be pasted back into the constant."""
    blocks = "\n".join(
        f"  {case_name}: wall order {table[(case_name, 'wall')]:.3f}, "
        f"interior order {table[(case_name, 'interior')]:.3f}\n{_rows(case_name)}"
        for case_name in sorted(CASES)
    )
    literal = ",\n".join(f'    ("{c}", "{r}"): {p:.3f}' for (c, r), p in sorted(table.items()))
    return f"the B16 order table:\n{blocks}\n\nRECORDED_ORDERS = {{\n{literal},\n}}"


# ---------------------------------------------------------------------------
# The contract — the observed order of the steady solution error
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case_name", sorted(CASES))
def test_steady_solution_error_converges_at_the_wall(blockamr_session, case_name):
    """The wall half of the D2 contract, on its own cells and its own norm.

    ``L-inf(T - T_exact)`` over the wall-region cells only, fitted over six
    meshes. The wall region is ``O(n)`` cells against the interior's ``O(n^3)``,
    so a combined norm would be the interior's norm wearing the wall's name
    (verification plan §4, §10) — hence its own assertion on its own cells.

    Two rows, one per wall datum, and the pairing is the point: a Dirichlet row
    and a Neumann row are different algebra on the same solution, and only
    running both shows whether they are equally accurate. ``ln r`` is harmonic,
    so there is no interior truncation term underneath either of them and the
    fitted order is the order of the wall treatment and of nothing else.

    They are **not** equally accurate, and that is B16's finding: the measured
    fits are 1.768 (``FixedValue``) against 1.073 (``FixedGradient``), and the
    Neumann row's underlying ``L∞`` sequence rises from ``n = 48`` on. Its 1.073
    clears :data:`MIN_ORDER` only in the arithmetic sense — see
    :data:`RECORDED_ORDERS` and the module docstring; the interior half of the
    same row fails outright and carries the gate's marker.
    """
    errors = [_steady_errors(case_name, n)[0] for n in RESOLUTIONS]
    order = _observed_order(errors)
    assert order > MIN_ORDER, _report(case_name, "wall Linf", order)


@pytest.mark.parametrize(
    "case_name",
    [
        "lnr-value",
        # B16 measured this row at 0.851 and it is left failing under a strict
        # xfail naming the gate (review.md §4 Q15's refutation path): the floor
        # is not lowered, the case is not dropped, the mask is not widened and
        # neither T_END nor DT_SAFETY is retuned to flatter the fit.
        pytest.param("lnr-gradient", marks=B18_NEUMANN_WALL_ACCURACY),
    ],
)
def test_steady_solution_error_converges_in_the_interior(blockamr_session, case_name):
    """The interior half of the same contract, on its own cells and norm.

    Note what this is *not*: it is not the rung-6 statement that the interior
    operator is exact. A steady **solution** error is global — the elliptic
    problem smears the wall error across the whole domain, so the interior
    carries the wall's accuracy too and converges at the wall's rate, not at the
    central-difference scheme's. That is why the floor here is the same
    :data:`MIN_ORDER` as the wall's and not the scheme's own order, and why an
    interior order of 2 while the wall sat at 1 would mean the two regions had
    decoupled — which for an elliptic problem is itself a defect.

    The ``lnr-gradient`` row is red **as measured** (0.851; B16, 2026-07-27) and
    stays red under ``B18_NEUMANN_WALL_ACCURACY``. That the Neumann wall drags
    the *whole domain* below first order — not just the two cells next to the
    body — is precisely the global-smearing property this test exists to expose.
    """
    errors = [_steady_errors(case_name, n)[1] for n in RESOLUTIONS]
    order = _observed_order(errors)
    assert order > MIN_ORDER, _report(case_name, "interior Linf", order)


def test_the_wall_order_table_is_the_recorded_one(blockamr_session):
    """The tabulated study of B16, pinned against the numbers that were recorded.

    The contract tests above are one-sided: they hold the wall to
    ``order > MIN_ORDER`` and say nothing when the order *improves*. This one
    says what the order actually **is**, so that a change of mechanism — a new
    reconstruction, a different row assembly, a backend that rounds differently
    — shows up as a diff against a recorded number instead of disappearing into
    the slack of a floor. The recorded values live in
    ``plans/IBM/tasks.md`` §1 and ``plans/IBM/verification.md`` §4 as well
    (orchestration §3: no measurement without a ledger entry).

    When it fails, the message *is* the new table: the six ``(n, wall,
    interior, global L2)`` rows and both fitted orders per case, plus a literal
    ready to paste into :data:`RECORDED_ORDERS`. Re-recording is deliberate and
    leaves a date and a ledger ID behind; it is not a tolerance retune.
    """
    table = _order_table()
    assert RECORDED_ORDERS is not None, _table_report(table)
    assert sorted(RECORDED_ORDERS) == sorted(table), _table_report(table)
    drift = {
        key: (RECORDED_ORDERS[key], order)
        for key, order in table.items()
        if abs(order - RECORDED_ORDERS[key]) > ORDER_TOLERANCE
    }
    assert not drift, _table_report(table)


@pytest.mark.parametrize("case_name", sorted(CASES))
def test_the_finest_mesh_wall_error_is_the_recorded_magnitude(
    blockamr_session, case_name
):
    """The magnitude guard behind :data:`RECORDED_ORDERS` (B16 review).

    A fitted order is scale-invariant: multiply every error by ten and every
    slope — and every assertion above — is unchanged. But the refutation's
    substance *is* a magnitude (the Neumann wall error sits ~10x above the
    Dirichlet one on the same mesh), so one number per case is pinned here: the
    ``n = 80`` wall ``L∞``, to :data:`MAGNITUDE_RTOL`. When it fails, either
    the mechanism changed (re-record, with a date and a ledger ID, together
    with :data:`RECORDED_ORDERS`) or the wall arithmetic drifted (find out
    which before touching the constant).
    """
    wall, _interior, _l2 = _steady_errors(case_name, 80)
    recorded = RECORDED_WALL_LINF_80[case_name]
    assert np.isclose(wall, recorded, rtol=MAGNITUDE_RTOL), (
        f"{case_name}: n=80 wall Linf {wall:.6e} vs recorded {recorded:.6e} "
        f"(rtol {MAGNITUDE_RTOL})\n{_rows(case_name)}"
    )


# ---------------------------------------------------------------------------
# The two D2 spec rows, kept under their historical names (review.md §4 Q5)
# ---------------------------------------------------------------------------


def test_steady_solution_error_converges_in_the_bulk(blockamr_session):
    """The bulk half of the contract, on its own cells and its own norm.

    Note what this is *not*: it is not the rung-6 statement that the bulk
    operator is exact. A steady **solution** error is global — the elliptic
    problem smears the wall error across the whole domain, so the bulk carries
    the wall's accuracy too and converges at the wall's rate, not at the
    central-difference scheme's. That is why the floor here is the same
    :data:`MIN_ORDER` as the band's and not the scheme's own order.

    This is the D2 spec row that x-passed at B15 (review.md §4 Q10); it keeps
    its historical name (Q5) and is the ``lnr-value`` column of the study above.
    """
    errors = [_steady_errors("lnr-value", n)[1] for n in RESOLUTIONS]
    order = _observed_order(errors)
    assert order > MIN_ORDER, _report("lnr-value", "bulk Linf", order)


def test_steady_solution_error_converges_in_the_band(blockamr_session):
    """The band contract, stated on the solution and not on the residual.

    This is the assertion that replaces ``abs(out[band] - 4.0).max() < 6.0/n``.
    The single-``evaluate`` band residual that bound was placed on is ``O(1)`` by
    construction (module docstring) and never shrinks, so no mesh makes it pass.
    The steady solution error over the same cells does shrink, and *that* is the
    thing the wall treatment can be held to.

    Asserted **separately** from the bulk, on its own norm over its own cells —
    the band is ``O(n)`` cells against the bulk's ``O(n^3)``, so a combined norm
    is the bulk's norm wearing the band's name (verification plan §4, §10).

    This is the D2 spec row that x-passed at B15 (review.md §4 Q10); it keeps
    its historical name (Q5) and is the ``lnr-value`` column of the study above.
    """
    errors = [_steady_errors("lnr-value", n)[0] for n in RESOLUTIONS]
    order = _observed_order(errors)
    assert order > MIN_ORDER, _report("lnr-value", "band Linf", order)
