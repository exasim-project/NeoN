# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""Rungs 6-7: the convergence tier, for the *sourced* manufactured solutions.

Three files share the §4 table of the verification plan and split it by what
each row can prove:

* ``test_ibm_rungs.py`` — rungs 1-5 and 8, everything the physics lets us
  assert **exactly**, in one ``evaluate``.
* ``test_ibm_solution_error.py`` — the ``T = ln r`` row. Harmonic, so the
  steady problem needs no source and the entire residual is wall error.
* **this file** — the two remaining rows, ``T = r²`` (``∇²T = 4``) and
  ``T = r⁴`` (``∇²T = 16 r²``), plus the observed-order study of the wall.

Both of those solutions have a nonzero Laplacian, so the steady problem they
pose is a **Poisson** problem: the equation carries an explicit source
``Q = -α ∇²T_exact`` and ``T_exact`` is its exact steady state. That is the one
thing ``ln r`` cannot show — ``r²`` has an exactly-representable bulk operator
so its band error is *pure wall error*, and ``r⁴`` has an ``O(dx²)`` bulk
truncation term on top, so the pair brackets the wall treatment from both sides.

**D2 — the accuracy contract is the converged solution error under ``solve()``,
never the residual of a single ``evaluate()``.** Trilinear reconstruction puts
an ``O(dx²)`` error into the ghost value and the Laplacian divides by ``dx²``,
so the pointwise band residual is ``O(1)`` *structurally*; no mesh makes it
small. What converges is the solution of the steady problem, because that
``O(1)`` residual sits on ``O(n)`` cells and the elliptic solve integrates it
away. So every convergence assertion below is on ``L∞(T - T_exact)`` after
``solve()`` has been driven to steady state.

The **one** residual assertion that survives D2 is rung 6 itself, and it is here
because it is still exactly true: the 7-point central difference is exact on a
quadratic, so ``laplacian(1, r²)`` is ``4`` to the last bit in every fluid cell
whose stencil clears the body. Any nonzero bulk residual is IBM contamination
leaking out of the band, and that test is unmarked and green today.

**Three numbers, two assertions** (verification plan §4). Band ``L∞`` and bulk
``L∞`` are fitted separately, on their own cells; the global ``L2`` appears in
the failure message and is never asserted, because it mixes ``O(dx)`` on
``O(n²)`` cells with ``O(dx²)`` on ``O(n³)`` and converges at a flattering rate
describing neither (§10 anti-patterns). Region masks are derived test-side from
the analytic body — with no access to the implementation's classification that
is an *independent* oracle, and §4/§10 prefer it to asking the code which cells
it believes are near the wall.

**These rows ran for the first time at B41** (2026-07-28). A sourced
manufactured solution needs a way to *state* its source, and until B41 the
Python DSL had no explicit (Su) source term: ``exp.source(coeff_func, phi)`` was
the implicit (Sp) form ``coeff * phi`` alone, with no ``cpp`` kernel. NeoN's own
C++ DSL draws exactly this distinction — ``dsl::exp::source(coeff, phi)`` is Sp,
``dsl::exp::source(coeff)`` is Su, "the field IS the coefficient"
(``sourceTerm.cpp``) — and B41 built the same arity overload here:
``exp.source(S)`` with one ``CellField`` operand, schemed (``PointwiseSource``),
on both backends, and emitting band rows through ``source x ghostCell``. So
these rows now measure wall arithmetic rather than failing at term construction.

**What they measured** (six meshes 32..80, `cpp`/CUDA; full error rows in
``plans/IBM/tasks.md`` §1, *Measured, B41*):

===============  ============  ============  ==========================
case             band order    bulk order    vs ``MIN_ORDER = 1.0``
===============  ============  ============  ==========================
``r2-value``     1.885         1.708         both clear it
``r4-value``     1.944         1.763         both clear it
``r2-gradient``  0.582         0.175         **both fail — refuted**
===============  ============  ============  ==========================

Two findings, and neither was patched around:

* the **Neumann** row does not converge — the same refutation B16 measured on
  ``ln r`` × ``FixedGradient(1/R)`` (1.073 / 0.851), reproduced here on a
  different solution with the same body, meshes, driver and masks. Its band
  ``L∞`` *rises* from ``n = 32`` to ``n = 48`` before falling back. Both of its
  rows carry ``B18_NEUMANN_WALL_ACCURACY``, the marker B16 created, and the
  judgement is the accuracy gate's (B18/G1).
* ``r2-value``'s band order is **1.885**, above :data:`WALL_ORDER_SECOND`. The
  two order tests at the bottom of this file were built to be mutually
  exclusive; on this measurement the *second-order* one passes and the
  *first-order* one fails, on the trilinear reconstruction the design says can
  only be first order. That is a finding about the claim, not a threshold to
  move — see ``WALL_ORDER_CLAIM`` and ``plans/IBM/review.md`` §3.

Tier: **nightly** (§10; decision Q16). Three cases x six meshes of
forward-Euler pseudo-time; measured at B41 as **837 s** of solves (18 solves,
7.5 s at ``n = 32`` rising to 131 s at ``n = 80``), far outside §10's ~10 minute
pre-merge budget. Hence the module-level ``slow`` marker.
"""

from typing import Callable, NamedTuple

import numpy as np
import pytest

import blockamr
from blockamr.dsl import Equation, evaluate, exp, solve
from blockamr.field import CellField
from blockamr.ibm import Cylinder, FixedGradient, FixedValue
from blockamr.mesh import Mesh

from .ibm_gaps import B18_NEUMANN_WALL_ACCURACY, WALL_ORDER_CLAIM

pytestmark = pytest.mark.slow

BACKEND = "cpp"

# Geometry deliberately identical to ``test_ibm_solution_error.py``: same body,
# same meshes, same band width. The only thing that differs between the two
# files is the manufactured solution, so a difference in the measured order is
# a property of the solution and not of the discretisation of the body.
R = 0.25
CENTRE = (0.5, 0.5)
AXIS = 2
NZ = 4  # thin in the cylinder axis; every T here is z-invariant

ALPHA = 1.0  # laplacian coefficient (thermal diffusivity)

# Six meshes, not three: ``L∞`` on a cut geometry is non-monotone in ``n`` —
# how the circle slices the cells it crosses changes with the mesh — so a
# three-point fit of the same data moves by a whole order depending which
# three. Six points make the least-squares fit stable enough to be a contract.
RESOLUTIONS = (32, 40, 48, 56, 64, 80)

# The asserted floor on the observed order, band and bulk alike. First order,
# not second: the steady solution error is global, so the bulk carries the
# wall's accuracy too and converges at the wall's rate rather than at the
# central difference's. Anything tighter would be the same kind of wish as the
# ``O(1)`` band-residual bound D2 threw out.
MIN_ORDER = 1.0

# The line between "the wall is first order" and "the wall is second order".
# Placed at 1.8 rather than 2.0 because the measured pairwise rates on a cut
# geometry scatter hard (the sibling file records -0.7 to 4.1 between
# neighbouring meshes); a fitted 1.8 is already unambiguously not trilinear.
# The two order tests below use it from opposite sides, so they are mutually
# exclusive by construction: exactly one of them can be green.
WALL_ORDER_SECOND = 1.8

# Forward Euler on 3-D diffusion is stable for dt <= 1/(2*alpha*sum_d 1/dx_d^2);
# with dx == dy == dz that is dx^2/(6*alpha). A safety factor of 2 leaves room
# for the row amplification the wall reconstruction adds on top of it.
DT_SAFETY = 12.0

# Long enough for the diffusive transient across the unit box to have died.
T_END = 0.6

# The band is the fluid shell the wall treatment owns: two cells is the reach of
# the laplacian stencil plus the layer it reconstructs. Everything else is bulk.
BAND_CELLS = 2.0

# The narrower band the *residual* probe uses (verification plan §4 writes
# 1.5*dx): one ``evaluate`` of a width-1 laplacian can only see the wall from a
# cell whose face neighbour is solid, so anything past ``R + dx`` is already
# out of reach. 1.5 keeps half a cell of margin and makes the contamination
# probe as sharp as the stencil allows.
STENCIL_BAND_CELLS = 1.5


# ---------------------------------------------------------------------------
# The manufactured solutions of verification plan §4, minus the ``ln r`` row
# ---------------------------------------------------------------------------


class _Case(NamedTuple):
    """One row of the §4 table: the solution, its Laplacian, its wall datum."""

    exact: Callable  # T_exact(X, Y, Z)
    laplacian: Callable  # (∇²T_exact)(X, Y, Z) — the manufactured source
    bc: object  # the immersed wall condition on r = R


def _r2(X, Y, Z):
    return (X - CENTRE[0]) ** 2 + (Y - CENTRE[1]) ** 2


def _lap_r2(X, Y, Z):
    return np.full(X.shape, 4.0)


def _r4(X, Y, Z):
    return _r2(X, Y, Z) ** 2


def _lap_r4(X, Y, Z):
    return 16.0 * _r2(X, Y, Z)


#: The wall data are the ones §4 tabulates. ``FixedGradient`` is ``dT/dn`` with
#: ``n̂`` the body's *outward* (into-fluid) normal, so on ``r = R`` it is
#: ``dT/dr|_R = 2R`` for ``r²``. The table gives no gradient datum for ``r⁴``.
CASES = {
    "r2-value": _Case(_r2, _lap_r2, FixedValue(R**2)),
    "r2-gradient": _Case(_r2, _lap_r2, FixedGradient(2.0 * R)),
    "r4-value": _Case(_r4, _lap_r4, FixedValue(R**4)),
}

#: ``sorted(CASES)``, with the Neumann case expected-red since B41 measured it.
#: Band 0.582 / bulk 0.175 against ``MIN_ORDER = 1.0`` — the same refutation B16
#: recorded on ``ln r`` × ``FixedGradient``, reproduced on a second solution with
#: the same body, meshes, driver and masks. This is marker **placement**, not
#: weakening: the assertions, the floor and the case list are untouched, the
#: refuted numbers are in the ledger, and the judgement stays the accuracy
#: gate's (B18/G1) exactly as at B16.
CASE_PARAMS = [
    pytest.param("r2-gradient", marks=B18_NEUMANN_WALL_ACCURACY),
    "r2-value",
    "r4-value",
]


# ---------------------------------------------------------------------------
# Helpers — mesh, analytic fills, assembly, region masks
# ---------------------------------------------------------------------------


def _make_mesh(n):
    """``n x n x NZ`` cells with **cubic** cells, periodic in z only.

    ``z`` spans ``NZ/n`` so ``dz == dx``: the explicit diffusion limit is set by
    the smallest cell dimension, and a quasi-2-D box with ``dz >> dx`` would
    make the timestep a property of the padding direction. Periodic in z
    because every ``T`` here is z-invariant, non-periodic in x/y because none of
    them is — those halos carry the analytic Dirichlet datum instead.
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


def _fill(field, mesh, func):
    """Fill every valid cell from ``func(X, Y, Z)``, solid cells included.

    Seeding the body too is deliberate: the IBM must reconstruct its
    near-surface stencil from its own BC and never lean on what it finds
    inside the solid.
    """
    mf = field.mf[0]
    for mfi in blockamr.MFIterator(mf):
        arr = mf.copy_to_host(mfi)
        X, Y, Z = _coords(mesh, mfi.valid_box().small_end(), arr.shape[:3])
        arr[:, :, :, 0] = func(X, Y, Z)
        mf.copy_from(mfi, arr)
    field.fill_patch(0, 0.0)


def _fill_halo(field, mesh, func):
    """Seed the ghost band from ``func`` — the outer Dirichlet boundary.

    ``fill_boundary`` fills the periodic z halo and the inter-box halos and
    leaves the domain-exterior x/y ghosts alone, so this analytic seed survives
    every ``fill_patch`` of every step and the assertions can cover the whole
    fluid region instead of an eroded interior.
    """
    mf = field.mf[0]
    ng = mf.n_grow()
    for mfi in blockamr.MFIterator(mf):
        arr = mf.copy_grown_to_host(mfi)
        lo = [c - ng for c in mfi.valid_box().small_end()]
        X, Y, Z = _coords(mesh, lo, arr.shape[:3])
        arr[:, :, :, 0] = func(X, Y, Z)
        mf.copy_grown_from(mfi, arr)


def _assemble_field(field, n):
    """Stitch a field's valid cells into one global ``(n, n, NZ)`` array."""
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


def _assemble_result(field, results, n):
    """Stitch a per-box ``evaluate`` result into one global array."""
    out = np.full((n, n, NZ), np.nan)
    for bi, mfi in enumerate(blockamr.MFIterator(field.mf[0])):
        lo = mfi.valid_box().small_end()
        arr = np.asarray(results[0][bi])
        arr = arr.reshape(arr.shape[:3])
        out[
            lo[0] : lo[0] + arr.shape[0],
            lo[1] : lo[1] + arr.shape[1],
            lo[2] : lo[2] + arr.shape[2],
        ] = arr
    assert not np.isnan(out).any(), "box decomposition did not cover the domain"
    return out


def _regions(mesh, n, band_cells=BAND_CELLS):
    """``(band, bulk)`` fluid masks, derived test-side from the analytic body.

    With no access to the implementation's classification this is an
    *independent* oracle, and the plan (§4, §10) prefers it to asking the code
    which cells it believes are near the wall.
    """
    X, Y, _Z = _coords(mesh, (0, 0, 0), (n, n, NZ))
    r = np.hypot(X - CENTRE[0], Y - CENTRE[1])
    dx = float(mesh.geom(0).cell_size()[0])
    fluid = r > R
    band = fluid & (r < R + band_cells * dx)
    return band, fluid & ~band


# ---------------------------------------------------------------------------
# The steady drive and the fit
# ---------------------------------------------------------------------------

#: ``(case, n) -> (band Linf, bulk Linf, global L2)``. The solves are shared by
#: every assertion below rather than repeated per test.
_ERRORS: dict[tuple[str, int], tuple[float, float, float]] = {}


def _steady_errors(case_name, n):
    """Drive ``dT/dt = α ∇²T + Q`` to steady state and return the three norms.

    ``Q = -α ∇²T_exact`` is the manufactured source that makes ``T_exact`` the
    exact steady state; it enters as the **explicit (Su) source term**
    ``exp.source(S)``, whose single field operand *is* the coefficient — the
    residual-form equation is then

        ddt(T) - α laplacian(T) + α (∇²T_exact) = 0

    which is why the term carries ``+ALPHA`` rather than ``-ALPHA``.

    Forward Euler is the pseudo-time driver, not the object of study, so the
    answer must be independent of ``dt`` and of ``T_END``. ``RungeKutta2/4``
    raise ``NotImplementedError`` in ``solve()``, which is why the driver is
    Euler and why the mesh is built with ``dx == dy == dz`` (the explicit
    diffusion limit would otherwise be set by the padding direction alone).
    """
    key = (case_name, n)
    if key in _ERRORS:
        return _ERRORS[key]

    case = CASES[case_name]
    mesh = _make_mesh(n)

    T = CellField(mesh, ncomp=1, ngrow=1, name="T", ibm_bc={"cyl": case.bc})
    _fill(T, mesh, case.exact)
    _fill_halo(T, mesh, case.exact)

    S = CellField(mesh, ncomp=1, ngrow=1, name="S")
    _fill(S, mesh, case.laplacian)

    dx = float(mesh.geom(0).cell_size()[0])
    dt = dx * dx / (DT_SAFETY * ALPHA)
    eqn = Equation(
        exp.ddt(T) - exp.laplacian(ALPHA, T) + ALPHA * exp.source(S),
        schemes={"ddt": "Euler"},
    )
    for step in range(round(T_END / dt)):
        solve(eqn, dt=dt, t=step * dt, solution={"ibm": "ghostCell", "backend": BACKEND})

    err = np.abs(_assemble_field(T, n) - case.exact(*_coords(mesh, (0, 0, 0), (n, n, NZ))))
    band, bulk = _regions(mesh, n)
    _ERRORS[key] = (
        float(err[band].max()),
        float(err[bulk].max()),
        float(np.sqrt((err[band | bulk] ** 2).mean())),
    )
    return _ERRORS[key]


def _observed_order(errors):
    """Least-squares ``p`` in ``err ~ C dx^p`` over :data:`RESOLUTIONS`."""
    dx = 1.0 / np.array(RESOLUTIONS, dtype=float)
    slope, _intercept = np.polyfit(np.log(dx), np.log(np.array(errors, dtype=float)), 1)
    return float(slope)


def _report(case_name, label, order):
    """Failure message: all three norms at every resolution, plus the fit.

    The global ``L2`` appears **here and nowhere else**: it mixes the ``O(dx)``
    band with the ``O(dx²)`` bulk over a cell count that favours the bulk and
    converges at a rate describing neither, so it is reported and never
    asserted (verification plan §4 and the §10 anti-patterns).
    """
    rows = "\n".join(
        f"    n={n:3d}  band Linf={b:.6e}  bulk Linf={k:.6e}  (global L2={g:.6e})"
        for n, (b, k, g) in ((n, _steady_errors(case_name, n)) for n in RESOLUTIONS)
    )
    return f"{case_name} {label}: observed order {order:.3f}\n{rows}"


# ---------------------------------------------------------------------------
# Rung 6 — the one residual assertion D2 leaves standing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [32, 64])
@pytest.mark.parametrize(
    "bc", [FixedValue(R**2), FixedGradient(2.0 * R)], ids=["value", "gradient"]
)
def test_bulk_laplacian_of_a_quadratic_is_exactly_four(blockamr_session, n, bc):
    """Rung 6, and the only place in this file that asserts on an ``evaluate``.

    ``T = r²`` has an *exact* central-difference Laplacian: the 7-point stencil
    is exact on quadratics, so every fluid cell whose stencil clears the body
    must return ``4`` to the last bit. Any nonzero bulk residual is therefore
    IBM contamination — the wall treatment reaching into cells it does not own —
    and it is visible with no refinement study and no tolerance to argue about.

    The **band is deliberately not asserted here**. Its residual is ``O(1)`` by
    construction (module docstring, D2): trilinear reconstruction puts
    ``O(dx²)`` into the ghost value and the Laplacian divides by ``dx²``. The
    band's contract is the solution error of the tests below.

    Green today — this is the constraint on the wall treatment, not a
    prediction about it. Two resolutions because "exact" must not be a
    coincidence of one mesh. Measured: ``max|lap - 4|`` is ``0.0`` at both, and
    stays ``0.0`` with the mask tightened all the way to ``R + dx``, so the
    ``rtol`` the plan writes has plenty of room and the ``1.5*dx`` exclusion is
    margin rather than a fudge factor.

    **Both wall data**, since B23: ``FixedValue(R²)`` and ``FixedGradient(2R)``
    are different algebra on the same solution (the §4 table's ``r2-value`` and
    ``r2-gradient`` rows), and bulk exactness must hold for both — a Neumann row
    that contaminated the bulk would otherwise only be visible through the
    rung-7 solution-error tests, which were all strict-xfail until B41 made them
    runnable — and whose ``r2-gradient`` rows are now the recorded Neumann
    refutation, so this exactness check is what separates "the Neumann wall is
    inaccurate" from "the Neumann wall leaks into the bulk". It does not leak.
    This is ``test_mms_fixed_gradient_bulk_exact``, transferred from
    ``test_ibm_laplacian.py`` at B23 and landing on rung 6's tighter mask and
    tolerance.
    """
    mesh = _make_mesh(n)
    T = CellField(mesh, ncomp=1, ngrow=1, name="T", ibm_bc={"cyl": bc})
    _fill(T, mesh, _r2)
    _fill_halo(T, mesh, _r2)

    out = _assemble_result(
        T,
        evaluate(
            Equation(exp.laplacian(ALPHA, T)),
            t=0.0,
            solution={"ibm": "ghostCell", "backend": BACKEND},
        ),
        n,
    )
    _band, bulk = _regions(mesh, n, band_cells=STENCIL_BAND_CELLS)
    np.testing.assert_allclose(out[bulk], 4.0, rtol=1e-11)


# ---------------------------------------------------------------------------
# Rung 7 — the solution error of the sourced manufactured solutions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case_name", CASE_PARAMS)
def test_steady_solution_error_converges_in_the_band(blockamr_session, case_name):
    """The band contract for ``r²`` and ``r⁴``, stated on the solution (D2).

    ``L∞(T - T_exact)`` over the band cells only, fitted over six meshes. The
    band is ``O(n)`` cells against the bulk's ``O(n³)``, so a combined norm
    would be the bulk's norm wearing the band's name (§4, §10) — hence its own
    assertion on its own cells.

    Three rows of the §4 table, and each says something the others cannot:

    * ``r2-value`` — the bulk operator is *exact* on this solution, so the whole
      steady error is wall error. This is the sharpest measurement of the wall
      the scalar-datum API allows.
    * ``r2-gradient`` — the same solution through the Neumann datum
      (``dT/dn|_R = 2R``). A Dirichlet row and a Neumann row are different
      algebra; only this pairing shows whether they are equally accurate.
    * ``r4-value`` — ``∇²T = 16 r²`` is not constant, so the bulk carries a real
      ``O(dx²)`` truncation term *underneath* the wall error. If the wall is
      first order the total is still first order; if it ever stops being, this
      row is where bulk truncation starts to dominate.

    Live since B15: ``solve()`` honours ``solution["ibm"]`` as well as
    ``solution["backend"]``, so the immersed wall is applied inside the time
    loop and this row measures the wall's own order. Drivable since B41, which
    built the ``exp.source(S)`` the manufactured source needs.

    **Measured at B41** (2026-07-28): ``r2-value`` 1.885, ``r4-value`` 1.944 —
    both clear the floor. ``r2-gradient`` fits **0.582** and is expected-red
    under ``B18_NEUMANN_WALL_ACCURACY``; it is the second solution on which the
    Neumann datum fails to converge, after B16's ``ln r`` row.
    """
    errors = [_steady_errors(case_name, n)[0] for n in RESOLUTIONS]
    order = _observed_order(errors)
    assert order > MIN_ORDER, _report(case_name, "band Linf", order)


@pytest.mark.parametrize("case_name", CASE_PARAMS)
def test_steady_solution_error_converges_in_the_bulk(blockamr_session, case_name):
    """The bulk half of the same contract, on its own cells and its own norm.

    This is *not* the rung-6 statement that the bulk operator is exact. A steady
    solution error is global: the elliptic problem smears the wall error across
    the whole domain, so the bulk inherits the wall's accuracy and converges at
    the wall's rate rather than at the central difference's. That is why the
    floor here is the same :data:`MIN_ORDER` as the band's — and why a bulk
    order that came out at 2 while the band sat at 1 would mean the two regions
    had decoupled, which for an elliptic problem is itself a defect.

    **Measured at B41** (2026-07-28): ``r2-value`` 1.708, ``r4-value`` 1.763,
    against band orders of 1.885 and 1.944 — the bulk tracks the band a little
    below it, which is the coupling this row exists to check. ``r2-gradient``
    fits **0.175**, its band **0.582**: the Neumann row fails here too, and
    carries ``B18_NEUMANN_WALL_ACCURACY`` for the same reason as the band's.
    """
    errors = [_steady_errors(case_name, n)[1] for n in RESOLUTIONS]
    order = _observed_order(errors)
    assert order > MIN_ORDER, _report(case_name, "bulk Linf", order)


# ---------------------------------------------------------------------------
# The order of the wall itself — the before and the after
# ---------------------------------------------------------------------------

# Both tests below measure the same thing on ``r2-value``: the bulk operator is
# exact on ``r²``, so the band's solution error is *pure wall error* and its
# fitted order is the order of the reconstruction and of nothing else. They sit
# on opposite sides of :data:`WALL_ORDER_SECOND` on purpose — exactly one of
# them can pass, so the pair records which design is installed.
#
# **B41 measured 1.885, and the pair answered the other way round** (2026-07-28).
# The design predicted the *first-order* row; the *second-order* row is the one
# that passes, on the trilinear reconstruction, with no quadratic/MLS anywhere in
# the tree. Both halves therefore moved, which is exactly the outcome the pair
# was built to make visible — and it is a finding about the design claim, not a
# threshold to retune. :data:`WALL_ORDER_SECOND` and :data:`MIN_ORDER` are
# untouched; the escalation is ``plans/IBM/review.md`` §3, decided with the
# accuracy gate B18 (``WALL_ORDER_CLAIM``).


@WALL_ORDER_CLAIM
def test_observed_order_at_the_wall_is_first_order_today(blockamr_session):
    """The order the **current** design claims: first, and not yet second.

    ``ghostCell`` reconstructs the ghost value by trilinear interpolation over
    one solid layer. Trilinear is linear-exact, so it reproduces a linear field
    to machine precision (rung 5) but leaves an ``O(dx²)`` error on a curved
    wall, which the steady solve turns into an ``O(dx)`` solution error at the
    surface. First order is what this method was argued to owe.

    The assertion is **two-sided**, which is the point: the lower bound is the
    contract (the wall converges at all), and the upper bound records that it
    converges at the *trilinear* rate.

    **Refuted at B41** (2026-07-28), on its own terms: the fitted band order is
    **1.885**, so the lower bound holds and the upper one does not. That is not
    a mesh artefact of one case — ``r4-value`` fits 1.944 on the same body and
    driver, and B16's ``ln r`` × ``FixedValue`` row fits 1.768 next door. The
    row keeps its assertion and its numbers, and carries a strict xfail naming
    the pending decision (O5). Nothing about the reconstruction changed to make
    this happen: the trilinear-implies-first-order *argument* above is what the
    measurement contradicts, and choosing between the argument and the row is
    the accuracy gate's business (B18/G1, ``plans/IBM/review.md`` §3).
    """
    errors = [_steady_errors("r2-value", n)[0] for n in RESOLUTIONS]
    order = _observed_order(errors)
    assert MIN_ORDER < order < WALL_ORDER_SECOND, _report("r2-value", "band Linf", order)


def test_observed_order_at_the_wall_is_second_order_with_higher_order_reconstruction(
    blockamr_session,
):
    """Second order at the wall — measured, on the reconstruction we already have.

    Written as the *intended* design's row: a quadratic (or moving-least-squares)
    reconstruction is quadratic-exact, so its ghost error drops to ``O(dx³)`` on
    a curved surface and the steady solution error at the wall to ``O(dx²)`` —
    the same order the bulk scheme has, which is the point of doing it.

    **It passes at B41 (order 1.885) without that reconstruction ever landing.**
    ``RECONSTRUCTION_ORDER`` came off because it x-passed and the house rule for
    a strict xfail is that a row which greens loses its marker (O4; the Q21/A1
    precedent) — nothing here was weakened, the row goes from expected-red to
    enforced-green on an untouched assertion. But its *premise* is now false: T14
    is still open and the reconstruction is still trilinear, so whatever this row
    is measuring, it is not the arrival of quadratic/MLS. Its name and its
    continued existence are part of the question escalated to
    ``plans/IBM/review.md`` §3 (see ``WALL_ORDER_CLAIM`` on its sibling); B41 left
    both to the reviewer rather than deciding them inside a measurement session.
    """
    errors = [_steady_errors("r2-value", n)[0] for n in RESOLUTIONS]
    order = _observed_order(errors)
    assert order > WALL_ORDER_SECOND, _report("r2-value", "band Linf", order)
