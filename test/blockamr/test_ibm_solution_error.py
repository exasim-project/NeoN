# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""The band accuracy contract: the error of the converged **solution**.

Companion to ``test_ibm_rungs.py`` (the exact, single-``evaluate`` ladder). This
file carries the one thing that ladder cannot express: how accurate the wall
treatment is, as opposed to whether it is consistent.

**The contract is a solution error, not an operator residual.** Trilinear
reconstruction puts an ``O(dx^2)`` error into the ghost value; the laplacian
divides by ``dx^2``; so the pointwise band *residual* of one ``evaluate`` is
``O(1)`` **by construction**. That is not a defect and there is no mesh fine
enough to make it small — asserting a bound like ``< 6/n`` on it asserts a wish.
What does converge is the solution of the steady problem, because the ``O(1)``
band residual acts on ``O(n)`` cells out of ``O(n^3)`` and is integrated away by
the elliptic solve. So the probe is ``solve()`` driven to steady state, and the
metric is ``L-inf(T - T_exact)``, reported in the band and the bulk separately.

The case is A1 of the verification plan §9 (concentric conduction) in its
sharpest form: ``T = ln r`` has ``laplacian(T) = 0`` identically, so the entire
steady residual is wall error and there is no bulk truncation term to subtract
off. The outer boundary is the analytic value in the ghost band — an exact
Dirichlet condition on the box — and the inner boundary is the immersed
``FixedValue(ln R)``. ``ln r`` is the unique harmonic function matching both, so
the exact solution is known in closed form.

**This file is red for two independent reasons, and both are the point:**

1. ``solve()`` never enters the IBM path. Its ``ForwardEuler`` branch calls
   ``impl.euler_step(...)`` directly; ``blockamr.ibm.evaluation`` is imported
   and used by ``evaluate()`` alone (``src/blockamr/dsl/solve.py``). A
   ``solution={"ibm": ...}`` key passed to ``solve()`` is read for ``"backend"``
   and otherwise ignored, so the immersed wall is simply absent and the field
   diffuses through the body.
2. The time integration a steady-state drive needs may not be there.
   ``RungeKutta2``/``RungeKutta4`` raise ``NotImplementedError`` in ``solve()``.
   Forward Euler does exist and does reach steady state here, so that is what
   this file uses — at the cost of an explicit-diffusion timestep limit,
   ``dt < dx^2 / (2 * sum_d 1/dx_d^2)``, which is why the mesh is built with
   ``dx == dy == dz`` (an anisotropic cell would set the limit by the thin
   direction alone) and why ``DT_SAFETY`` sits well inside it.

Tier: pre-merge (verification plan §10), a few seconds at these resolutions.
"""

import numpy as np

import blockamr
from blockamr.dsl import Equation, exp, solve
from blockamr.field import CellField
from blockamr.ibm import Cylinder, FixedValue
from blockamr.mesh import Mesh

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
# order depending which three (measured: bulk 1.08 to 1.91). Six points make the
# least-squares fit stable enough to be a contract.
RESOLUTIONS = (32, 40, 48, 56, 64, 80)

# The asserted floor on the observed order, band and bulk alike. Not 2, though
# the reconstruction is linear-exact and the bulk scheme is second order: with a
# reference stepper that does apply the wall condition, this case measures ~1.6
# in the band and ~1.4 in the bulk, and the pairwise rates scatter from -0.7 to
# 4.1 across neighbouring resolutions for the geometric reason above. First
# order is what the method actually owes here; anything tighter would be the
# same kind of wish as the O(1) residual bound this file replaces.
MIN_ORDER = 1.0

# Forward Euler on 3-D diffusion is stable for dt <= 1/(2*alpha*sum_d 1/dx_d^2);
# with dx == dy == dz that is dx^2/(6*alpha). A safety factor of 2 leaves room
# for the row amplification the wall reconstruction adds on top of it.
DT_SAFETY = 12.0

# Long enough for the diffusive transient across the unit box to have died: the
# state at 0.6 and at 1.2 agree to ~1e-5, so this is the steady state and not a
# snapshot of one.
T_END = 0.6

# The band is the fluid shell the wall treatment owns; two cells is the reach of
# the laplacian stencil plus the layer it reconstructs. Everything else is bulk.
BAND_CELLS = 2.0


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


def _exact(X, Y, Z):
    """``T = ln r`` about the cylinder axis. ``laplacian(T) == 0`` identically."""
    return np.log(np.hypot(X - CENTRE[0], Y - CENTRE[1]))


def _seed(field, mesh):
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
        arr[:, :, :, 0] = _exact(X, Y, Z)
        mf.copy_from(mfi, arr)
    field.fill_patch(0, 0.0)

    ng = mf.n_grow()
    for mfi in blockamr.MFIterator(mf):
        arr = mf.copy_grown_to_host(mfi)
        lo = [c - ng for c in mfi.valid_box().small_end()]
        X, Y, Z = _coords(mesh, lo, arr.shape[:3])
        arr[:, :, :, 0] = _exact(X, Y, Z)
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
    """``(band, bulk)`` fluid masks, derived test-side from the analytic body.

    With no access to the implementation's classification this is an
    *independent* oracle, and the plan (§4, §10) prefers it to asking the code
    which cells it believes are near the wall.
    """
    X, Y, _Z = _coords(mesh, (0, 0, 0), (n, n, NZ))
    r = np.hypot(X - CENTRE[0], Y - CENTRE[1])
    dx = float(mesh.geom(0).cell_size()[0])
    fluid = r > R
    band = fluid & (r < R + BAND_CELLS * dx)
    return band, fluid & ~band


#: ``n -> (band Linf, bulk Linf, global L2)``. The three solves are shared by
#: the tests below rather than repeated per assertion.
_ERRORS: dict[int, tuple[float, float, float]] = {}


def _steady_errors(n):
    """Drive ``dT/dt = alpha laplacian(T)`` to steady state and return the norms.

    The equation is the transient form of the steady problem the case actually
    poses; forward Euler is the pseudo-time driver, not the object of study, so
    the answer must be independent of ``dt`` and of ``T_END`` — it is, to ~1e-5.
    """
    if n in _ERRORS:
        return _ERRORS[n]

    mesh = _make_mesh(n)
    T = CellField(mesh, ncomp=1, ngrow=1, name="T", ibm_bc={"cyl": FixedValue(np.log(R))})
    _seed(T, mesh)

    dx = float(mesh.geom(0).cell_size()[0])
    dt = dx * dx / (DT_SAFETY * ALPHA)
    eqn = Equation(exp.ddt(T) - exp.laplacian(ALPHA, T), schemes={"ddt": "Euler"})
    for step in range(round(T_END / dt)):
        solve(eqn, dt=dt, t=step * dt, solution={"ibm": "ghostCell", "backend": BACKEND})

    err = np.abs(_assemble(T, n) - _exact(*_coords(mesh, (0, 0, 0), (n, n, NZ))))
    band, bulk = _regions(mesh, n)
    _ERRORS[n] = (
        float(err[band].max()),
        float(err[bulk].max()),
        float(np.sqrt((err[band | bulk] ** 2).mean())),
    )
    return _ERRORS[n]


def _observed_order(errors):
    """Least-squares ``p`` in ``err ~ C dx^p`` over :data:`RESOLUTIONS`."""
    dx = 1.0 / np.array(RESOLUTIONS, dtype=float)
    slope, _intercept = np.polyfit(np.log(dx), np.log(np.array(errors, dtype=float)), 1)
    return float(slope)


def _report(label, errors, order):
    """Failure message: the three norms at every resolution, plus the fit.

    The global ``L2`` appears **here and nowhere else**. A single global ``L2``
    mixes the ``O(dx)`` band with the ``O(dx^2)`` bulk over a cell count that
    favours the bulk, and converges at a flattering rate describing neither — so
    it is reported and never asserted (verification plan §4 and the §10
    anti-patterns).
    """
    rows = "\n".join(
        f"    n={n:3d}  band Linf={b:.6e}  bulk Linf={k:.6e}  (global L2={g:.6e})"
        for n, (b, k, g) in ((n, _steady_errors(n)) for n in RESOLUTIONS)
    )
    return f"{label}: observed order {order:.3f} from {list(errors)}\n{rows}"


def test_steady_solution_error_converges_in_the_bulk(blockamr_session):
    """The bulk half of the contract, on its own cells and its own norm.

    Note what this is *not*: it is not the rung-6 statement that the bulk
    operator is exact. A steady **solution** error is global — the elliptic
    problem smears the wall error across the whole domain, so the bulk carries
    the wall's accuracy too and converges at the wall's rate, not at the
    central-difference scheme's. That is why the floor here is the same
    :data:`MIN_ORDER` as the band's and not the scheme's own order.

    Red today for reason 1 of the module docstring: ``solve()`` never applies the
    immersed condition, the body is invisible to the time loop, and the field
    relaxes to the harmonic function of the *outer* boundary alone — an ``O(1)``
    error that does not converge at all.
    """
    errors = [_steady_errors(n)[1] for n in RESOLUTIONS]
    order = _observed_order(errors)
    assert order > MIN_ORDER, _report("bulk Linf", errors, order)


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

    Red today for the same reason 1 as the bulk test: with no IBM in ``solve()``
    there is no wall, so there is no band error to converge.
    """
    errors = [_steady_errors(n)[0] for n in RESOLUTIONS]
    order = _observed_order(errors)
    assert order > MIN_ORDER, _report("band Linf", errors, order)
