# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""Rung 10 — time integration, through ``solve()``.

Everything above this rung is a single ``evaluate``: one operator, one state,
no clock. This file adds the clock and nothing else. The flux field stays
**prescribed and divergence-free**, so there is no pressure solve anywhere and
no second suspect — every number here is the wall treatment plus the
integrator, and the two are separated by construction:

* **without a body**, the semi-discrete problem is a single Fourier mode that
  is an *exact eigenvector* of the discrete laplacian, so the exact solution of
  the ODE the code is actually integrating is known in closed form. The spatial
  error therefore cancels **identically** and the measured order is the
  integrator's, at any resolution (the mesh here is 8x4x4).
* **with a body**, there is no closed form, so the order is measured by
  Richardson self-convergence against a much finer ``dt`` — the spatial
  operator is bit-identical across the ``dt`` sequence, so it cannot enter the
  fit. The question that answers is not "is the wall accurate" (that is
  ``test_ibm_solution_error.py``) but "does the wall treatment cost temporal
  order".

**The state of the world this file is written against.** ``solve()`` now reads
``solution["ibm"]`` as well as ``solution["backend"]``: it validates the name
and applies the wall on the ``solve()`` path, not in ``evaluate()`` alone, and
``RungeKutta2`` / ``RungeKutta4`` have integrators behind their registered
pydantic schemes. That gap is closed (B15), so every row here is green with no
marker. The two Euler rows — the order study and the amplification factor —
were the ones green *before* it closed, on purpose: they are the controls, and
they proved the measurement apparatus (the exact semi-discrete reference, the
fit, the mesh, the equation) sound while the RK rows were still red, so the
numbers the RK rows now produce can be believed.

**The ddt spelling.** ``solve()`` resolves its time scheme with
``lookup_scheme(equation.schemes, ["ddt", "Ddt"], ...)`` — i.e. from the
*equation*, ``Equation(..., schemes={"ddt": "RK4"})``, with the value spelled
as the ``SCHEME_REGISTRY["ddt"]`` name (``Euler`` / ``RK2`` / ``RK4``). The
verification plan §7 writes it as ``solution={"ddt": "RK4"}`` instead. Every
test below drives the scheme through the equation, which is the route that
exists, so that a red row has exactly one cause; the plan's ``solution``
spelling gets its own dedicated test at the bottom, because today that key is
accepted and ignored, which is the same silent-wrong-answer failure mode as
``solution["ibm"]``.

Tier: pre-merge (verification plan §10) — a few seconds.
"""

import numpy as np
import pytest

import blockamr
from blockamr.dsl import Equation, exp, solve
from blockamr.field import CellField, FaceField
from blockamr.ibm import Cylinder, FixedValue
from blockamr.mesh import Mesh
from blockamr.operators.div import update_face_fluxes

BACKEND = "cpp"

# ---------------------------------------------------------------------------
# The integrator study, no body: one Fourier mode under pure diffusion
# ---------------------------------------------------------------------------

# 8 cells in x is not a resolution claim — it is the whole point. The reference
# is the exact solution of the *semi-discrete* system, so the spatial error
# cancels identically and a coarse mesh only buys a kinder explicit-diffusion
# limit: the stable dt scales with dx^2 while the mode's decay rate does not,
# so a fine mesh would force mu*dt so small that the Euler error sinks toward
# roundoff and the order fit stops meaning anything.
MODE_SHAPE = (8, 4, 4)
ALPHA = 1.0  # diffusivity
K = 2.0 * np.pi  # one wavelength across the unit box: periodic on this grid

# mu*T_END ~ 1.5: the mode decays by e^-1.5 over the run, so the study has real
# dynamic range. Forward Euler is stable here — the limit is
# dt < 2 / (ALPHA * sum_d 4/dx_d^2) = 2/384 = 5.2e-3, and the coarsest dt of
# the sequence below is 2.5e-3. The bound is set by the *worst* mode on the
# mesh, not by the one being integrated, and violating it lets roundoff at the
# grid scale blow up while the mode itself still looks fine.
MODE_T_END = 0.04
MODE_NSTEPS = (16, 32, 64, 128)

# One step at the coarsest dt of the sequence, for the amplification-factor
# test. Small enough that every scheme is comfortably inside its stability
# region, large enough that the schemes differ by ~4e-3 — eleven orders of
# magnitude above the tolerance the identity is asserted to.
ONE_STEP_DT = MODE_T_END / MODE_NSTEPS[0]

# The observed order may sit this far below the design order. Euler measures
# 1.01 here, so the slack is for the RK rows, whose leading term is contaminated
# by the next one at the coarse end of the sequence.
ORDER_SLACK = 0.2

# The stability polynomial of each scheme: the exact one-step amplification of
# an eigenvector, R(z) with z = -mu*dt. For RK2 this is the polynomial of
# *every* two-stage second-order explicit RK (Heun, midpoint, ...), so the test
# does not pin which one is implemented; for RK4 it is the classical method.
STABILITY_POLYNOMIAL = {
    "Euler": lambda z: 1.0 + z,
    "RK2": lambda z: 1.0 + z + z**2 / 2,
    "RK4": lambda z: 1.0 + z + z**2 / 2 + z**3 / 6 + z**4 / 24,
}

DESIGN_ORDER = {"Euler": 1, "RK2": 2, "RK4": 4}

# ---------------------------------------------------------------------------
# The transport case, with a body (verification plan §7)
# ---------------------------------------------------------------------------

R = 0.2
CENTRE = (0.5, 0.5)
AXIS = 2
BODY_SHAPE = (32, 32, 4)  # thin in the cylinder axis; nothing here varies in z

T_BODY = 1.0  # the immersed wall datum, FixedValue on the cylinder
T_AMP = 0.5  # amplitude of the transported perturbation about it

# Deliberately *not* consistent with the wall: T varies over the surface, so a
# run that applies the wall condition cannot produce the field a run that
# ignores it produces. A constant field would be annihilated by the operators
# and by the wall alike, and would make every test in this file vacuous.
NU = 0.01  # diffusivity of the transport case
DT = 1.0e-3  # CFL 0.03 on the unit flux; diffusive limit is 0.024
WALL_STEPS = 5

# The Richardson study with a body: three dt at fixed mesh, against a reference
# 32x finer than the coarsest. 32x is chosen for RK4 — the reference must be
# negligible against the coarsest error, and at fourth order 32^4 = 1e6 is the
# margin that buys.
BODY_T_END = 0.02
BODY_NSTEPS = (10, 20, 40)
BODY_REF_NSTEPS = 320

# ---------------------------------------------------------------------------
# The drift characterization (verification plan §7, case A8)
# ---------------------------------------------------------------------------

#: The measured relative drift of the fluid-integrated scalar over
#: :data:`DRIFT_STEPS` steps of the rotating-wall case at
#: :data:`DRIFT_SHAPE`.
#:
#: **This is a measurement, not a target.** ``ghostCell`` is not conservative:
#: its wall rows overwrite band cells without a compensating flux, so the
#: scalar integral moves, and asserting 0 here would assert a wish —
#: permanently red, or permanently weakened until it tests nothing. The value
#: below was measured on 2026-07-27, in the run that first had ``solve()``
#: apply the IBM (ledger ID B39, ``plans/IBM/tasks.md`` §1); before that the
#: case was an exact discrete fixed point and the drift was identically zero
#: for the wrong reason, so there was nothing truthful to write here.
#:
#: To **re-measure**, when the discretization deliberately changes: empty the
#: constant (set it back to ``None``), run this test, read the measured value
#: out of the assertion message, and put it here with the new date and ledger
#: ID. It is characterized at exactly this ``(shape, dt, steps, scheme)``
#: — changing any of them means re-measuring, not rescaling.
CHARACTERIZED_DRIFT = -4.190878e-05  # measured 2026-07-27 (B15), cpp backend

#: How far the drift may move from the characterized value before the test
#: fails. A factor, not an absolute: the assertion is "the same number", loose
#: enough to survive a compiler or a box decomposition, tight enough that a
#: change of mechanism cannot hide in it.
DRIFT_TOLERANCE = 2.0

DRIFT_SHAPE = (32, 32, 4)
DRIFT_STEPS = 200
DRIFT_CFL = 0.1
OMEGA = 1.0 / R  # tangential speed exactly 1 on the cylinder surface
B_DRIFT = 0.5  # T = B*(r^2 - R^2): radial, and zero on the wall


# ---------------------------------------------------------------------------
# Helpers — mesh/field construction, analytic fills, extraction
# ---------------------------------------------------------------------------


def _make_mesh(shape, bodies=None, periodic=(1, 1, 1)):
    """Mesh on the unit cube with ``shape`` cells, periodic by default."""
    nx, ny, nz = shape
    box = blockamr.Box([0, 0, 0], [nx - 1, ny - 1, nz - 1])
    rb = blockamr.RealBox([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    geom = blockamr.Geometry(box, rb, 0, list(periodic))
    ba = blockamr.BoxArray(box)
    ba.max_size(max(shape))
    dm = blockamr.DistributionMapping(ba)
    mesh = Mesh(ba, dm, geom)
    mesh.bodies = {} if bodies is None else bodies
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
    """Fill ``field``'s valid cells from ``func(X, Y, Z)``, then fill_patch.

    Solid cells are seeded too: the IBM must reconstruct from its own BC and
    never lean on what it finds inside the body.
    """
    mf = field.mf[0]
    for mfi in blockamr.MFIterator(mf):
        arr = mf.copy_to_host(mfi)
        X, Y, Z = _coords(mesh, mfi.valid_box().small_end(), arr.shape[:3])
        arr[:, :, :, 0] = func(X, Y, Z)
        mf.copy_from(mfi, arr)
    field.fill_patch(0, 0.0)


def _fill_halo(field, mesh, func):
    """Seed the ghost band analytically, after ``fill_patch``.

    Only needed on a non-periodic mesh: ``fill_boundary`` fills inter-box and
    periodic halos and leaves the domain-exterior ghosts alone — so this seed
    is an exact Dirichlet condition on the box that survives the ``fill_patch``
    of every subsequent step.
    """
    mf = field.mf[0]
    ng = mf.n_grow()
    for mfi in blockamr.MFIterator(mf):
        arr = mf.copy_grown_to_host(mfi)
        lo = [c - ng for c in mfi.valid_box().small_end()]
        X, Y, Z = _coords(mesh, lo, arr.shape[:3])
        arr[:, :, :, 0] = func(X, Y, Z)
        mf.copy_grown_from(mfi, arr)


def _assemble(field, shape):
    """Stitch the field's valid cells into one global ``shape`` array."""
    out = np.full(shape, np.nan)
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


def _solid_mask(mesh, shape):
    """Cells whose centre lies inside the cylinder.

    Computed test-side from the analytic body — with no access to the
    implementation's own classification that is an independent oracle rather
    than duplication (verification plan §10). No cell centre of these meshes
    lands on the surface (``r^2 = R^2`` has no solution in half-integer
    multiples of ``dx``), so the mask has no tie to break.
    """
    X, Y, _Z = _coords(mesh, (0, 0, 0), shape)
    return np.hypot(X - CENTRE[0], Y - CENTRE[1]) < R


def _observed_order(dts, errors):
    """Least-squares ``p`` in ``err ~ C dt^p``."""
    slope, _intercept = np.polyfit(np.log(np.asarray(dts)), np.log(np.asarray(errors)), 1)
    return float(slope)


def _order_report(label, dts, errors, order):
    rows = "\n".join(f"    dt={d:.3e}  err={e:.6e}" for d, e in zip(dts, errors))
    return f"{label}: observed temporal order {order:.3f}\n{rows}"


# -- the no-body mode case --------------------------------------------------


def _mode(X, Y, Z):
    """``sin(K x)`` — an exact eigenvector of the discrete central laplacian.

    ``(sin(K(x+dx)) - 2 sin(Kx) + sin(K(x-dx)))/dx^2 = -mu/ALPHA * sin(Kx)``
    exactly, and the y/z second differences of a y/z-invariant field are
    exactly zero whatever ``dy``, ``dz`` are. So the semi-discrete system is
    the scalar ODE ``dT/dt = -mu T``, whose solution is known in closed form.
    """
    return np.sin(K * X)


def _semi_discrete_rate(mesh):
    """``mu`` in ``dT/dt = -mu T``: the discrete laplacian's eigenvalue, not
    ``ALPHA*K^2``. Using the continuum rate would fold the (perfectly real,
    perfectly second-order) spatial error into the temporal fit and cap every
    observed order at 2."""
    dx = float(mesh.geom(0).cell_size()[0])
    return ALPHA * 4.0 * np.sin(0.5 * K * dx) ** 2 / dx**2


def _run_mode(ddt, dt, nsteps, route="schemes"):
    """Integrate ``dT/dt = ALPHA laplacian(T)`` from the single mode.

    ``route`` picks how the time scheme is requested: ``"schemes"`` is
    ``Equation(schemes={"ddt": ...})``, the route ``solve()`` implements;
    ``"solution"`` is the verification plan §7 spelling ``solution["ddt"]``.
    """
    mesh = _make_mesh(MODE_SHAPE)
    T = CellField(mesh, ncomp=1, ngrow=1, name="T", ibm_bc={})
    _fill(T, mesh, _mode)

    if route == "schemes":
        eqn = Equation(exp.ddt(T) - exp.laplacian(ALPHA, T), schemes={"ddt": ddt})
        solution = {"backend": BACKEND}
    else:
        eqn = Equation(exp.ddt(T) - exp.laplacian(ALPHA, T))
        solution = {"backend": BACKEND, "ddt": ddt}

    for step in range(nsteps):
        solve(eqn, dt=dt, t=step * dt, solution=solution)
    return mesh, _assemble(T, MODE_SHAPE)


def _mode_exact(mesh, t):
    """The exact solution of the semi-discrete system at time ``t``."""
    X, Y, Z = _coords(mesh, (0, 0, 0), MODE_SHAPE)
    return _mode(X, Y, Z) * np.exp(-_semi_discrete_rate(mesh) * t)


# -- the with-body transport case -------------------------------------------


def _uniform_velocity(x, y, z, t):
    return np.ones_like(x), np.ones_like(x), np.ones_like(x)


def _transport_case(ddt="Euler"):
    """The verification plan §7 equation: ``ddt(T) + div(phi, T) - laplacian(NU, T)``.

    ``phi`` is the prescribed, exactly divergence-free unit flux; the body is a
    cylinder carrying ``FixedValue(T_BODY)``; the mesh is periodic in all three
    directions and so is the initial field, so the halo is exact at every step
    and no domain edge has to be eroded out of an assertion.
    """
    bodies = {"cyl": Cylinder(centre=CENTRE, radius=R, axis=AXIS)}
    mesh = _make_mesh(BODY_SHAPE, bodies=bodies)
    T = CellField(mesh, ncomp=1, ngrow=1, name="T", ibm_bc={"cyl": FixedValue(T_BODY)})
    _fill(T, mesh, lambda X, Y, Z: T_BODY + T_AMP * np.sin(K * X) * np.sin(K * Y))

    phi = FaceField(mesh, ncomp=1, ngrow=T.ngrow, name="phi")
    update_face_fluxes(phi[0], _uniform_velocity, mesh.geom(0), t=0.0)

    eqn = Equation(
        exp.ddt(T) + exp.div(phi, T) - exp.laplacian(NU, T),
        schemes={"ddt": ddt, "Div": "linear"},
    )
    return mesh, T, eqn


def _sol(ibm=None):
    """The fvSolution block: no ``"ibm"`` key at all means no IBM."""
    return {"backend": BACKEND} if ibm is None else {"ibm": ibm, "backend": BACKEND}


def _run_transport(ddt, dt, nsteps, ibm):
    mesh, T, eqn = _transport_case(ddt)
    for step in range(nsteps):
        solve(eqn, dt=dt, t=step * dt, solution=_sol(ibm))
    return mesh, _assemble(T, BODY_SHAPE)


# -- the drift case (A8: rotating wall, radial scalar) -----------------------


def _rotation_velocity(x, y, z, t):
    """``u = omega x r`` about the cylinder axis — solid-body rotation.

    Divergence-free analytically *and* discretely: ``u_x`` does not depend on x
    and ``u_y`` does not depend on y, so both face differences vanish
    identically, whatever the mesh.
    """
    return -OMEGA * (y - CENTRE[1]), OMEGA * (x - CENTRE[0]), np.zeros_like(x)


def _radial(X, Y, Z):
    """``T = B(r^2 - R^2)``: radial, so ``u . grad T == 0``, and zero on the
    wall, so it is consistent with ``FixedValue(0)`` and carries no constant
    pedestal to dilute a relative drift."""
    return B_DRIFT * ((X - CENTRE[0]) ** 2 + (Y - CENTRE[1]) ** 2 - R**2)


def _total_scalar_drift(nsteps, ibm):
    """Relative change of the fluid-integrated scalar over ``nsteps``.

    The exact answer is **zero drift, exactly**: the flux is tangential to the
    cylinder and to every circle, ``T`` is radial, so ``div(u T) = u . grad T``
    vanishes pointwise — and it vanishes discretely too on the ``linear`` flux
    interpolation (the x- and y-differences cancel to the last bit; the rung-8
    test of ``test_ibm_rungs.py`` asserts exactly this). The field is therefore
    a discrete fixed point of the bulk operator, and **every** part of the
    measured drift is the wall treatment. That is what makes the number worth
    characterizing rather than merely bounding.

    Forward Euler because it is the only integrator that exists; the sum is
    over fluid cells only, because the scalar the method fails to conserve is
    the one outside the body.
    """
    bodies = {"cyl": Cylinder(centre=CENTRE, radius=R, axis=AXIS)}
    mesh = _make_mesh(DRIFT_SHAPE, bodies=bodies, periodic=(0, 0, 0))
    T = CellField(mesh, ncomp=1, ngrow=1, name="T", ibm_bc={"cyl": FixedValue(0.0)})
    _fill(T, mesh, _radial)
    _fill_halo(T, mesh, _radial)

    phi = FaceField(mesh, ncomp=1, ngrow=T.ngrow, name="phi")
    update_face_fluxes(phi[0], _rotation_velocity, mesh.geom(0), t=0.0)
    eqn = Equation(exp.ddt(T) + exp.div(phi, T), schemes={"ddt": "Euler", "Div": "linear"})

    # |u| is largest at the far corner of the box, and that is what sets the
    # advective CFL — not the O(1) speed at the surface.
    dx = float(mesh.geom(0).cell_size()[0])
    u_max = OMEGA * float(np.hypot(0.5, 0.5))
    dt = DRIFT_CFL * dx / u_max

    fluid = ~_solid_mask(mesh, DRIFT_SHAPE)
    before = float(_assemble(T, DRIFT_SHAPE)[fluid].sum())
    for step in range(nsteps):
        solve(eqn, dt=dt, t=step * dt, solution=_sol(ibm))
    after = float(_assemble(T, DRIFT_SHAPE)[fluid].sum())
    return (after - before) / abs(before)


# ---------------------------------------------------------------------------
# The headline rung-10 test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ddt", ["Euler", "RK2", "RK4"])
def test_wall_is_held_after_every_rk_stage(blockamr_session, ddt):
    """A per-step application passes a per-step check and still leaks the wall
    value inside RK's stages — so the probe is a multi-stage scheme, and the
    assert is on the state after each completed step of an RK run.

    Transport of ``T`` on a prescribed, divergence-free flux field: the time
    behaviour of the wall condition, with no projection in the loop.

    ``Euler`` is in the parametrization as the control, not as a second copy of
    the same demand: it is the one row a per-step application satisfies
    trivially, so if the RK rows fail while it passes, the defect is the
    *schedule* (once per step instead of once per stage) and not the wall rows
    themselves. If Euler fails too, the wall is not being applied at all and
    the RK rows say nothing.

    Red on all three rows today, and for two different reasons — ``solve()``
    never enters the IBM path for any of them, and RK2/RK4 additionally raise
    ``NotImplementedError`` before they get near a wall. ``directForcing``
    carries a third (task T6: it still pins a 3-component velocity through a
    jnp mask rather than applying its ``ibm_bc`` datum as wall rows), which is
    why the plan names it here: it is the method whose whole definition is
    "hold the solid value between stages".
    """
    mesh, T, eqn = _transport_case(ddt)
    solid = _solid_mask(mesh, BODY_SHAPE)
    solution = _sol("directForcing")

    for step in range(WALL_STEPS):
        solve(eqn, dt=DT, t=step * DT, solution=solution)
        held = _assemble(T, BODY_SHAPE)[solid]
        np.testing.assert_allclose(
            held, T_BODY, atol=1e-13, err_msg=f"{ddt}: wall not held after step {step}"
        )


#: The stage times of one step started at ``t``, per ddt scheme — the schedule
#: ``solve()`` is built from (``_rk4_step``'s ``stages`` table, ``_rk2_step``'s
#: pairs, ``ForwardEuler``'s single call), written out independently here so the
#: assertion is against the method's definition and not against the code.
STAGE_TIMES = {
    "Euler": (0.0,),
    "RK2": (0.0, 0.5),
    "RK4": (0.0, 0.5, 0.5, 1.0),
}


@pytest.mark.parametrize("ddt", ["Euler", "RK2", "RK4"])
def test_a_time_dependent_wall_datum_is_re_evaluated_at_every_stage(blockamr_session, ddt):
    """B42. The wall datum is a *schedule*, and the schedule is per **stage**.

    A datum refreshed once per step instead of once per stage is exactly the
    defect A4 exists to catch — it collapses the RK rows to first order — and
    it is invisible with a constant datum, because then the two refreshes
    produce identical numbers. Here the datum records the times it is asked
    for, so the schedule is asserted directly and in one step, with no order
    study and no fit.

    The equation carries exactly **one** spatial term on purpose: the band rows
    are rebuilt per term per apply, so with ``div + laplacian`` the recorded
    times would be each stage time twice and the row would be asserting the
    term count as much as the schedule.

    Also asserts the datum is evaluated at the **wall foot points**: every
    recorded point lies on the cylinder to roundoff, which an evaluation at the
    band cells' centres would miss by up to half a cell.
    """
    calls = []

    def datum(x, y, z, t):
        calls.append((float(t), np.asarray(x).copy(), np.asarray(y).copy()))
        return np.full(np.shape(x), T_BODY)

    bodies = {"cyl": Cylinder(centre=CENTRE, radius=R, axis=AXIS)}
    mesh = _make_mesh(BODY_SHAPE, bodies=bodies)
    T = CellField(mesh, ncomp=1, ngrow=1, name="T", ibm_bc={"cyl": FixedValue(datum)})
    _fill(T, mesh, lambda X, Y, Z: T_BODY + T_AMP * np.sin(K * X) * np.sin(K * Y))
    eqn = Equation(exp.ddt(T) - exp.laplacian(NU, T), schemes={"ddt": ddt})

    t0 = 0.4  # not zero: a datum asked for at the *step* time only would still
    # produce 0.4 for the first stage, so a nonzero start is what makes the
    # later stage times distinguishable from a repeated t0.
    solve(eqn, dt=DT, t=t0, solution=_sol("ghostCell"))

    expected = [t0 + c * DT for c in STAGE_TIMES[ddt]]
    assert [c[0] for c in calls] == pytest.approx(expected), (
        f"{ddt}: the wall datum was asked for at {[c[0] for c in calls]}, "
        f"not at the stage times {expected}"
    )
    for _t, x, y in calls:
        assert x.size > 0, f"{ddt}: the datum was called with no wall points"
        radius = np.hypot(x - CENTRE[0], y - CENTRE[1])
        np.testing.assert_allclose(
            radius, R, atol=1e-12, err_msg=f"{ddt}: the datum was not evaluated on the surface"
        )


# ---------------------------------------------------------------------------
# Temporal order — without a body, then with one
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ddt", ["Euler", "RK2", "RK4"])
def test_temporal_order_without_a_body(blockamr_session, ddt):
    """The integrator alone, against an exact reference.

    The spatial error is not "negligible" here, it is **cancelled**: the
    initial field is an exact eigenvector of the discrete laplacian, so the
    system being integrated is the scalar ODE ``dT/dt = -mu T`` with ``mu``
    the *discrete* eigenvalue, and its solution is ``exp(-mu t)`` exactly. Any
    departure from it is the integrator and only the integrator, at 8 cells or
    at 8000.

    This is the control for the whole file. The Euler row is green today and
    must stay green: it is what makes the RK rows' eventual numbers credible,
    and it is what distinguishes "RK4 is not implemented" from "the harness
    cannot measure an order".
    """
    dts = [MODE_T_END / n for n in MODE_NSTEPS]
    errors = []
    for dt, nsteps in zip(dts, MODE_NSTEPS):
        mesh, num = _run_mode(ddt, dt, nsteps)
        errors.append(float(np.abs(num - _mode_exact(mesh, MODE_T_END)).max()))

    order = _observed_order(dts, errors)
    expected = DESIGN_ORDER[ddt]
    assert order > expected - ORDER_SLACK, _order_report(ddt, dts, errors, order)


@pytest.mark.parametrize("ddt", ["Euler", "RK2", "RK4"])
def test_temporal_order_with_a_body_is_not_degraded(blockamr_session, ddt):
    """The same order, measured with an immersed wall in the loop.

    There is no closed form with a body, so the reference is a run at
    ``dt/32`` (Richardson self-convergence, verification plan §9.2). That is
    the right tool precisely because it cannot see the spatial error: the
    spatial operator is bit-identical across the ``dt`` sequence, so it enters
    every run as the same constant and cancels out of the fit. What is left is
    the temporal order of the integrator *coupled to the wall treatment* — and
    a reconstruction refreshed once per step instead of once per stage shows up
    here as a collapse to first order.

    **The guard is not decoration.** A self-convergence study measures the
    integrator's order whether or not the wall treatment ever ran — with
    ``solution["ibm"]`` silently ignored, the Euler row would fit a clean 1.0
    and report success for a feature that is entirely absent. Asserting first
    that the key changes the answer is what makes the order claim a claim about
    the *wall*.
    """
    with_ibm = _run_transport("Euler", DT, 1, "ghostCell")[1]
    without = _run_transport("Euler", DT, 1, None)[1]
    assert not np.array_equal(with_ibm, without), (
        "solve() produced a bitwise identical step with and without "
        "solution['ibm']: the wall treatment never ran, so any order measured "
        "below would be the integrator's alone and would not be about the IBM"
    )

    dts = [BODY_T_END / n for n in BODY_NSTEPS]
    ref_dt = BODY_T_END / BODY_REF_NSTEPS
    mesh, reference = _run_transport(ddt, ref_dt, BODY_REF_NSTEPS, "ghostCell")
    fluid = ~_solid_mask(mesh, BODY_SHAPE)

    errors = []
    for dt, nsteps in zip(dts, BODY_NSTEPS):
        _m, num = _run_transport(ddt, dt, nsteps, "ghostCell")
        errors.append(float(np.abs(num - reference)[fluid].max()))

    order = _observed_order(dts, errors)
    expected = DESIGN_ORDER[ddt]
    assert order > expected - ORDER_SLACK, _order_report(f"{ddt} + ghostCell", dts, errors, order)


# ---------------------------------------------------------------------------
# The one measured quantity in the file
# ---------------------------------------------------------------------------


def test_scalar_drift_matches_the_characterized_value(blockamr_session):
    """``ghostCell`` is not conservative. Assert the known drift, not zero: a
    test that demands 0 here would be testing a wish, and would be permanently
    red or permanently weakened.

    So the contract is a *characterization*: :data:`CHARACTERIZED_DRIFT` is a
    measurement of this exact configuration, and the test's job is to notice
    when the number moves. It cannot be measured before task T15, because
    until ``solve()`` applies the IBM the case is an exact discrete fixed point
    and the drift is identically zero — a zero that means "nothing happened",
    not "nothing was lost". Writing that zero into the constant now would bake
    the bug in as the contract, so the constant is ``None`` and the assertion
    that reads it prints the value to fill it with.
    """
    drift = _total_scalar_drift(nsteps=DRIFT_STEPS, ibm="ghostCell")

    assert CHARACTERIZED_DRIFT is not None, (
        f"unfilled characterization: this run measured drift={drift:.6e} over "
        f"{DRIFT_STEPS} steps at {DRIFT_SHAPE}. If solve() now applies the IBM, "
        f"set CHARACTERIZED_DRIFT = {drift:.6e} and drop the T15 marker; if it "
        "does not, this number is a fixed point's zero and means nothing."
    )
    assert abs(drift) < DRIFT_TOLERANCE * abs(CHARACTERIZED_DRIFT), (
        f"scalar drift {drift:.6e} moved away from the characterized "
        f"{CHARACTERIZED_DRIFT:.6e} by more than {DRIFT_TOLERANCE}x — the wall "
        "treatment's conservation error changed mechanism, not magnitude"
    )


# ---------------------------------------------------------------------------
# solve() must not silently ignore what it is handed
# ---------------------------------------------------------------------------


def test_ibm_key_is_not_silently_ignored_by_solve(blockamr_session):
    """The worst failure mode in this design, stated as a test.

    ``solve(eqn, solution={"ibm": "ghostCell"})`` runs happily today and does
    no IBM at all: the key is read for ``"backend"`` and dropped. Nothing
    raises, nothing warns, and the run produces a plausible field with no wall
    in it — the same class of failure the row-format document calls out for a
    stale table ("plausible wrong numbers, so this check is not optional"). A
    user cannot tell that from a correct run by looking at it.

    The probe is bitwise: one step with the key and one without, on a field
    that is deliberately *not* consistent with its wall value, so a wall that
    was applied cannot leave the same numbers behind. Not ``allclose`` — the
    question is whether the code path ran at all, and a tolerance cannot answer
    that.
    """
    _m1, with_ibm = _run_transport("Euler", DT, 1, "ghostCell")
    _m2, without = _run_transport("Euler", DT, 1, None)

    assert not np.array_equal(with_ibm, without), (
        "one step of solve() with solution={'ibm': 'ghostCell'} was bitwise "
        "identical to one step without it: the immersed wall was silently "
        "ignored and the run is a plausible wrong answer"
    )


def test_unknown_ibm_name_passed_to_solve_raises(blockamr_session):
    """The cheapest possible proof that ``solve()`` reads the key at all.

    ``evaluate()`` already rejects an unknown method by name and lists the
    valid ones (``test_ibm_rungs.py``); ``solve()`` accepts anything, because
    it never looks. The two entry points take the same ``solution`` block and
    must validate it the same way — a name that is a typo in ``evaluate`` does
    not become a silent no-op by being passed to ``solve`` instead.
    """
    _mesh, _T, eqn = _transport_case("Euler")
    with pytest.raises(ValueError) as excinfo:
        solve(eqn, dt=DT, t=0.0, solution=_sol("noSuchMethod"))

    msg = str(excinfo.value)
    assert "noSuchMethod" in msg
    assert "ghostCell" in msg, f"the valid methods are missing from: {msg}"


# ---------------------------------------------------------------------------
# The ddt scheme must be reachable, and by the spelling the plan uses
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ddt", ["Euler", "RK2", "RK4"])
def test_ddt_scheme_from_the_equation_is_honoured_by_solve(blockamr_session, ddt):
    """Rung 10 needs a multi-stage scheme to exist. ``RK2``/``RK4`` are in
    ``SCHEME_REGISTRY["ddt"]`` and are valid pydantic models, and ``solve()``
    raises ``NotImplementedError`` for both — so "reachable" is the demand this
    test makes, and it makes it by asserting the *exact* answer rather than
    merely that nothing raised.

    One step of a single eigenvector is exactly ``T0 * R(z)``, with ``R`` the
    scheme's stability polynomial and ``z = -mu*dt``: an identity, to machine
    precision, that names which scheme ran. The three polynomials differ by
    ``~4e-3`` at this ``dt``, so this also fails loudly for a scheme that is
    accepted and quietly downgraded to Euler.

    ``Euler`` is green today and is the control: it proves the identity, the
    eigenvector and the tolerance are right, so a red RK row is about RK.
    """
    mesh, got = _run_mode(ddt, ONE_STEP_DT, 1)
    z = -_semi_discrete_rate(mesh) * ONE_STEP_DT
    X, Y, Z = _coords(mesh, (0, 0, 0), MODE_SHAPE)
    expected = _mode(X, Y, Z) * STABILITY_POLYNOMIAL[ddt](z)

    np.testing.assert_allclose(
        got, expected, atol=1e-14, err_msg=f"one step of {ddt} is not R(z)={ddt}'s polynomial"
    )


@pytest.mark.parametrize("ddt", ["RK2", "RK4"])
def test_ddt_scheme_from_the_solution_dict_is_honoured_by_solve(blockamr_session, ddt):
    """The verification plan §7 spelling: ``solution={"ddt": "RK4"}``.

    ``solve()`` resolves its time scheme from ``equation.schemes`` — a ``ddt``
    key in ``solution`` is accepted and ignored, exactly like ``"ibm"``, and
    the run silently proceeds in Forward Euler. That is a second silent wrong
    answer hiding behind the same entry point, and the plan's own rung-10
    snippet is written in the spelling that triggers it, so either ``solve()``
    honours the key or the plan and this test move to ``Equation(schemes=...)``
    together. Until then it is red.

    ``Euler`` is deliberately **not** in the parametrization: the default is
    already Forward Euler, so an ignored ``solution["ddt"] = "Euler"`` produces
    the right answer for the wrong reason and would be a green row proving
    nothing.
    """
    mesh, got = _run_mode(ddt, ONE_STEP_DT, 1, route="solution")
    z = -_semi_discrete_rate(mesh) * ONE_STEP_DT
    X, Y, Z = _coords(mesh, (0, 0, 0), MODE_SHAPE)

    euler = _mode(X, Y, Z) * STABILITY_POLYNOMIAL["Euler"](z)
    assert not np.allclose(got, euler, atol=1e-14), (
        f"solve() with solution={{'ddt': {ddt!r}}} took a Forward Euler step: "
        "the key was accepted and silently ignored"
    )
    np.testing.assert_allclose(
        got,
        _mode(X, Y, Z) * STABILITY_POLYNOMIAL[ddt](z),
        atol=1e-14,
        err_msg=f"solution['ddt']={ddt!r} did not produce {ddt}'s stability polynomial",
    )
