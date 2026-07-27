# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""Rung 11, the **unsteady** analytic set: A4, A5, A6, A8 (verification plan §9).

Rungs 1-10 are all steady statements — an identity that one ``evaluate`` must
satisfy, or a solution error at a fixed point. The plan's §9 opens with the
reason that is not enough:

    a wall condition can be exact in steady state and still lag, damp, or leak
    in time, and none of rungs 1-10 would see it.

So every case here has a **closed-form, time-dependent** reference, and every
one is **projection-free**: A4/A5/A6 are unidirectional flows whose exact
solution is divergence-free by construction, A8 transports a scalar on a
prescribed rigid rotation. Nothing in this file solves for a pressure; if a
pressure solve ever becomes necessary here, the case has been mis-set.

    A4  Stokes' 2nd problem   inclined plane   u = U0 e^-z cos(wt - z), z = y/delta
    A5  Stokes' 1st problem   inclined plane   u = U0 erfc(y / 2 sqrt(nu t))
    A6  Womersley channel     two planes       closed form in alpha = h sqrt(w/nu)
    A8  Rotating-wall scalar  cylinder         dT/dt == 0 on a tangential field

**A4 is the case this file is built around.** Its discriminating metric is a
*phase*, not a magnitude, which is exactly the signature of a wall
reconstruction refreshed once per step instead of once per stage: a method can
reproduce the amplitude to 1% and still be a whole timestep late. The phase is
asserted at three depths (0.5, 1, 2 delta) precisely so a uniform time shift (a
schedule bug) is separable from a depth-dependent error (a diffusion/spatial
bug) — the failure message reports that decomposition.

**Every wall is inclined.** ``N_HAT = (1, 2, 3)/|(1, 2, 3)|`` — the inclination
``test_ibm_rungs`` already established as non-degenerate: no trilinear fraction
is 0, so every reconstruction row is a genuine 3-D interpolation. An
axis-aligned wall collapses the 8-donor stencil to a 2-point 1-D one and hides
interpolation error entirely.

**Resolution.** Every unidirectional case resolves its diffusive length scale
with ``SCALE_CELLS = 5`` cells, and that number is not free: trilinear
reconstruction is linear-exact, so the ghost value carries an ``O(dx^2 u'')``
error, and against a profile whose curvature scale is ``delta`` that is a
relative error of ``(dx/delta)^2 = 1/25 = 4%``. The plan's own A4 tolerance is
``rel=0.05``. Coarser than 5 cells and the tolerance is unreachable for reasons
that are not the wall treatment's fault; much finer and the runs get expensive
for no new information (cost per period scales as ``(delta/dx)^2``).

**Boundaries.** The immersed wall is the object of study, so the *outer* box
boundary is removed as a variable: the domain-exterior ghost band is written
with the analytic solution at the current time before every step. That is an
exact Dirichlet condition, so no domain-size study is needed and the assertions
can cover the whole fluid instead of an eroded interior. Solid cells are seeded
with ``0.0`` — a value inconsistent with every wall datum in this file — so a
method that leans on what it finds inside the body fails visibly.

**Tier: nightly** (verification plan §10) — the driven cases are marked
``slow``, deselect with ``-m 'not slow'``.

**Live since B15.** ``solve()`` honours ``solution["ibm"]`` as well as
``solution["backend"]``, and ``RungeKutta2``/``RungeKutta4`` are implemented, so
every case in this file runs against a real immersed wall — what each one then
measures is a physics result, not a construction-time gap. The two *oracles*
every conclusion here
rests on — the harmonic fit, and the closed forms themselves — are unit-tested
against synthetic data and against their own PDEs, and those tests are green: a
validation suite whose measuring instrument is unverified is worth nothing.
"""

import math

import numpy as np
import pytest

import blockamr
from blockamr.dsl import Equation, evaluate, exp, solve
from blockamr.field import CellField, FaceField
from blockamr.ibm import Cylinder, FixedValue, Plane
from blockamr.mesh import Mesh
from blockamr.operators.div import update_face_fluxes

from .ibm_gaps import (
    B27_UNSTEADY_VALIDATION_MEASUREMENT,
    T17_MOVING_BODIES,
    T18_FORCES,
)

BACKEND = "cpp"

# The inclination shared by A4/A5/A6, normalised at import.
N_HAT = np.array([1.0, 2.0, 3.0]) / np.linalg.norm([1.0, 2.0, 3.0])

#: Cells per diffusive length scale — see the module docstring.
SCALE_CELLS = 5

#: Forward Euler on 3-D diffusion is stable for dt <= dx^2/(6 nu) when the
#: cells are cubic. The factor 2 of margin leaves room for the row
#: amplification the wall reconstruction adds on top of it.
DIFFUSION_DT_DIVISOR = 12.0

#: The fluid shell the wall treatment owns: the laplacian's reach plus the
#: layer it reconstructs. Band and bulk are always reported separately (§4,
#: §10) — a combined norm is the bulk's norm wearing the band's name.
BAND_CELLS = 2.0

U0 = 1.0  # wall speed / velocity scale of every unidirectional case
NU = 0.01  # kinematic viscosity of every unidirectional case

_erfc = np.vectorize(math.erfc)


# ---------------------------------------------------------------------------
# Oracle 1: harmonic fitting and power-law fitting
# ---------------------------------------------------------------------------


def _fit_harmonic(t, y, omega):
    """Least-squares ``(amplitude, phase)`` of ``y ~ c + A cos(omega t + phi)``.

    This is *the* measuring instrument of A4 and A6 — every conclusion those
    cases draw is a number this function returned — so it is fitted linearly
    (no optimiser, no initial guess, no convergence to argue about) and it is
    unit-tested against synthetic data below.

    The constant column is not decoration: a wall treatment that leaks
    introduces a slow mean drift, and without a DC column that drift would be
    projected onto the cosine and read as an amplitude error. With it, the fit
    reports the amplitude and phase of the *oscillation* alone.

    ``omega`` is the drive frequency — known exactly, never fitted, because the
    quantity under test is the response's phase *relative to a known drive*.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    basis = np.stack([np.cos(omega * t), np.sin(omega * t), np.ones_like(t)], axis=1)
    coeffs, _res, _rank, _sv = np.linalg.lstsq(basis, y, rcond=None)
    a, b = float(coeffs[0]), float(coeffs[1])
    # A cos(wt + phi) = (A cos phi) cos(wt) - (A sin phi) sin(wt)
    return float(np.hypot(a, b)), float(np.arctan2(-b, a))


def _fit_power(x, y):
    """Least-squares exponent ``p`` in ``y ~ C x^p`` — the order-fitting helper.

    Used by A5, where the *shape* of the wall-shear decay (``t^-1/2``) is the
    assertion: an exponent is dimensionless and carries no arbitrary tolerance,
    unlike the magnitude it is fitted from.
    """
    slope, _intercept = np.polyfit(np.log(np.asarray(x, dtype=float)), np.log(np.abs(y)), 1)
    return float(slope)


def _wrap(angle):
    """Wrap a phase difference into ``(-pi, pi]``."""
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


@pytest.mark.parametrize(
    "amp, phase, offset",
    [
        (1.0, 0.0, 0.0),
        (0.37, -1.25, 0.0),
        (2.5, 2.9, 0.0),
        (0.37, -1.25, 4.2),  # a large DC offset must not bleed into A or phi
        (0.05, -0.4, -0.9),  # a small signal riding on a larger offset
    ],
)
def test_fit_harmonic_recovers_a_known_amplitude_and_phase(amp, phase, offset):
    """The oracle test. If this fails, A4 and A6 measure nothing at all.

    The samples are laid out as a run lays them out — a non-integer number of
    periods (2.5), so no orthogonality accident helps the fit — and the
    recovery must be to machine precision, because the data is exact.
    """
    omega = 3.7
    t = np.linspace(0.0, 2.5 * 2.0 * np.pi / omega, 211)
    y = offset + amp * np.cos(omega * t + phase)

    fit_amp, fit_phase = _fit_harmonic(t, y, omega)

    assert fit_amp == pytest.approx(amp, rel=1e-10)
    assert _wrap(fit_phase - phase) == pytest.approx(0.0, abs=1e-10)


def test_fit_harmonic_reports_the_phase_of_a_pure_time_shift():
    """The specific reading A4 depends on: a signal delayed by ``tau`` must fit
    a phase of ``-omega*tau``.

    A4's whole claim is that a stage-stale reconstruction shows up as a phase
    error; that claim only holds if the fit turns a time shift into exactly
    that phase, with the sign the analytic reference uses (``cos(wt - y/delta)``
    lags, so a lag is a *negative* phase).
    """
    omega, tau = 2.0, 0.31
    t = np.linspace(0.0, 6.0 * np.pi / omega, 401)
    _amp, phase = _fit_harmonic(t, np.cos(omega * (t - tau)), omega)
    assert _wrap(phase + omega * tau) == pytest.approx(0.0, abs=1e-10)


def test_fit_power_recovers_a_known_exponent():
    """The A5 oracle: the ``t^-1/2`` wall-shear decay is read by this fit."""
    x = np.geomspace(0.5, 40.0, 25)
    assert _fit_power(x, 3.1 * x**-0.5) == pytest.approx(-0.5, abs=1e-12)


# ---------------------------------------------------------------------------
# Mesh / field helpers — the plane cases all share one geometry
# ---------------------------------------------------------------------------


def _make_mesh(n, nz=None, bodies=None):
    """One box, ``n x n x nz`` cells on the unit cube, non-periodic.

    **One box** on purpose: with no interior box boundaries, "outside the valid
    box" and "outside the domain" are the same set, which is what makes the
    per-step analytic halo seeding below a boundary condition rather than a
    correction applied to inter-box halos too. (Box-decomposition invariance is
    already a rung-5 test; it is not this file's question.)

    Non-periodic in every direction because none of these solutions is periodic
    in any of them — an inclined wall makes every axis a mix of wall-normal and
    wall-tangential.
    """
    nz = n if nz is None else nz
    box = blockamr.Box([0, 0, 0], [n - 1, n - 1, nz - 1])
    rb = blockamr.RealBox([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    geom = blockamr.Geometry(box, rb, 0, [0, 0, 0])
    ba = blockamr.BoxArray(box)
    ba.max_size(max(n, nz))
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


def _normal_coord(X, Y, Z, point):
    """Signed wall-normal coordinate ``(x - point) . N_HAT``.

    Every A4/A5/A6 reference is a function of this one coordinate and time —
    which is what makes an inclined wall cost nothing in the analysis while
    costing the reconstruction everything.
    """
    return (X - point[0]) * N_HAT[0] + (Y - point[1]) * N_HAT[1] + (Z - point[2]) * N_HAT[2]


def _fill(field, mesh, func, t):
    """Fill the valid cells from ``func(X, Y, Z, t)`` — the initial condition."""
    mf = field.mf[0]
    for mfi in blockamr.MFIterator(mf):
        arr = mf.copy_to_host(mfi)
        X, Y, Z = _coords(mesh, mfi.valid_box().small_end(), arr.shape[:3])
        arr[:, :, :, 0] = func(X, Y, Z, t)
        mf.copy_from(mfi, arr)
    field.fill_patch(0, t)


def _ghost_seeder(field, mesh, func):
    """Return ``seed(t)``: write ``func(X, Y, Z, t)`` into the exterior ghosts.

    Only the ghosts — the valid cells are the solution and must survive. The
    coordinate meshgrid and the ghost mask are built once; per step this is one
    ``where`` over the grown array.

    This is the outer boundary condition of A4/A5/A6 and it is *exact*, so the
    only wall in the problem is the immersed one. It is refreshed at the start
    of each step: within an RK stage the outer halo is stale by at most one dt,
    which is a property of the driven boundary rather than of the immersed
    wall, and every probe sits several cells inside it.
    """
    mf = field.mf[0]
    ng = mf.n_grow()
    grids = []
    for mfi in blockamr.MFIterator(mf):
        arr = mf.copy_grown_to_host(mfi)
        lo = [c - ng for c in mfi.valid_box().small_end()]
        X, Y, Z = _coords(mesh, lo, arr.shape[:3])
        ghost = np.ones(arr.shape[:3], dtype=bool)
        ghost[ng:-ng, ng:-ng, ng:-ng] = False
        grids.append((X, Y, Z, ghost))

    def seed(t):
        for bi, mfi in enumerate(blockamr.MFIterator(mf)):
            X, Y, Z, ghost = grids[bi]
            arr = mf.copy_grown_to_host(mfi)
            arr[:, :, :, 0] = np.where(ghost, func(X, Y, Z, t), arr[:, :, :, 0])
            mf.copy_grown_from(mfi, arr)

    return seed


def _assemble_field(field, shape):
    """Stitch a field's valid cells into one global array."""
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


def _assemble_result(field, results, shape):
    """Stitch a per-level, per-box ``evaluate`` result into one global array."""
    out = np.full(shape, np.nan)
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


def _interior_mask(shape, margin=2):
    """Cells at least ``margin`` cells away from the domain boundary."""
    mask = np.zeros(shape, dtype=bool)
    mask[margin:-margin, margin:-margin, margin:-margin] = True
    return mask


def _probe_cell(coord, target, interior):
    """Index of the cell whose ``coord`` is closest to ``target``.

    A *single* cell, and the assertion is then made against that cell's own
    exact coordinate rather than against the nominal target: on an inclined
    wall no cell centre sits exactly at 0.5/1/2 delta, and the offset is up to
    dx/2 = 0.1 delta — larger than the phase tolerance being asserted.
    Averaging a shell instead would smear a nonlinear profile; comparing
    against the nominal depth would import a 10% error that has nothing to do
    with the wall.

    ``interior`` excludes cells within reach of the analytically driven outer
    halo, so no probe measures the boundary condition instead of the wall.
    Ties (which floating-point ``coord`` values make essentially impossible)
    break towards the domain centre, for determinism.
    """
    err = np.abs(np.where(interior, coord, np.inf) - target)
    best = err == err.min()
    idx = np.indices(coord.shape)
    centre = (np.array(coord.shape) - 1.0) / 2.0
    dist = sum((idx[d] - centre[d]) ** 2 for d in range(3)).astype(float)
    dist[~best] = np.inf
    return np.unravel_index(int(np.argmin(dist)), coord.shape)


def _drive(eqn, solution, dt, t_start, nsteps, seed, sample_every, probe):
    """Step the equation, re-seeding the exact outer halo before every step.

    ``probe(t)`` is whatever the case measures, sampled every ``sample_every``
    steps. Returns ``(times, samples)``.
    """
    times, samples = [], []
    for k in range(nsteps):
        t = t_start + k * dt
        seed(t)
        solve(eqn, dt=dt, t=t, solution=solution)
        if (k + 1) % sample_every == 0:
            times.append(t + dt)
            samples.append(probe(t + dt))
    return np.array(times), np.array(samples)


def _sol(method):
    return {"ibm": method, "backend": BACKEND}


# ---------------------------------------------------------------------------
# A4 — Stokes' second problem: an oscillating inclined wall
# ---------------------------------------------------------------------------

N_A4 = 32
DX_A4 = 1.0 / N_A4
DELTA_A4 = SCALE_CELLS * DX_A4  # Stokes layer thickness = sqrt(2 nu / omega)
OMEGA_A4 = 2.0 * NU / DELTA_A4**2
PERIOD_A4 = 2.0 * np.pi / OMEGA_A4

# The wall passes through (0.2, 0.2, 0.2) with normal N_HAT: on the unit cube
# that leaves a solid slab 0.32 deep (10 cells, far more than the single layer
# ghostCell reconstructs) and a fluid wedge 1.28 deep (8.2 delta, so the outer
# halo sits where the analytic amplitude is e^-8.2 ~ 3e-4).
WALL_POINT_A4 = (0.2, 0.2, 0.2)

PERIODS_A4 = 6  # verification plan §9.1
FIT_PERIODS = 2  # the fit window: the last two, after the transient has died
PROBE_DEPTHS = (0.5, 1.0, 2.0)  # in units of delta

# From the plan's own A4 sketch. The 5% amplitude tolerance is the reason
# SCALE_CELLS is 5 (module docstring); 0.08 rad is ~1.3% of a period and far
# above the ~0.007 rad a single timestep of lag costs here, so a stage-stale
# reconstruction has to be more than ten steps late before it could hide.
AMP_RTOL = 0.05
PHASE_ATOL = 0.08


def _stokes2(s, t):
    """``u = U0 exp(-s/delta) cos(omega t - s/delta)`` — the fluid solution."""
    z = np.asarray(s, dtype=float) / DELTA_A4
    return U0 * np.exp(-z) * np.cos(OMEGA_A4 * t - z)


def _stokes2_seed(X, Y, Z, t):
    """The field over the whole box: the exact solution in the fluid, and a
    deliberately inconsistent ``0`` inside the body."""
    s = _normal_coord(X, Y, Z, WALL_POINT_A4)
    return np.where(s > 0.0, _stokes2(np.maximum(s, 0.0), t), 0.0)


def _oscillating_wall(x, y, z, t):
    """The wall's own tangential speed, ``U0 cos(omega t)``.

    Spelled with the repo's standard coefficient signature ``(x, y, z, t)`` —
    the one ``exp.source``'s ``coeff_func`` and ``update_face_fluxes``'s
    velocity function already use — so a time- and space-dependent surface
    datum needs no new concept, only that ``FixedValue`` accept a callable and
    that the row builder evaluate it at the wall foot points **at the stage
    time**. Evaluating it once per step instead is precisely the defect A4
    exists to catch.
    """
    return np.full(np.shape(x), U0 * np.cos(OMEGA_A4 * t))


def _a4_case(ddt_scheme):
    """Mesh, field, equation, timestep and step count for one A4 run."""
    mesh = _make_mesh(N_A4, bodies={"wall": Plane(point=WALL_POINT_A4, normal=tuple(N_HAT))})
    u = CellField(mesh, ncomp=1, ngrow=1, name="u", ibm_bc={"wall": FixedValue(_oscillating_wall)})
    _fill(u, mesh, _stokes2_seed, 0.0)
    eqn = Equation(exp.ddt(u) - exp.laplacian(NU, u), schemes={"ddt": ddt_scheme})
    dt = DX_A4**2 / (DIFFUSION_DT_DIVISOR * NU)
    nsteps = round(PERIODS_A4 * PERIOD_A4 / dt)
    return mesh, u, eqn, dt, nsteps


@pytest.mark.slow
@B27_UNSTEADY_VALIDATION_MEASUREMENT
@pytest.mark.parametrize("ddt_scheme", ["Euler", "RK2", "RK4"])
def test_stokes_layer_phase_lag_matches_the_analytic_solution(blockamr_session, ddt_scheme):
    """A4. ``u = U0 exp(-y/delta) cos(omega t - y/delta)`` on an inclined wall.

    The amplitude alone passes even when the ghost values are a stage stale;
    the phase does not. Asserted at **three** depths so the two failure modes
    separate: a schedule bug shifts every depth by the same amount, a diffusion
    bug shifts them by different amounts. The failure message reports that
    decomposition (the common shift, in timesteps, and the residual after
    removing it) because "the phase is wrong" without it names neither.

    The wall datum is time-dependent (``FixedValue(_oscillating_wall)``), which
    is what makes the stage schedule observable at all: with a constant datum a
    reconstruction refreshed once per step and one refreshed once per stage
    produce identical numbers.

    Parametrised over the three ddt schemes because the defect is a property of
    the *stage* structure — Forward Euler has one stage and cannot exhibit it,
    so a suite that only ran Euler would be reporting a null result as a pass.
    The scheme is named in the equation's ``schemes`` (fvSchemes), which is
    where ``solve()`` looks a ddt scheme up; it is not a solver setting.

    Live for the whole parametrization since B15: ``solve()`` honours
    ``solution["ibm"]`` alongside ``solution["backend"]``, so the immersed wall
    is in the time loop, and ``RK2``/``RK4`` reach the stage schedule this test
    is about instead of raising ``NotImplementedError``.
    """
    mesh, u, eqn, dt, nsteps = _a4_case(ddt_scheme)
    shape = (N_A4, N_A4, N_A4)
    X, Y, Z = _coords(mesh, (0, 0, 0), shape)
    s = _normal_coord(X, Y, Z, WALL_POINT_A4)
    cells = [_probe_cell(s, d * DELTA_A4, _interior_mask(shape)) for d in PROBE_DEPTHS]

    def probe(_t):
        field = _assemble_field(u, shape)
        return [field[c] for c in cells]

    times, hist = _drive(
        eqn,
        _sol("ghostCell"),
        dt,
        0.0,
        nsteps,
        _ghost_seeder(u, mesh, _stokes2_seed),
        sample_every=8,
        probe=probe,
    )
    window = times >= times[-1] - FIT_PERIODS * PERIOD_A4

    rows, shifts = [], []
    for i, cell in enumerate(cells):
        depth = float(s[cell]) / DELTA_A4  # the probe's own depth, not the nominal
        amp, phase = _fit_harmonic(times[window], hist[window, i], OMEGA_A4)
        rows.append((depth, amp, U0 * np.exp(-depth), phase, -depth))
        shifts.append(_wrap(phase + depth))

    shift = float(np.mean(shifts))
    report = "\n".join(
        [
            (
                f"A4 ddt={ddt_scheme}  delta={DELTA_A4:.4f} = {SCALE_CELLS} dx  "
                f"omega={OMEGA_A4:.4f}  dt={dt:.2e}  steps={nsteps}"
            ),
            "  y/delta   amp        amp_exact   phase      phase_exact",
        ]
        + [f"  {d:7.3f}  {a:.6f}  {ae:.6f}  {p:9.4f}  {pe:9.4f}" for d, a, ae, p, pe in rows]
        + [
            (
                f"  common phase shift (schedule) = {shift:+.4f} rad "
                f"= {shift / (OMEGA_A4 * dt):+.2f} timesteps"
            ),
            f"  residual per depth (diffusion) = {[round(x - shift, 4) for x in shifts]}",
        ]
    )

    for _depth, amp, amp_exact, phase, phase_exact in rows:
        assert amp == pytest.approx(amp_exact, rel=AMP_RTOL), report
        assert _wrap(phase - phase_exact) == pytest.approx(0.0, abs=PHASE_ATOL), report


@pytest.mark.slow
@T18_FORCES
def test_stokes_layer_skin_friction_leads_the_wall_velocity_by_45_degrees(blockamr_session):
    """A4's third metric, and the only one that is purely a wall quantity.

    ``tau(t) = -rho nu du/dn|_w`` (``n`` into the fluid, so this is the
    traction the fluid exerts *on the body*) has the closed form

        tau = sqrt(2) rho nu U0 / delta * cos(omega t + pi/4)

    — amplitude ``sqrt(2) rho nu U0/delta`` at a **45 degree lead** over the
    wall's own velocity ``U0 cos(omega t)``. The lead is the sharp half: a pure
    phase, independent of ``rho``, ``nu``, ``U0`` and of any calibration, so it
    cannot be tuned into agreement. The formula is verified against numerical
    differentiation of the reference in
    ``test_the_wall_shear_formulas_match_numerical_differentiation``.

    Measured through a per-patch diagnostic, not by finite-differencing the
    solution: the wall gradient is what the reconstruction rows already carry
    (``alpha phi_w + beta dphi/dn = gamma``, per row), and re-deriving it from
    cell centres would measure the interior scheme instead. The API this asks
    for is ``blockamr.ibm.wall_gradient(field, patch, solution=...)`` returning
    the area-averaged ``dphi/dn`` on that patch with shape ``(ncomp,)`` — the
    same function A1's "wall flux ``dT/dr|_R``" metric needs.

    Red on T18: that diagnostic does not exist. It is imported inside the test
    body so the ``ImportError`` lands inside the xfail rather than taking the
    whole module down at collection.
    """
    from blockamr.ibm import wall_gradient

    mesh, u, eqn, dt, nsteps = _a4_case("Euler")
    solution = _sol("ghostCell")

    times, hist = _drive(
        eqn,
        solution,
        dt,
        0.0,
        nsteps,
        _ghost_seeder(u, mesh, _stokes2_seed),
        sample_every=8,
        probe=lambda _t: float(wall_gradient(u, "wall", solution=solution)[0]),
    )
    window = times >= times[-1] - FIT_PERIODS * PERIOD_A4
    amp, phase = _fit_harmonic(times[window], -NU * hist[window], OMEGA_A4)

    exact_amp = np.sqrt(2.0) * NU * U0 / DELTA_A4
    report = (
        f"A4 skin friction: amp={amp:.6e} (exact {exact_amp:.6e}), "
        f"phase={phase:+.4f} rad (exact {np.pi / 4:+.4f} = 45 deg lead)"
    )
    assert amp == pytest.approx(exact_amp, rel=AMP_RTOL), report
    assert _wrap(phase - np.pi / 4.0) == pytest.approx(0.0, abs=PHASE_ATOL), report


def _a4_wall_probe(mesh, shape, bc, t):
    """``laplacian(nu, T)`` under ``ghostCell`` with wall condition ``bc``.

    One field per call, seeded identically, so the *only* difference between
    two calls is the wall condition and the evaluation time.
    """
    T = CellField(mesh, ncomp=1, ngrow=1, name="T", ibm_bc={"wall": bc})
    _fill(T, mesh, _stokes2_seed, 0.0)
    out = evaluate(Equation(exp.laplacian(NU, T)), t=t, solution=_sol("ghostCell"))
    return _assemble_result(T, out, shape)


@T17_MOVING_BODIES
def test_a_time_dependent_wall_datum_is_evaluated_at_the_evaluation_time(blockamr_session):
    """The one piece of A4 that is testable *today*, at operator level.

    A4 and A6 both need a surface datum that varies in time; nothing else in
    the IBM suite does. That requirement can be isolated from ``solve()``
    entirely, because ``evaluate()`` **does** enter the IBM path and does take
    a ``t``: with ``FixedValue(f)`` for a callable ``f(x, y, z, t)``, two
    evaluations at two times must differ, and each must reproduce the result of
    the equivalent *constant* datum at that instant bit for bit — a callable is
    a schedule for the datum, not a different wall condition.

    Marked T17: a wall whose prescribed surface velocity is nonzero is the
    stationary-geometry half of moving bodies (no fresh cells, because the body
    itself does not move), and it is the half A4 needs.

    This is a cheap, every-commit-tier test, not a nightly one: it is one
    ``evaluate`` on a 24^3 box, and it fails today for a crisp, local reason —
    ``ibm_bc`` data reach the row builder through ``robin()``, whose ``gamma``
    goes straight into ``broadcast_gamma`` -> ``np.asarray(..., dtype=float)``,
    and the tables are built once per ``evaluate`` with no ``t`` in sight.
    """
    n = 24
    mesh = _make_mesh(n, bodies={"wall": Plane(point=WALL_POINT_A4, normal=tuple(N_HAT))})
    shape = (n, n, n)
    t_a, t_b = 0.0, 0.3 * PERIOD_A4

    varying_a = _a4_wall_probe(mesh, shape, FixedValue(_oscillating_wall), t_a)
    varying_b = _a4_wall_probe(mesh, shape, FixedValue(_oscillating_wall), t_b)
    const_a = _a4_wall_probe(mesh, shape, FixedValue(U0 * np.cos(OMEGA_A4 * t_a)), t_a)
    const_b = _a4_wall_probe(mesh, shape, FixedValue(U0 * np.cos(OMEGA_A4 * t_b)), t_b)

    assert not np.array_equal(varying_a, varying_b), "the wall datum did not change with t"
    np.testing.assert_array_equal(varying_a, const_a)
    np.testing.assert_array_equal(varying_b, const_b)


# ---------------------------------------------------------------------------
# A5 — Stokes' first problem: an impulsively started inclined wall
# ---------------------------------------------------------------------------

N_A5 = 32
DX_A5 = 1.0 / N_A5
WALL_POINT_A5 = WALL_POINT_A4

# The early-time singularity is real physics, not a numerical artefact: as
# t -> 0 the profile becomes a step and its wall gradient diverges, so *no*
# mesh resolves it and a run started at t = 0 would be measuring the initial
# condition's representation error. The case therefore starts at the time T0 at
# which the layer eta(t) = 2 sqrt(nu t) is already SCALE_CELLS cells wide,
# seeded with the exact profile there, and runs until the layer has grown by a
# factor of four (t = 16 T0). The solution is self-similar and has no memory of
# an origin, so a shifted start is the same problem — and the t^-1/2 shear
# decay is then asserted across that whole decade of growth.
ETA0_A5 = SCALE_CELLS * DX_A5
T0_A5 = ETA0_A5**2 / (4.0 * NU)
T_END_A5 = 16.0 * T0_A5

# Tolerances from the truncation estimate, not from taste: the reconstruction
# is linear-exact, so the ghost error is O(dx^2 u''), and against a profile of
# curvature scale eta that is (dx/eta)^2 = 4% at the start of the run and less
# later. The band carries it; the bulk sees only the interior scheme.
BAND_TOL_A5 = 0.06 * U0
BULK_TOL_A5 = 0.02 * U0

SHEAR_EXPONENT_ATOL = 0.05


def _stokes1(s, t):
    """``u = U0 erfc(s / (2 sqrt(nu t)))`` — the fluid solution, ``t > 0``."""
    return U0 * _erfc(np.asarray(s, dtype=float) / (2.0 * np.sqrt(NU * t)))


def _stokes1_seed(X, Y, Z, t):
    s = _normal_coord(X, Y, Z, WALL_POINT_A5)
    return np.where(s > 0.0, _stokes1(np.maximum(s, 0.0), t), 0.0)


def _a5_case():
    mesh = _make_mesh(N_A5, bodies={"wall": Plane(point=WALL_POINT_A5, normal=tuple(N_HAT))})
    u = CellField(mesh, ncomp=1, ngrow=1, name="u", ibm_bc={"wall": FixedValue(U0)})
    _fill(u, mesh, _stokes1_seed, T0_A5)
    eqn = Equation(exp.ddt(u) - exp.laplacian(NU, u), schemes={"ddt": "Euler"})
    dt = DX_A5**2 / (DIFFUSION_DT_DIVISOR * NU)
    nsteps = round((T_END_A5 - T0_A5) / dt)
    return mesh, u, eqn, dt, nsteps


@pytest.mark.slow
def test_stokes_first_problem_profile_follows_erfc(blockamr_session):
    """A5. The startup transient: an impulsively started wall, ``u = U0 erfc``.

    Where A4 asks whether a periodic response keeps its phase, A5 asks whether
    a *transient* keeps its shape while the only length scale in the problem
    grows through the mesh by a factor of four. A wall treatment accurate at
    one boundary-layer-to-cell ratio and not at another shows up here and
    nowhere else in the suite.

    ``L-inf`` is reported over the **band and the bulk separately** and the
    worst over the whole run is asserted (verification plan §4, §10): the band
    is O(n^2) cells against the bulk's O(n^3), so a single norm over the fluid
    is the bulk's norm wearing the band's name.

    The wall datum is a constant here (``FixedValue(U0)``), so A5 needs nothing
    A4 does not. It is red purely because ``solve()`` has no IBM path, which
    lets the wall value diffuse away into the zero-seeded solid instead of
    being held.
    """
    mesh, u, eqn, dt, nsteps = _a5_case()
    shape = (N_A5, N_A5, N_A5)
    X, Y, Z = _coords(mesh, (0, 0, 0), shape)
    s = _normal_coord(X, Y, Z, WALL_POINT_A5)
    fluid = s > 0.0
    band = fluid & (s < BAND_CELLS * DX_A5)
    bulk = fluid & ~band

    def probe(t):
        exact = np.where(fluid, _stokes1(np.maximum(s, 0.0), t), 0.0)
        err = np.abs(_assemble_field(u, shape) - exact)
        return [float(err[band].max()), float(err[bulk].max())]

    times, hist = _drive(
        eqn,
        _sol("ghostCell"),
        dt,
        T0_A5,
        nsteps,
        _ghost_seeder(u, mesh, _stokes1_seed),
        sample_every=25,
        probe=probe,
    )
    report = "\n".join(
        [f"A5 eta(t0)={ETA0_A5:.4f} = {SCALE_CELLS} dx, t0={T0_A5:.4f}, steps={nsteps}"]
        + [
            f"  t={t:.4f}  eta={2 * np.sqrt(NU * t) / DX_A5:5.2f} dx  "
            f"band Linf={b:.4e}  bulk Linf={k:.4e}"
            for t, (b, k) in zip(times[::8], hist[::8])
        ]
    )
    assert hist[:, 0].max() < BAND_TOL_A5, report
    assert hist[:, 1].max() < BULK_TOL_A5, report


@pytest.mark.slow
@T18_FORCES
def test_stokes_first_problem_wall_shear_decays_as_one_over_sqrt_t(blockamr_session):
    """A5's wall metric: ``tau(t) = rho U0 sqrt(nu / (pi t))``.

    The primary assertion is the **exponent**, fitted over the decade of layer
    growth the run covers — dimensionless, calibration-free, and a far sharper
    statement than any magnitude bound: a wall gradient that is merely offset
    by a constant, or that is evaluated one cell out from the surface, does not
    fit ``t^-1/2``. The prefactor is asserted too, but second.

    Same missing diagnostic as A4's skin friction (T18), imported inside the
    body so the failure is an xfail and not a collection error.
    """
    from blockamr.ibm import wall_gradient

    mesh, u, eqn, dt, nsteps = _a5_case()
    solution = _sol("ghostCell")

    times, hist = _drive(
        eqn,
        solution,
        dt,
        T0_A5,
        nsteps,
        _ghost_seeder(u, mesh, _stokes1_seed),
        sample_every=25,
        probe=lambda _t: float(wall_gradient(u, "wall", solution=solution)[0]),
    )
    tau = -NU * hist
    exponent = _fit_power(times, tau)
    exact = U0 * np.sqrt(NU / (np.pi * times))

    report = "\n".join(
        [f"A5 wall shear: fitted exponent {exponent:+.4f} (exact -0.5)"]
        + [
            f"  t={t:.4f}  tau={a:.6e}  exact={e:.6e}"
            for t, a, e in zip(times[::8], tau[::8], exact[::8])
        ]
    )
    assert exponent == pytest.approx(-0.5, abs=SHEAR_EXPONENT_ATOL), report
    np.testing.assert_allclose(tau, exact, rtol=0.1, err_msg=report)


# ---------------------------------------------------------------------------
# A6 — Womersley: an oscillating channel between two inclined walls
# ---------------------------------------------------------------------------

# alpha = h sqrt(omega/nu) = sqrt(2) h / delta, and the mesh is sized from it:
# delta is always SCALE_CELLS cells, so the resolution of the physics is the
# same at both ends of the sweep and only alpha changes. That makes the channel
# half-width alpha*SCALE_CELLS/sqrt(2) cells and fixes n. Two points rather
# than one because the case's whole claim is that it spans the diffusive
# (alpha ~ 1: quasi-parabolic, nearly in phase with the forcing) and the
# inertial (alpha >> 1: flat core lagging 90 degrees, with Stokes layers on the
# walls) regimes — one alpha tests one regime.
#
# n is chosen so the channel plus a solid slab of at least six cells on each
# side fits the box's normal extent, which for N_HAT on the unit cube is
# |(1,2,3)|_1 / |(1,2,3)|_2 = 6/sqrt(14) = 1.604 box sides.
WOMERSLEY_CASES = [(3.0, 24), (6.0, 36)]

G_A6 = 1.0  # amplitude of the driving acceleration, -1/rho dp/dx = G cos(wt)
PERIODS_A6 = 4
# y/h. The +-0.9 pair is the load-bearing one: it lands in the band (1 cell
# from the wall at alpha=3, 2 at alpha=6), where the reconstruction is what
# sets the answer. The centreline is the opposite extreme and is included as a
# control, not as evidence — at large alpha the core is inertia-dominated and
# u there is nearly u_p, which the test adds back itself, so agreement at
# y = 0 is close to unconditional and must not be read as a pass.
STATIONS_A6 = (-0.9, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 0.9)


def _womersley_uhat(y, h, delta, omega):
    """``u(y, t) = Re[uhat(y) exp(i omega t)]`` for ``-1/rho dp/dx = G cos(wt)``.

    ``nu u'' - i omega u = -G`` with ``u(+-h) = 0`` gives
    ``uhat = -i G/omega [1 - cosh(k y)/cosh(k h)]``, ``k = (1+i)/delta`` — the
    Womersley solution written in ``delta`` rather than ``alpha``
    (``k h = (1+i) alpha/sqrt(2)``).
    """
    k = (1.0 + 1.0j) / delta
    return (-1.0j * G_A6 / omega) * (1.0 - np.cosh(k * np.asarray(y)) / np.cosh(k * h))


def _womersley_v(y, h, delta, omega, t):
    """The *solved* field: ``v = u - u_p`` with ``u_p(t) = (G/omega) sin(wt)``.

    ``u_p`` is the spatially uniform response of an unbounded fluid to the same
    forcing, so subtracting it removes the body force from the equation exactly
    (``du_p/dt = G cos(wt)`` and ``u_p'' = 0``) and leaves plain diffusion with
    a **time-dependent wall value** ``v(+-h, t) = -(G/omega) sin(wt)``.

    That is the frame in which A6 needs no new operator: the driving pressure
    gradient becomes an oscillating datum on the two immersed walls, which is
    the capability A4 already asks for. The assertions are still made on the
    physical ``u = v + u_p`` — the Womersley profile — never on ``v``.
    """
    vhat = _womersley_uhat(y, h, delta, omega) + 1.0j * G_A6 / omega
    return np.real(vhat * np.exp(1.0j * omega * t))


def _a6_case(alpha, n):
    """Mesh, field, equation, seeding function and run settings for one alpha."""
    dx = 1.0 / n
    delta = SCALE_CELLS * dx
    omega = 2.0 * NU / delta**2
    h = alpha * delta / np.sqrt(2.0)
    centre = np.array([0.5, 0.5, 0.5])
    bodies = {
        "lower": Plane(point=tuple(centre - h * N_HAT), normal=tuple(N_HAT)),
        "upper": Plane(point=tuple(centre + h * N_HAT), normal=tuple(-N_HAT)),
    }

    def wall(x, y, z, t):
        return np.full(np.shape(x), -(G_A6 / omega) * np.sin(omega * t))

    def seed(X, Y, Z, t):
        y = _normal_coord(X, Y, Z, centre)
        inside = np.abs(y) < h
        return np.where(inside, _womersley_v(np.clip(y, -h, h), h, delta, omega, t), 0.0)

    mesh = _make_mesh(n, bodies=bodies)
    v = CellField(
        mesh,
        ncomp=1,
        ngrow=1,
        name="v",
        ibm_bc={"lower": FixedValue(wall), "upper": FixedValue(wall)},
    )
    _fill(v, mesh, seed, 0.0)
    eqn = Equation(exp.ddt(v) - exp.laplacian(NU, v), schemes={"ddt": "Euler"})
    dt = dx**2 / (DIFFUSION_DT_DIVISOR * NU)
    period = 2.0 * np.pi / omega
    cfg = {
        "delta": delta,
        "omega": omega,
        "h": h,
        "centre": centre,
        "dt": dt,
        "period": period,
        "nsteps": round(PERIODS_A6 * period / dt),
    }
    return mesh, v, eqn, seed, cfg


@pytest.mark.slow
@B27_UNSTEADY_VALIDATION_MEASUREMENT
@pytest.mark.parametrize("alpha, n", WOMERSLEY_CASES, ids=["alpha-3", "alpha-6"])
def test_womersley_amplitude_and_phase_profiles(blockamr_session, alpha, n):
    """A6. Two immersed walls, driven by an oscillating pressure gradient.

    What this adds over A4: **two** walls, close enough together to interact.
    A4's single wall can be right about its own layer and still leave the two
    reconstructions of a channel inconsistent with each other — the profile has
    to close in the middle, and only a case with two walls asks it to. The
    amplitude *and* the phase are asserted at nine stations across the
    channel, symmetric about the centreline, so a difference between the two
    walls reads as an asymmetry rather than as one number being off. The
    stations that carry the case are the outermost pair (see
    :data:`STATIONS_A6`); the centreline is a control.

    ``alpha`` is swept because the profile shape is a function of it alone, and
    the mesh is sized per ``alpha`` to keep ``delta`` at ``SCALE_CELLS`` cells,
    so the two points differ in physics and not in resolution.

    Red for the same T15 reason as A4 and A5: with no IBM in ``solve()`` there
    are no walls, the channel is not a channel, and the field relaxes to the
    uniform ``u_p`` of an unbounded fluid.
    """
    mesh, v, eqn, seed, cfg = _a6_case(alpha, n)
    shape = (n, n, n)
    X, Y, Z = _coords(mesh, (0, 0, 0), shape)
    y = _normal_coord(X, Y, Z, cfg["centre"])
    interior = _interior_mask(shape)
    cells = [_probe_cell(y, f * cfg["h"], interior) for f in STATIONS_A6]

    def probe(t):
        field = _assemble_field(v, shape)
        u_p = (G_A6 / cfg["omega"]) * np.sin(cfg["omega"] * t)
        return [field[c] + u_p for c in cells]

    times, hist = _drive(
        eqn,
        _sol("ghostCell"),
        cfg["dt"],
        0.0,
        cfg["nsteps"],
        _ghost_seeder(v, mesh, seed),
        sample_every=8,
        probe=probe,
    )
    window = times >= times[-1] - FIT_PERIODS * cfg["period"]

    rows = []
    for i, cell in enumerate(cells):
        y_probe = float(y[cell])
        uhat = _womersley_uhat(y_probe, cfg["h"], cfg["delta"], cfg["omega"])
        amp, phase = _fit_harmonic(times[window], hist[window, i], cfg["omega"])
        rows.append((y_probe / cfg["h"], amp, abs(uhat), phase, float(np.angle(uhat))))

    report = "\n".join(
        [
            (
                f"A6 alpha={alpha} n={n}  h={cfg['h']:.4f}  delta={cfg['delta']:.4f}"
                f" = {SCALE_CELLS} dx  steps={cfg['nsteps']}"
            ),
            "    y/h     amp        amp_exact   phase      phase_exact",
        ]
        + [f"  {q:7.3f}  {a:.6f}  {ae:.6f}  {p:9.4f}  {pe:9.4f}" for q, a, ae, p, pe in rows]
    )

    for _station, amp, amp_exact, phase, phase_exact in rows:
        assert amp == pytest.approx(amp_exact, rel=AMP_RTOL), report
        assert _wrap(phase - phase_exact) == pytest.approx(0.0, abs=PHASE_ATOL), report


# ---------------------------------------------------------------------------
# A8 — a rotating wall must not leak a radial scalar
# ---------------------------------------------------------------------------

N_A8 = 32
NZ_A8 = 4
R_A8 = 0.2
CENTRE_A8 = (0.5, 0.5)
AXIS_A8 = 2
OMEGA_A8 = 1.0 / R_A8  # so the surface speed is exactly |u| = OMEGA*R = 1
A_A8, B_A8 = 0.3, 0.5  # T(r) = A + B (r^2 - R^2); T|_R = A exactly

REVOLUTIONS_A8 = 10
CFL_A8 = 0.4

#: The largest ``|T - T_exact|`` tolerated in the band after ten revolutions,
#: as a fraction of the scalar's own range over the fluid. It is a
#: **specification**, not a measurement: §10 requires a non-conservative
#: method's drift to be characterized rather than asserted to be zero, and the
#: measurement cannot be taken while T15 blocks the run. What *is* derivable is
#: the shape of the budget — the bulk drift is exactly zero (see the test's
#: docstring), so the whole of it belongs to the band. Whoever lands T15 must
#: replace this number with the measured one.
DRIFT_FRACTION_A8 = 0.05


def _t_exact_a8(X, Y, _Z):
    r2 = (X - CENTRE_A8[0]) ** 2 + (Y - CENTRE_A8[1]) ** 2
    return A_A8 + B_A8 * (r2 - R_A8**2)


def _rotation_velocity(x, y, z, t):
    """``u = omega x r`` — divergence-free analytically *and* discretely, and
    exactly tangential to a concentric cylinder."""
    return -OMEGA_A8 * (y - CENTRE_A8[1]), OMEGA_A8 * (x - CENTRE_A8[0]), np.zeros_like(x)


@pytest.mark.slow
@B27_UNSTEADY_VALIDATION_MEASUREMENT
@pytest.mark.parametrize("ddt_scheme", ["RK2", "RK4"])
def test_rotating_wall_does_not_leak_a_radial_scalar(blockamr_session, ddt_scheme):
    """A8. A curved wall under a large tangential velocity, for many revolutions.

    ``u = omega x r`` is tangential to the cylinder everywhere and
    ``T(r) = A + B(r^2 - R^2)`` is radial, so ``div(u T) = u . grad T == 0``:
    the exact solution *is* the initial condition, for all time, and every
    departure from it is error with nothing analytic to subtract off. Rung 8
    already proves the single ``evaluate`` is exact here — bitwise, on the
    linear flux — so what A8 adds is *time*: the wall sees ``|u| = 1`` at its
    surface for ten revolutions, and a reconstruction that leaks a little of
    the wall value into the tangential flux each step accumulates it.

    Two different statements, both derivable, asserted separately. In the
    **bulk** the discrete divergence of this configuration cancels to the last
    bit, so the bulk drift is not "small", it is *zero*. In the **band** the
    trilinear reconstruction of a quadratic ``T`` carries an ``O(dx^2)`` ghost
    error and ``ghostCell`` is not conservative, so the honest expectation is a
    small nonzero drift — asserted as a characterized fraction of the field's
    range (:data:`DRIFT_FRACTION_A8`) rather than as zero, which §10 names as
    an anti-pattern.

    Forward Euler is deliberately **not** in the parametrization: central
    differencing of pure advection is unconditionally unstable under it, so an
    Euler run would measure the amplification of the time scheme and blame it
    on the wall. The multi-stage schemes are the only stable drivers here,
    which also makes A8 red on T15 immediately (``NotImplementedError``) rather
    than after ten revolutions of wasted work.
    """
    mesh = _make_mesh(
        N_A8, nz=NZ_A8, bodies={"cyl": Cylinder(centre=CENTRE_A8, radius=R_A8, axis=AXIS_A8)}
    )
    shape = (N_A8, N_A8, NZ_A8)
    T = CellField(mesh, ncomp=1, ngrow=1, name="T", ibm_bc={"cyl": FixedValue(A_A8)})
    _fill(T, mesh, lambda X, Y, Z, _t: _t_exact_a8(X, Y, Z), 0.0)
    # The exact solution is steady, so the analytic outer halo is written once
    # and stays correct for the whole run.
    _ghost_seeder(T, mesh, lambda X, Y, Z, _t: _t_exact_a8(X, Y, Z))(0.0)

    phi = FaceField(mesh, ncomp=1, ngrow=T.ngrow, name="phi")
    update_face_fluxes(phi[0], _rotation_velocity, mesh.geom(0), t=0.0)
    eqn = Equation(exp.ddt(T) + exp.div(phi, T), schemes={"Div": "linear", "ddt": ddt_scheme})

    X, Y, Z = _coords(mesh, (0, 0, 0), shape)
    exact = _t_exact_a8(X, Y, Z)
    r = np.hypot(X - CENTRE_A8[0], Y - CENTRE_A8[1])
    dx = float(mesh.geom(0).cell_size()[0])
    fluid = r > R_A8
    band = fluid & (r < R_A8 + BAND_CELLS * dx)
    bulk = fluid & ~band
    scalar_range = float(exact[fluid].max() - exact[fluid].min())

    dt = CFL_A8 * dx / (OMEGA_A8 * np.hypot(0.5, 0.5))
    nsteps = round(REVOLUTIONS_A8 * 2.0 * np.pi / OMEGA_A8 / dt)
    for step in range(nsteps):
        solve(eqn, dt=dt, t=step * dt, solution=_sol("ghostCell"))

    diff = _assemble_field(T, shape) - exact
    err = np.abs(diff)
    integral = float(diff[fluid].sum()) * dx * dx
    report = (
        f"A8 ddt={ddt_scheme}  {REVOLUTIONS_A8} revolutions, {nsteps} steps\n"
        f"  band Linf={err[band].max():.4e}  bulk Linf={err[bulk].max():.4e}  "
        f"(range {scalar_range:.4f}, integral drift {integral:+.4e})"
    )
    np.testing.assert_allclose(err[bulk], 0.0, atol=1e-12, err_msg=report)
    assert err[band].max() < DRIFT_FRACTION_A8 * scalar_range, report


# ---------------------------------------------------------------------------
# Oracle 2: the closed forms themselves
# ---------------------------------------------------------------------------


def _pde_residual(u, s, t, length_scale, time_scale, forcing=0.0):
    """``du/dt - nu d2u/ds2 - forcing``, by central differences.

    The steps are the scales divided by 3000: small enough that truncation
    (``O(h^2 u'''')``, ~1e-6 here, dominated by the time difference) is well
    below the tolerance, large enough that cancellation in the second
    difference (~1e-9) is too.
    """
    hs, ht = length_scale / 3000.0, time_scale / 3000.0
    dudt = (u(s, t + ht) - u(s, t - ht)) / (2.0 * ht)
    d2uds2 = (u(s + hs, t) - 2.0 * u(s, t) + u(s - hs, t)) / hs**2
    return dudt - NU * d2uds2 - forcing


def test_the_analytic_references_satisfy_their_own_pdes():
    """The second oracle. A validation suite is only as good as its reference.

    Each closed form is checked against the equation it is supposed to solve —
    numerically, by central differences, so an algebra slip in a ``cosh`` or a
    dropped factor of two cannot survive — and against its own boundary
    condition. Green, and it must stay green: these three functions are what
    A4, A5 and A6 assert their solutions against.
    """
    # A4: pure diffusion, wall value U0 cos(omega t).
    t4 = 0.3 * PERIOD_A4
    s4 = np.linspace(0.3, 3.0, 17) * DELTA_A4
    np.testing.assert_allclose(_pde_residual(_stokes2, s4, t4, DELTA_A4, PERIOD_A4), 0.0, atol=1e-5)
    assert _stokes2(0.0, t4) == pytest.approx(U0 * np.cos(OMEGA_A4 * t4), rel=1e-12)

    # A5: pure diffusion, wall value U0 for every t > 0, decaying to 0 far away.
    t5 = 4.0 * T0_A5
    eta5 = 2.0 * np.sqrt(NU * t5)
    np.testing.assert_allclose(
        _pde_residual(_stokes1, np.linspace(0.3, 3.0, 17) * eta5, t5, eta5, t5), 0.0, atol=1e-5
    )
    assert _stokes1(0.0, t5) == pytest.approx(U0, rel=1e-12)
    assert _stokes1(10.0 * eta5, t5) == pytest.approx(0.0, abs=1e-12)

    # A6: diffusion plus the uniform oscillating body force, no-slip at +-h.
    alpha, n = WOMERSLEY_CASES[1]
    delta = SCALE_CELLS / n
    omega = 2.0 * NU / delta**2
    h = alpha * delta / np.sqrt(2.0)
    period = 2.0 * np.pi / omega

    def u_womersley(y, t):
        return np.real(_womersley_uhat(y, h, delta, omega) * np.exp(1.0j * omega * t))

    t6 = 0.3 * period
    y6 = np.linspace(-0.9, 0.9, 19) * h
    np.testing.assert_allclose(
        _pde_residual(u_womersley, y6, t6, delta, period, forcing=G_A6 * np.cos(omega * t6)),
        0.0,
        atol=1e-5,
    )
    for wall in (-h, h):
        assert u_womersley(wall, t6) == pytest.approx(0.0, abs=1e-12)

    # ...and the change of variable A6 actually solves is the same problem: the
    # solved v must equal u - u_p, and must carry the oscillating wall datum.
    u_p = (G_A6 / omega) * np.sin(omega * t6)
    np.testing.assert_allclose(_womersley_v(y6, h, delta, omega, t6), u_womersley(y6, t6) - u_p)
    assert _womersley_v(h, h, delta, omega, t6) == pytest.approx(-u_p, abs=1e-12)


def test_the_wall_shear_formulas_match_numerical_differentiation():
    """The third oracle, and the one the 45-degree claim rests on.

    ``tau = -rho nu du/dn|_w`` is the traction on the *body* (``n`` points into
    the fluid), and the two closed forms A4 and A5 assert against are

        A4:  tau = sqrt(2) rho nu U0/delta cos(omega t + pi/4)
        A5:  tau = rho U0 sqrt(nu / (pi t))

    The 45-degree *lead* in A4 is one sign convention away from a 135-degree
    lag, and no amount of care in prose settles which; differentiating the
    reference numerically does. Checked at several phases of the cycle, so a
    formula that happens to agree at one instant cannot pass.
    """
    hs = DELTA_A4 / 3000.0
    tau_amp = np.sqrt(2.0) * NU * U0 / DELTA_A4
    for frac in (0.0, 0.17, 0.4, 0.63, 0.85):
        t = frac * PERIOD_A4
        numeric = -NU * float(_stokes2(hs, t) - _stokes2(-hs, t)) / (2.0 * hs)
        closed = tau_amp * np.cos(OMEGA_A4 * t + np.pi / 4.0)
        # To 1e-6 of the *amplitude*, not of the instantaneous value: 0.63 of a
        # period is within 0.03 of a zero crossing, where a relative tolerance
        # measures nothing but the central difference's own truncation.
        assert numeric == pytest.approx(closed, abs=1e-6 * tau_amp)

    for mult in (1.0, 2.0, 4.0, 16.0):
        t = mult * T0_A5
        h = 2.0 * np.sqrt(NU * t) / 3000.0
        numeric = -NU * float(_stokes1(h, t) - _stokes1(-h, t)) / (2.0 * h)
        assert numeric == pytest.approx(U0 * np.sqrt(NU / (np.pi * t)), rel=1e-6)
