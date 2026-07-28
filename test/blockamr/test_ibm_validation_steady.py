# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""Rung 11, the **steady** analytic validation cases — A1, A2, A3 (and A7).

`plans/IBM/ibm-verification-plan.md` §9.1, the steady half of the analytic
set. Every case here has a closed-form solution, so the assertion is an error
norm and an **observed order**, never a citation or a magic tolerance:

    A1  concentric annulus conduction   cylinder        T(r) = A ln r + B
    A2  inclined plane Poiseuille       two planes      u(s) = (G/2nu)(h^2 - s^2)
    A3  Taylor-Couette                  cylinder        u_theta(r) = A r + B/r
    A7  transient annulus conduction    two cylinders   Bessel series (optional)

**The ``A`` here means a validation case, never a design decision**: `design.md`
numbers its *decisions* ``A1``-``A15`` while this file and `verification.md` §9
number the *validation cases* ``A1``-``A8``, so a bare ``A1`` resolves only
against the note it sits in (review.md F13). Renaming the validation cases
``V1``-``V8`` is decided but deferred to **B43**, post-gate-G1 (review.md §4 Q6).

**Every case is projection-free** (§9.1): A1/A3/A7 transport a scalar or solve
a vector *Laplace* problem, A2 is a unidirectional body-force-driven flow whose
exact solution is divergence-free by construction. No pressure solve appears
anywhere in this file; if one is ever needed here the case has been mis-set.

**This file is the design gate** (§9.3). A1 says whether the steady wall
*gradient* is accurate enough. A value-based row can get the field visually
right and the gradient wrong, which is exactly what the wall-flux tests exist
to catch — so for A1 and A3 the field norm is the cheap half and the surface
quantity (flux, torque) is the discriminating one. If those do not converge,
flux rows (T19) are not an extension of the design but a restart of it.

Red by construction, for three named reasons:

1. The A2 and A3 cases cannot be *posed* at all. ``solve()`` has applied
   ``solution["ibm"]`` since B15, so the fields here are runs *with* a wall,
   but those two rows keep ``B26_STEADY_VALIDATION_MEASUREMENT`` after B26 ran
   (2026-07-28) precisely because no measurement reaches them: A2 needs a
   field-independent source term and A3 a callable wall datum (**B42**), and
   both rows raise before any wall arithmetic. The A1 annulus field row is
   **no longer among them**: it was re-pointed away from the retired
   ``D2_SOLUTION_ERROR_CONTRACT`` at B16 (review.md §4 Q18) and immediately
   x-passed, so it is unmarked and green — the same manufactured solution B16
   measures next door in ``test_ibm_solution_error.py`` (``ln r``,
   ``FixedValue(ln R)``, cylinder, ``cpp``), which is exactly what should have
   been expected, and B26 measured the two orders to prove it (band 1.768 /
   bulk 1.439; ``plans/IBM/tasks.md`` §1). Forward Euler is the
   pseudo-time driver throughout — it is the *driver*, not the object of study,
   and the answer must be independent of ``dt`` and of :data:`T_END`.
2. There is no surface-diagnostic API. The wall flux, the wall shear and the
   wall torque all need per-patch surface data that no public function returns
   today (``T18_FORCES``). Those tests are written against the spelling this
   file proposes — :func:`blockamr.ibm.wall_samples`, documented at
   :func:`_wall_samples_contract` — which is an **API decision to review**, not
   an established interface.
3. Two smaller gaps have no marker of their own and are called out where they
   bite: the DSL has no field-independent source term (A2 needs
   ``exp.body_force``; ``exp.source`` multiplies by the field, so it cannot
   express a constant drive), and an ``ibm_bc`` datum is a constant, so it
   cannot express the *rotating* wall velocity A3 requires (A3 passes a
   callable to ``FixedValue``).

Tier: **nightly** (§10) — the whole analytic set has a ~1 h budget and this
file is a small part of it. The module carries ``pytest.mark.slow`` (the root
``pyproject.toml`` marker) so it deselects with ``-m "not slow"``.

Every mask here is derived test-side from the analytic body — with no access to
the implementation's classification that is an *independent* oracle, and the
plan (§4, §10) prefers it. Band and bulk are always asserted separately; a
single global norm is the bulk's norm wearing the band's name.
"""

import numpy as np
import pytest

import blockamr
from blockamr.dsl import Equation, exp, solve
from blockamr.field import CellField
from blockamr.ibm import Cylinder, FixedValue, Plane
from blockamr.mesh import Mesh

from .ibm_gaps import B26_STEADY_VALIDATION_MEASUREMENT, T18_FORCES

# The whole file is nightly tier (verification plan §10).
pytestmark = pytest.mark.slow

BACKEND = "cpp"

NZ = 4  # thin in the third direction; every case here is z-invariant

#: The refinement set of every order study in this file. Six meshes, not
#: three: ``L-inf`` on a cut geometry is non-monotone in ``n`` — how the surface
#: slices the cells it crosses changes with the mesh — so a three-point fit of
#: the same data moves by a whole order depending which three (the sibling
#: study ``test_ibm_solution_error`` measured 1.08 to 1.91 that way). Six points
#: make the least-squares fit stable enough to be a contract, and the whole file
#: still runs in well under a minute.
RESOLUTIONS = (32, 40, 48, 56, 64, 80)

#: The asserted floor on every observed order. First order is what a
#: trilinear (linear-exact) reconstruction over one solid layer owes at the
#: wall; the bulk of a steady *solution* error converges at the wall's rate too,
#: because the elliptic problem smears the wall error over the whole domain.
MIN_ORDER = 1.0

#: The §9.3 gate itself: the relative error of the finest-mesh surface
#: quantity. An order alone says the method converges to *something*; this says
#: the number it converges to is the right one and is usable. 5% is the design
#: decision this file proposes — it is the number to argue about, and it is
#: stated once here rather than sprinkled through the assertions.
GATE = 0.05

#: Forward Euler on 3-D diffusion is stable for ``dt <= 1/(2 nu sum_d 1/dx_d^2)``;
#: with cubic cells that is ``dx^2/(6 nu)``. Safety factor 2 leaves room for the
#: amplification the wall rows add on top of it.
DT_SAFETY = 12.0

#: Long enough for the diffusive transient across the unit box to have died at
#: the diffusivities used here (all O(1)): the state at ``T_END`` and at
#: ``2*T_END`` agree to ~1e-5 in the sibling study (``test_ibm_solution_error``).
T_END = 0.6

#: The band is the fluid shell the wall treatment owns: two cells, the reach of
#: the laplacian stencil plus the layer it reconstructs. Everything else is bulk.
BAND_CELLS = 2.0


# ---------------------------------------------------------------------------
# The surface-diagnostic API this file is written against (task T18)
# ---------------------------------------------------------------------------


def _wall_samples_contract():
    """The proposed spelling of the per-patch surface diagnostic — **invented**.

    Nothing in ``src/`` returns wall data today: the wall table carries
    ``patch[r]`` "for diagnostics and forces" (``plans/IBM/ibm-row-format.md``
    §2) and stops there. A1's wall flux, A2's wall shear and A3's wall torque
    all need the same three things — where the surface is, which way it faces,
    and what the field does there — so this file asks for one function rather
    than one per quantity::

        from blockamr.ibm import wall_samples

        samples = wall_samples(T, solution={"ibm": "ghostCell", "backend": "cpp"})
        s = samples["cyl"]      # keyed like mesh.bodies / ibm_bc

        s.point    # (n, 3)          surface point of each wall row
        s.normal   # (n, 3)          unit normal, outward = into the fluid
        s.area     # (n,)            surface area carried by the row;
                   #                 sum(s.area) is the patch's wetted area
        s.value    # (n, ncomp)      phi at the wall
        s.grad     # (n, ncomp, 3)   d phi_i / d x_j at the wall

    ``grad`` is the full tensor, not just the normal derivative, because the
    viscous traction ``sigma.n`` with ``sigma = mu (grad u + grad u^T)`` cannot
    be built from ``d u/d n`` alone: on a rotating cylinder the two differ by
    the ``u_theta/r`` term and the resulting torque is wrong by ~40%. A force
    diagnostic that only knows the normal derivative reports the wrong force.

    ``area`` is the demanding entry — a per-row surface area is a cut-cell
    aperture in all but name, so T18's force diagnostic may well depend on
    T19's flux rows. The pointwise metrics (A1's ``dT/dn``, A2's shear) need
    only ``normal`` and ``grad``; only the integrated ones (A1's flux, A3's
    torque) need ``area``.
    """
    raise NotImplementedError("documentation only — see the tests that use it")


def _normal_derivative(samples):
    """``d phi/d n`` at every wall sample, from the wall gradient and normal."""
    return np.einsum("nij,nj->ni", samples.grad, samples.normal)


def _traction(samples, mu):
    """Viscous wall traction ``sigma.n``, ``sigma = mu (grad u + grad u^T)``.

    The pressure part of the stress is absent by construction: every case in
    this file is projection-free (§9.1), so the traction on the immersed
    surface is viscous only.
    """
    sigma = mu * (samples.grad + np.transpose(samples.grad, (0, 2, 1)))
    return np.einsum("nij,nj->ni", sigma, samples.normal)


# ---------------------------------------------------------------------------
# Mesh, fields, norms — shared by every case
# ---------------------------------------------------------------------------


def _make_mesh(n, bodies, periodic=(0, 0, 1)):
    """``n x n x NZ`` cells with **cubic** cells, periodic in z only.

    ``z`` spans ``NZ/n`` so ``dz == dx``: the explicit diffusion limit is set by
    the smallest cell dimension, and a quasi-2-D box with ``dz >> dx`` would
    make the timestep a property of the padding direction rather than of the
    study. Periodic in z because every case here is z-invariant; non-periodic
    in x/y, where the halo carries the analytic Dirichlet datum instead.
    """
    box = blockamr.Box([0, 0, 0], [n - 1, n - 1, NZ - 1])
    rb = blockamr.RealBox([0.0, 0.0, 0.0], [1.0, 1.0, NZ / n])
    geom = blockamr.Geometry(box, rb, 0, list(periodic))
    ba = blockamr.BoxArray(box)
    ba.max_size(max(n, NZ))
    dm = blockamr.DistributionMapping(ba)
    mesh = Mesh(ba, dm, geom)
    mesh.bodies = bodies
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


def _seed(field, mesh, exact):
    """Fill the valid cells *and* the ghost band from ``exact(X, Y, Z)``.

    Valid cells: the initial condition. Every steady case here seeds the exact
    solution, which is also the fixed point — so the run has only the
    *discrete* deviation to relax and :data:`T_END` is a generous margin rather
    than a marginal one. Solid cells are seeded too: the IBM must reconstruct
    its near-wall stencil from its own BC and never lean on what it finds
    inside the body.

    Ghost band: the outer Dirichlet boundary. ``fill_boundary`` fills the
    periodic z halo and the inter-box halos and leaves the domain-exterior x/y
    ghosts alone, so this analytic seed survives every ``fill_patch`` of every
    step.

    ``exact`` returns the full ``(nx, ny, nz, ncomp)`` block, so one helper
    serves the scalar and the vector cases alike.
    """
    mf = field.mf[0]
    for mfi in blockamr.MFIterator(mf):
        arr = mf.copy_to_host(mfi)
        X, Y, Z = _coords(mesh, mfi.valid_box().small_end(), arr.shape[:3])
        arr[...] = exact(X, Y, Z)
        mf.copy_from(mfi, arr)
    field.fill_patch(0, 0.0)

    ng = mf.n_grow()
    for mfi in blockamr.MFIterator(mf):
        arr = mf.copy_grown_to_host(mfi)
        lo = [c - ng for c in mfi.valid_box().small_end()]
        X, Y, Z = _coords(mesh, lo, arr.shape[:3])
        arr[...] = exact(X, Y, Z)
        mf.copy_grown_from(mfi, arr)


def _assemble(field, n):
    """Stitch the field's valid cells into one ``(n, n, NZ, ncomp)`` array."""
    out = np.full((n, n, NZ, field.ncomp), np.nan)
    mf = field.mf[0]
    for mfi in blockamr.MFIterator(mf):
        lo = mfi.valid_box().small_end()
        arr = np.asarray(mf.copy_to_host(mfi))
        out[
            lo[0] : lo[0] + arr.shape[0],
            lo[1] : lo[1] + arr.shape[1],
            lo[2] : lo[2] + arr.shape[2],
            :,
        ] = arr
    assert not np.isnan(out).any(), "box decomposition did not cover the domain"
    return out


def _step_to_steady(eqn, mesh, diffusivity):
    """Drive ``eqn`` with forward Euler to :data:`T_END`.

    The equation is the transient form of the steady problem the case poses;
    the stepper is the pseudo-time driver, not the object of study.
    """
    dx = float(mesh.geom(0).cell_size()[0])
    dt = dx * dx / (DT_SAFETY * diffusivity)
    for step in range(round(T_END / dt)):
        solve(eqn, dt=dt, t=step * dt, solution={"ibm": "ghostCell", "backend": BACKEND})


def _band_and_bulk(dist, dx):
    """``(band, bulk)`` fluid masks from the analytic signed distance.

    ``dist`` is the union signed distance of the case's bodies — positive in
    the fluid — evaluated test-side at the cell centres. That makes the split
    an independent oracle rather than a copy of the implementation's own
    classification (§4, §10).
    """
    fluid = dist > 0.0
    band = fluid & (dist < BAND_CELLS * dx)
    return band, fluid & ~band


def _linf(err, mask):
    """``L-inf`` of a per-cell error over ``mask``; ``err`` may be vector."""
    return float(np.abs(err[mask]).max())


def _observed_order(errors):
    """Least-squares ``p`` in ``err ~ C dx^p`` over :data:`RESOLUTIONS`."""
    dx = 1.0 / np.array(RESOLUTIONS, dtype=float)
    slope, _intercept = np.polyfit(np.log(dx), np.log(np.array(errors, dtype=float)), 1)
    return float(slope)


def _report(label, errors, order):
    """Failure message: the norm at every resolution plus the fitted order."""
    rows = "\n".join(f"    n={n:3d}  {label}={e:.6e}" for n, e in zip(RESOLUTIONS, errors))
    return f"{label}: observed order {order:.3f} (floor {MIN_ORDER})\n{rows}"


# ---------------------------------------------------------------------------
# A1 — concentric annulus conduction: T = ln r about an immersed cylinder
# ---------------------------------------------------------------------------

A1_R = 0.25  # cylinder radius — resolved by the coarsest mesh of the study
A1_CENTRE = (0.5, 0.5)
A1_AXIS = 2
A1_ALPHA = 1.0  # laplacian coefficient (thermal diffusivity, and the flux gamma)

#: ``n -> (mesh, T)`` for the converged A1 solve; shared by the two A1 tests
#: rather than solved twice.
_A1: dict = {}


def _a1_exact(X, Y, Z):
    """``T = ln r`` about the cylinder axis.

    ``laplacian(T) == 0`` identically, so the *entire* steady residual is wall
    error and there is no bulk truncation term to subtract off — the sharpest
    form of A1 (verification plan §4). ``ln r`` is also the unique harmonic
    function matching the analytic Dirichlet halo on the box and
    ``FixedValue(ln R)`` on the immersed circle, so the exact solution of the
    discrete problem's continuum limit is known in closed form.
    """
    return np.log(np.hypot(X - A1_CENTRE[0], Y - A1_CENTRE[1]))[..., np.newaxis]


def _a1_converged(n):
    """Drive ``dT/dt = alpha laplacian(T)`` to steady state on an ``n`` mesh."""
    if n in _A1:
        return _A1[n]
    mesh = _make_mesh(n, {"cyl": Cylinder(centre=A1_CENTRE, radius=A1_R, axis=A1_AXIS)})
    T = CellField(mesh, ncomp=1, ngrow=1, name="T", ibm_bc={"cyl": FixedValue(np.log(A1_R))})
    _seed(T, mesh, _a1_exact)
    eqn = Equation(exp.ddt(T) - exp.laplacian(A1_ALPHA, T), schemes={"ddt": "Euler"})
    _step_to_steady(eqn, mesh, A1_ALPHA)
    _A1[n] = (mesh, T)
    return _A1[n]


def _a1_regions(mesh, n):
    """Band/bulk masks about the immersed circle, from the analytic body."""
    X, Y, _Z = _coords(mesh, (0, 0, 0), (n, n, NZ))
    dist = np.hypot(X - A1_CENTRE[0], Y - A1_CENTRE[1]) - A1_R
    return _band_and_bulk(dist[..., np.newaxis], float(mesh.geom(0).cell_size()[0]))


def test_a1_annulus_temperature_converges_in_band_and_bulk(blockamr_session):
    """A1, the field half — the cheap metric, and the one that can look right.

    **Unmarked since B16, and the marker's removal is the finding.** This row
    was expected to be the anomaly of the suite — the same manufactured
    solution (``ln r``, ``FixedValue(ln R)``, an immersed cylinder on ``cpp``)
    that ``test_ibm_solution_error.py`` measures green, apparently still red
    here. It is not: run in the slow tier it **x-passes** (strict xfail, so the
    stale marker showed up as a failure), reproduced isolated and in-group. The
    marker is therefore removed under the rule its own module states — a row
    that x-passes loses its marker (review.md §4 Q7/Q10) — and both orders are
    now enforced rather than expected-red. The pass predates B16 (nothing that
    session touched runs in this file).

    **Measured by B26, 2026-07-28** (``plans/IBM/tasks.md`` §1): band ``L-inf``
    order **1.768**, bulk **1.439**, both against :data:`MIN_ORDER` = 1.0, on
    the six meshes of :data:`RESOLUTIONS`, cpp backend. All twelve error values
    are **bit-identical** to B16's ``lnr-value`` wall/interior columns next
    door — this row and that one are the same numerical experiment, so A1's
    field half reproduces B16 rather than adding independent evidence.

    ``L-inf(T - T_exact)`` of the converged solution, asserted **separately**
    over the band and the bulk: the band is ``O(n)`` cells against the bulk's
    ``O(n^3)``, so a combined norm is the bulk's norm wearing the band's name
    (§4, §10 anti-patterns).

    This is the *weak* half of A1 on purpose. A value-based row can reproduce
    the field to plotting accuracy and still get ``dT/dr`` at the wall wrong,
    which is what the flux test below exists to catch — §9.1 lists both metrics
    for this case for exactly that reason.
    """
    band_err, bulk_err = [], []
    for n in RESOLUTIONS:
        mesh, T = _a1_converged(n)
        err = _assemble(T, n) - _a1_exact(*_coords(mesh, (0, 0, 0), (n, n, NZ)))
        band, bulk = _a1_regions(mesh, n)
        band_err.append(_linf(err, band))
        bulk_err.append(_linf(err, bulk))

    band_order = _observed_order(band_err)
    bulk_order = _observed_order(bulk_err)
    assert band_order > MIN_ORDER, _report("band Linf", band_err, band_order)
    assert bulk_order > MIN_ORDER, _report("bulk Linf", bulk_err, bulk_order)


@T18_FORCES
def test_a1_annulus_wall_flux_matches_the_analytic_gradient(blockamr_session):
    """A1, the discriminating half: the **wall flux**, ``dT/dr|_R = A/R``.

    With ``T = ln r`` (so ``A = 1``) the normal derivative on the surface is the
    constant ``1/R``, and the total conductive flux through the immersed patch
    is ``alpha * (1/R) * (2 pi R Lz) = 2 pi alpha Lz`` — independent of ``R``,
    and of everything the discretisation does. Two assertions, because they
    fail differently:

    * the **pointwise** ``L-inf`` of ``dT/dn - 1/R`` over the wall samples,
      with an observed order — a row whose reconstruction is only linear-*ish*
      shows up here as a stalled order even when the integral still cancels;
    * the **integrated** flux at the finest mesh against the closed form,
      relative, against the §9.3 gate — an order alone would be satisfied by
      converging to the wrong constant, and this is the number the design gate
      is actually about.

    Red for ``T18``: no public function returns per-patch surface data, so this
    is written against the proposed :func:`blockamr.ibm.wall_samples` (see
    :func:`_wall_samples_contract`). It is *also* blocked by ``T15`` — with
    ``solve()`` ignoring ``solution["ibm"]`` the converged field has no wall in
    it at all — so this test cannot go green before both land.
    """
    from blockamr.ibm import wall_samples

    pointwise, flux_rel = [], []
    for n in RESOLUTIONS:
        _mesh, T = _a1_converged(n)
        samples = wall_samples(T, solution={"ibm": "ghostCell", "backend": BACKEND})["cyl"]
        dtdn = _normal_derivative(samples)[:, 0]
        pointwise.append(float(np.abs(dtdn - 1.0 / A1_R).max()))

        flux = float(np.sum(A1_ALPHA * dtdn * samples.area))
        exact_flux = 2.0 * np.pi * A1_ALPHA * (NZ / n)  # 2 pi alpha R Lz / R
        flux_rel.append(abs(flux - exact_flux) / abs(exact_flux))

    order = _observed_order(pointwise)
    assert order > MIN_ORDER, _report("wall dT/dn Linf", pointwise, order)
    assert flux_rel[-1] < GATE, _report("wall flux rel. err", flux_rel, _observed_order(flux_rel))


# ---------------------------------------------------------------------------
# A2 — inclined plane Poiseuille: a channel that is *not* grid-aligned
# ---------------------------------------------------------------------------

#: The wall inclination. 30 degrees to the x axis, in the x-y plane: the wall
#: normal is ``(cos 30, sin 30, 0)``, whose slope ``tan 60 = sqrt(3)`` is
#: irrational, so the staircase never repeats and no cell face ever lines up
#: with the surface. An axis-aligned wall is a degenerate probe — the
#: interpolation collapses to 1-D and hides every interpolation bug — and 45
#: degrees is nearly as degenerate, being symmetric in x and y. The flow is
#: along **z**, which is perpendicular to the tilt, so ``u`` is z-invariant and
#: the z direction can stay periodic.
A2_THETA = np.pi / 6.0
A2_NORMAL = np.array([np.cos(A2_THETA), np.sin(A2_THETA), 0.0])
A2_CENTRE = np.array([0.5, 0.5, 0.0])  # the channel's mid-plane passes here
A2_H = 0.25  # channel half-width
A2_NU = 1.0  # kinematic viscosity; rho = 1, so mu = nu
A2_G = 1.0  # body force per unit mass, along +z


def _a2_s(X, Y):
    """Wall-normal coordinate: the signed distance from the channel centre."""
    return (X - A2_CENTRE[0]) * A2_NORMAL[0] + (Y - A2_CENTRE[1]) * A2_NORMAL[1]


def _a2_exact(X, Y, Z):
    """``u = (0, 0, (G/2nu)(h^2 - s^2))`` — the inclined Poiseuille profile.

    Divergence-free by construction: the only component is ``u_z`` and it does
    not depend on z. The momentum balance is ``nu laplacian(u_z) + G = 0``
    (``|grad s| = 1``), so no pressure enters and the case is projection-free.
    """
    out = np.zeros(np.shape(X) + (3,))
    out[..., 2] = (A2_G / (2.0 * A2_NU)) * (A2_H**2 - _a2_s(X, Y) ** 2)
    return out


def _a2_body_force(x, y, z, t):
    """The uniform drive ``f = (0, 0, G)``, evaluated at the cell centres."""
    out = np.zeros(np.shape(x) + (3,))
    out[..., 2] = A2_G
    return out


def _a2_bodies():
    """The two half-space walls: fluid is ``|s| < h``, solid on both sides."""
    return {
        "lower": Plane(point=tuple(A2_CENTRE - A2_H * A2_NORMAL), normal=tuple(A2_NORMAL)),
        "upper": Plane(point=tuple(A2_CENTRE + A2_H * A2_NORMAL), normal=tuple(-A2_NORMAL)),
    }


_A2: dict = {}


def _a2_converged(n):
    """Drive ``dU/dt = nu laplacian(U) + G z_hat`` to steady state.

    **Blocked on a missing DSL term.** ``exp.source(S, U)`` is ``S*phi`` — it
    multiplies by the field, so it cannot express a field-independent drive,
    and the C++ backend has no kernel for it either. The minimal spelling this
    file proposes is ``exp.body_force(f, U)`` with ``f(x, y, z, t)`` returning
    the per-component forcing, added to the RHS as it stands. Without it there
    is no way to pose A2 at all through the public API.
    """
    if n in _A2:
        return _A2[n]
    mesh = _make_mesh(n, _a2_bodies())
    U = CellField(
        mesh,
        ncomp=3,
        ngrow=1,
        name="U",
        ibm_bc={"lower": FixedValue((0.0, 0.0, 0.0)), "upper": FixedValue((0.0, 0.0, 0.0))},
    )
    _seed(U, mesh, _a2_exact)
    eqn = Equation(
        exp.ddt(U) - exp.laplacian(A2_NU, U) - exp.body_force(_a2_body_force, U),
        schemes={"ddt": "Euler"},
    )
    _step_to_steady(eqn, mesh, A2_NU)
    _A2[n] = (mesh, U)
    return _A2[n]


def _a2_regions(mesh, n):
    """Band/bulk masks inside the channel, from the analytic half-spaces."""
    X, Y, _Z = _coords(mesh, (0, 0, 0), (n, n, NZ))
    dist = A2_H - np.abs(_a2_s(X, Y))  # union sdf of the two planes
    return _band_and_bulk(dist[..., np.newaxis], float(mesh.geom(0).cell_size()[0]))


@B26_STEADY_VALIDATION_MEASUREMENT
def test_a2_inclined_poiseuille_profile_converges_in_band_and_bulk(blockamr_session):
    """A2, the profile: ``L-inf(u - u_exact)``, band and bulk separately.

    What this case uniquely catches is **staircase error**. On a grid-aligned
    wall the trilinear reconstruction degenerates to a 1-D two-point stencil
    and the profile comes out right whatever the interpolation does; tilted, an
    interpolation bug is an ``O(dx)`` term in the profile and a stalled order
    here.

    Two immersed walls rather than one, so the case also exercises patch
    attribution (``patch[r]``) — the two half-spaces are separate bodies with
    separate ``ibm_bc`` entries and the rows must find the *nearer* surface.
    """
    band_err, bulk_err = [], []
    for n in RESOLUTIONS:
        mesh, U = _a2_converged(n)
        err = _assemble(U, n) - _a2_exact(*_coords(mesh, (0, 0, 0), (n, n, NZ)))
        band, bulk = _a2_regions(mesh, n)
        band_err.append(_linf(err, np.broadcast_to(band, err.shape)))
        bulk_err.append(_linf(err, np.broadcast_to(bulk, err.shape)))

    band_order = _observed_order(band_err)
    bulk_order = _observed_order(bulk_err)
    assert band_order > MIN_ORDER, _report("band Linf(u)", band_err, band_order)
    assert bulk_order > MIN_ORDER, _report("bulk Linf(u)", bulk_err, bulk_order)


@T18_FORCES
def test_a2_inclined_poiseuille_wall_shear_matches_the_analytic_value(blockamr_session):
    """A2, the surface metric: the wall shear ``tau = rho G h`` on both walls.

    With ``u_z = (G/2nu)(h^2 - s^2)`` the wall-normal derivative at ``|s| = h``
    is ``(G/nu) h``, so the viscous traction is ``mu (G/nu) h = rho G h``,
    directed along the flow (``+z``) on *both* walls — a constant over each
    surface, which makes the pointwise ``L-inf`` over the wall samples the
    natural norm and needs no surface area at all.

    (The plan's §9.1 table writes this as ``rho nu G h``; with ``G`` defined as
    the force per unit mass that its own profile formula implies — ``nu u'' =
    -G`` — the viscosity cancels and the shear is ``rho G h``. The dependence
    on ``nu`` in the table is a slip of the pen, not a different case.)

    Asserted **per patch**, not summed: a wall treatment that is biased along
    the tilt direction gets one wall too high and the other too low, and a sum
    would cancel exactly that error.
    """
    from blockamr.ibm import wall_samples

    errors = {"lower": [], "upper": []}
    for n in RESOLUTIONS:
        _mesh, U = _a2_converged(n)
        samples = wall_samples(U, solution={"ibm": "ghostCell", "backend": BACKEND})
        for patch in ("lower", "upper"):
            traction = _traction(samples[patch], A2_NU)  # rho = 1 -> mu = nu
            exact = np.zeros_like(traction)
            exact[:, 2] = A2_G * A2_H
            errors[patch].append(float(np.abs(traction - exact).max()))

    for patch in ("lower", "upper"):
        order = _observed_order(errors[patch])
        assert order > MIN_ORDER, _report(f"{patch} wall shear Linf", errors[patch], order)
        rel = errors[patch][-1] / (A2_G * A2_H)
        assert rel < GATE, _report(f"{patch} wall shear rel. err {rel:.3e}", errors[patch], order)


# ---------------------------------------------------------------------------
# A3 — Taylor-Couette: a curved wall with a nonzero tangential velocity
# ---------------------------------------------------------------------------

A3_RI = 0.2  # the immersed (inner) cylinder
A3_RO = 0.45  # the virtual outer cylinder — sets the constants, is not meshed
A3_CENTRE = (0.5, 0.5)
A3_AXIS = 2
A3_OMEGA = 1.0  # inner cylinder angular velocity
A3_NU = 1.0  # kinematic viscosity; rho = 1, so mu = nu

#: ``u_theta = A r + B/r`` with ``u_theta(RI) = OMEGA*RI`` and
#: ``u_theta(RO) = 0`` — solved here rather than written out, so the two
#: boundary conditions stay visible.
A3_A, A3_B = np.linalg.solve(
    np.array([[A3_RI, 1.0 / A3_RI], [A3_RO, 1.0 / A3_RO]]),
    np.array([A3_OMEGA * A3_RI, 0.0]),
)


def _a3_exact(X, Y, Z):
    """Taylor-Couette in Cartesian components: ``u = (A + B/r^2) z_hat x r``.

    Each Cartesian component is **harmonic**: ``y/r^2`` and ``x/r^2`` are the
    partial derivatives of ``ln r``, which is harmonic, and derivatives of
    harmonic functions are harmonic. So this is a pure vector *Laplace*
    problem — no pressure gradient, no advection, nothing to project — and yet
    the wall carries an O(1) tangential velocity, which is the thing A1 cannot
    test.
    """
    dx = X - A3_CENTRE[0]
    dy = Y - A3_CENTRE[1]
    f = A3_A + A3_B / (dx * dx + dy * dy)
    out = np.zeros(np.shape(X) + (3,))
    out[..., 0] = -f * dy
    out[..., 1] = f * dx
    return out


def _a3_wall_velocity(x, y, z):
    """``u = omega x r`` on the inner cylinder — **not** constant on the surface.

    This is the second capability gap: a rotating wall's datum is not uniform —
    its direction turns with theta — so it must be a *spatial* function of the
    row's surface point. B42 built the callable datum, but deliberately as a
    function **of time only** — ``f(x, y, z, t)`` evaluated at the wall foot
    points, where A4/A6's oscillating walls need the ``t`` and ignore the
    coordinates (Q25 OP-1, review.md §4). This spelling, ``f(x, y, z)``, now
    raises on its arity; whether A3 is served by widening the datum or by
    respelling this helper is decided at the session that schedules A3.
    """
    out = np.zeros(np.shape(x) + (3,))
    out[..., 0] = -A3_OMEGA * (y - A3_CENTRE[1])
    out[..., 1] = A3_OMEGA * (x - A3_CENTRE[0])
    return out


_A3: dict = {}


def _a3_converged(n):
    """Drive ``dU/dt = nu laplacian(U)`` to steady state around the cylinder."""
    if n in _A3:
        return _A3[n]
    mesh = _make_mesh(n, {"cyl": Cylinder(centre=A3_CENTRE, radius=A3_RI, axis=A3_AXIS)})
    U = CellField(mesh, ncomp=3, ngrow=1, name="U", ibm_bc={"cyl": FixedValue(_a3_wall_velocity)})
    _seed(U, mesh, _a3_exact)
    eqn = Equation(exp.ddt(U) - exp.laplacian(A3_NU, U), schemes={"ddt": "Euler"})
    _step_to_steady(eqn, mesh, A3_NU)
    _A3[n] = (mesh, U)
    return _A3[n]


def _a3_regions(mesh, n):
    """Band/bulk masks about the inner cylinder, from the analytic body."""
    X, Y, _Z = _coords(mesh, (0, 0, 0), (n, n, NZ))
    dist = np.hypot(X - A3_CENTRE[0], Y - A3_CENTRE[1]) - A3_RI
    return _band_and_bulk(dist[..., np.newaxis], float(mesh.geom(0).cell_size()[0]))


@B26_STEADY_VALIDATION_MEASUREMENT
def test_a3_taylor_couette_profile_converges_in_band_and_bulk(blockamr_session):
    """A3, the profile: ``L-inf(u - u_exact)``, band and bulk separately.

    The outer cylinder is *not* meshed — the analytic profile is harmonic
    componentwise, so seeding the box's ghost band with it is an exact
    Dirichlet condition and ``A r + B/r`` remains the unique solution of the
    box problem. That keeps the case to one immersed body ("inner immersed",
    §9.1) and avoids needing a body whose solid side is the *exterior* of a
    cylinder.

    What this adds over A1: the wall carries a tangential velocity of
    ``OMEGA*RI`` and the field is a vector, so a reconstruction that is right
    for a scalar and wrong for a rotating vector datum shows up here and
    nowhere above.
    """
    band_err, bulk_err = [], []
    for n in RESOLUTIONS:
        mesh, U = _a3_converged(n)
        err = _assemble(U, n) - _a3_exact(*_coords(mesh, (0, 0, 0), (n, n, NZ)))
        band, bulk = _a3_regions(mesh, n)
        band_err.append(_linf(err, np.broadcast_to(band, err.shape)))
        bulk_err.append(_linf(err, np.broadcast_to(bulk, err.shape)))

    band_order = _observed_order(band_err)
    bulk_order = _observed_order(bulk_err)
    assert band_order > MIN_ORDER, _report("band Linf(u)", band_err, band_order)
    assert bulk_order > MIN_ORDER, _report("bulk Linf(u)", bulk_err, bulk_order)


@T18_FORCES
def test_a3_taylor_couette_wall_torque_matches_the_analytic_value(blockamr_session):
    """A3, the discriminating metric: the **wall torque** on the inner cylinder.

    For ``u_theta = A r + B/r`` the shear stress is
    ``sigma_r,theta = mu r d(u_theta/r)/dr = -2 mu B / r^2``, so the torque the
    fluid exerts on the inner cylinder is ``-4 pi mu B`` per unit length — free
    of ``R``, of ``A``, and of everything the discretisation does.

    The torque is assembled test-side from the surface samples (position,
    traction, area) rather than asked for as a number: the post-processing is
    then an independent oracle, and the API stays one function. It is also the
    reason :func:`_wall_samples_contract` demands the **full** wall gradient
    tensor — using ``mu du/dn`` in place of the traction ``sigma.n`` drops the
    ``u_theta/r`` term and lands ~40% low, a wrong force that converges
    beautifully to the wrong answer.
    """
    from blockamr.ibm import wall_samples

    rel = []
    for n in RESOLUTIONS:
        _mesh, U = _a3_converged(n)
        s = wall_samples(U, solution={"ibm": "ghostCell", "backend": BACKEND})["cyl"]
        arm = s.point - np.array([A3_CENTRE[0], A3_CENTRE[1], 0.0])
        torque = np.sum(np.cross(arm, _traction(s, A3_NU)) * s.area[:, np.newaxis], axis=0)
        exact = -4.0 * np.pi * A3_NU * A3_B * (NZ / n)  # per unit length x Lz
        rel.append(abs(float(torque[2]) - exact) / abs(exact))

    order = _observed_order(rel)
    assert order > MIN_ORDER, _report("wall torque rel. err", rel, order)
    assert rel[-1] < GATE, _report("wall torque rel. err", rel, order)


# ---------------------------------------------------------------------------
# A7 — transient annulus conduction (optional, §9.1: the unsteady analogue of A1)
# ---------------------------------------------------------------------------

A7_RI, A7_RO = 0.2, 0.45  # both walls immersed: the annulus needs two bodies
A7_CENTRE = (0.5, 0.5)
A7_AXIS = 2
A7_ALPHA = 1.0
A7_TW = 1.0  # T(RI) = TW, T(RO) = 0, T(r, 0) = 0

#: Sample times of the centreline history. The gap's diffusive time is
#: ``(RO-RI)^2/(pi^2 alpha) ~ 6e-3``, so these span "barely started" to
#: "practically steady" and the last one pins the steady state A1 also checks.
A7_TIMES = (0.002, 0.005, 0.010, 0.030)

#: Terms of the eigenfunction expansion. The n-th mode decays like
#: ``exp(-alpha lambda_n^2 t)`` with ``lambda_n ~ n pi/(RO-RI)``, so 40 terms
#: are far more than the earliest sample time needs.
A7_TERMS = 40


class _OutsideCylinder:
    """Body whose **solid** side is the exterior of a cylinder — A7's outer wall.

    The body protocol is duck-typed (``sdf``/``normal``, positive and outward
    into the fluid; see :mod:`blockamr.ibm.body`), so the complement of a
    cylinder needs no library change: negate both. Whether ``Cylinder`` should
    grow an ``invert=True`` — or the library an ``Annulus`` — is an API
    question this case raises rather than answers.
    """

    def __init__(self, centre, radius, axis):
        self._inner = Cylinder(centre=centre, radius=radius, axis=axis)

    def sdf(self, x, y, z):
        return -self._inner.sdf(x, y, z)

    def normal(self, x, y, z):
        return -np.asarray(self._inner.normal(x, y, z))


def _a7_eigenvalues():
    """Roots of ``C0(lambda RO) = 0``, the annulus eigenvalue condition.

    ``C0(lambda r) = J0(lambda r) Y0(lambda RI) - Y0(lambda r) J0(lambda RI)``
    vanishes at ``r = RI`` by construction; requiring it to vanish at ``RO``
    quantises lambda. The roots interlace with spacing approaching
    ``pi/(RO-RI)``, so a bracketing scan on a fraction of that spacing cannot
    step over one.
    """
    from scipy.optimize import brentq

    step = np.pi / (A7_RO - A7_RI) / 8.0
    roots, lam = [], step
    while len(roots) < A7_TERMS:
        if _a7_c0(lam, A7_RO) * _a7_c0(lam + step, A7_RO) < 0.0:
            roots.append(brentq(_a7_c0, lam, lam + step, args=(A7_RO,), xtol=1e-13))
        lam += step
    return np.array(roots)


def _a7_c0(lam, r):
    """The cross-product Bessel eigenfunction, zero at ``r = RI``."""
    from scipy.special import j0, y0

    return j0(lam * r) * y0(lam * A7_RI) - y0(lam * r) * j0(lam * A7_RI)


def _a7_steady(r):
    """``T = TW ln(r/RO) / ln(RI/RO)`` — the ``t -> inf`` limit (A1's ``ln r``)."""
    return A7_TW * np.log(r / A7_RO) / np.log(A7_RI / A7_RO)


def _a7_exact_history(r, times):
    """``T(r, t)`` of the annulus started from zero, by eigenfunction expansion.

    ``T = T_steady(r) + sum_n c_n C0(lambda_n r) exp(-alpha lambda_n^2 t)`` with
    the ``c_n`` fixed by ``T(r, 0) = 0``, i.e. by expanding ``-T_steady`` in the
    ``C0`` basis under the Sturm-Liouville weight ``r dr``. The inner products
    are taken on a fine uniform radial grid: the integrands are smooth and the
    highest mode has ~40 half-waves across the gap, which 20001 points resolve
    to far below the discretisation error this test is measuring.
    """
    grid = np.linspace(A7_RI, A7_RO, 20001)
    weight = grid * _a7_steady(grid)
    out = np.full(len(times), _a7_steady(r))
    for lam in _a7_eigenvalues():
        basis = _a7_c0(lam, grid)
        coeff = -np.trapezoid(weight * basis, grid) / np.trapezoid(grid * basis**2, grid)
        out = out + coeff * _a7_c0(lam, r) * np.exp(-A7_ALPHA * lam**2 * np.asarray(times))
    return out


def _a7_probe_radius(mesh, n, radius):
    """Mask of the fluid cells whose centre is within half a cell of ``radius``.

    The "centreline" of the annulus is a circle, not a point, and the mesh has
    no cell centred on it — so the history is read as the mean over the ring of
    cells straddling it. The mean is exact to ``O(dx^2)`` for a smooth radial
    field and is derived from the analytic geometry, not from the solver.
    """
    X, Y, _Z = _coords(mesh, (0, 0, 0), (n, n, NZ))
    r = np.hypot(X - A7_CENTRE[0], Y - A7_CENTRE[1])
    dx = float(mesh.geom(0).cell_size()[0])
    return np.abs(r - radius) < 0.5 * dx


def test_a7_transient_annulus_centreline_history_matches_the_bessel_series(blockamr_session):
    """A7 — the unsteady scalar analogue of A1, and the cheapest unsteady case.

    Both walls are immersed (inner at ``TW``, outer at 0), the fluid starts at
    zero, and the mid-radius history is compared with the closed-form Bessel
    series at four times spanning the transient. A steady test cannot see a
    wall condition that is applied a step late or damps the approach; this can,
    at the cost of one short run per mesh (the gap's diffusive time is ~6e-3,
    so the runs are far shorter than the steady cases above).

    Optional in §9.1 and release-tier in §10 — included because the transient
    is cheap once the annulus geometry exists, and because the second (outer)
    patch makes this the only case here with the fluid *between* two bodies.

    ``L-inf`` over the sampled history, with an observed order over the same
    refinement set as the steady cases.
    """
    pytest.importorskip("scipy")  # the eigenvalue oracle, not the code under test

    r_mid = 0.5 * (A7_RI + A7_RO)

    errors = []
    for n in RESOLUTIONS:
        mesh = _make_mesh(
            n,
            {
                "inner": Cylinder(centre=A7_CENTRE, radius=A7_RI, axis=A7_AXIS),
                "outer": _OutsideCylinder(centre=A7_CENTRE, radius=A7_RO, axis=A7_AXIS),
            },
        )
        T = CellField(
            mesh,
            ncomp=1,
            ngrow=1,
            name="T",
            ibm_bc={"inner": FixedValue(A7_TW), "outer": FixedValue(0.0)},
        )
        _seed(T, mesh, lambda X, Y, Z: np.zeros(np.shape(X) + (1,)))
        eqn = Equation(exp.ddt(T) - exp.laplacian(A7_ALPHA, T), schemes={"ddt": "Euler"})

        dx = float(mesh.geom(0).cell_size()[0])
        dt = dx * dx / (DT_SAFETY * A7_ALPHA)
        ring = _a7_probe_radius(mesh, n, r_mid)
        history, stamps, t = [], [], 0.0
        for target in A7_TIMES:
            while t < target - 0.5 * dt:
                solve(eqn, dt=dt, t=t, solution={"ibm": "ghostCell", "backend": BACKEND})
                t += dt
            history.append(float(_assemble(T, n)[..., 0][ring].mean()))
            stamps.append(t)  # the step lands within dt/2 of the target
        # The series is evaluated at the times actually reached, so the O(dt)
        # offset of the nearest step is not charged to the wall treatment.
        exact = _a7_exact_history(r_mid, stamps)
        errors.append(float(np.abs(np.array(history) - exact).max()))

    order = _observed_order(errors)
    assert order > MIN_ORDER, _report("centreline history Linf", errors, order)
