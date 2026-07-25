# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""The IBM diagnostic ladder — rungs 1-5, the scheme x method grid, error paths.

Companion to ``test_ibm_laplacian.py`` (the MMS convergence suite); this file
is the *ordered* suite of `plans/IBM/ibm-verification-plan.md` §2: each rung
adds exactly one demand over the rung above, so **the first failing rung names
the defect**. Red by design until the IBM path lands.

    1  laplacian(C), no body                -> 0
    2  + body far outside the domain        -> bitwise equal to rung 1
    3  + body inside, FixedValue(C)         -> exactly 0
    4  + FixedGradient(0)                   -> exactly 0
    5  plane body, T linear in the normal   -> exactly 0

Everything here is *exact*: literals, analytic formulae or parametrize data,
no tolerance to argue about and no resolution needed (verification plan §3).
Region masks are computed test-side from the analytic body — with no access to
the implementation's classification that is an independent oracle, not
duplication (§10).

The whole file is the "every commit" tier: tiny meshes, < 60 s.
"""

import numpy as np
import pytest

import blockamr
from blockamr.dsl import Equation, evaluate, exp
from blockamr.field import CellField, FaceField
from blockamr.ibm import Cylinder, FixedGradient, FixedValue, Mixed, Plane
from blockamr.mesh import Mesh
from blockamr.operators.div import update_face_fluxes
from blockamr.schemes.registry import SCHEME_REGISTRY

# The backend every rung runs on (verification plan §3 spells "cpp"
# throughout). `jax` parity is a pre-merge tier concern, not a rung.
BACKEND = "cpp"

N = 32  # the exact tests need no resolution
NZ = 4  # thin in the cylinder axis direction; every probe is z-invariant

CONST = 3.0  # the constant field value of rungs 1, 3 and 4

# MMS quadratic of test_ibm_laplacian, reused by the rotation probe:
# T(r) = A + B*(r^2 - R^2)  ->  laplacian(T) = 4B exactly, T|_R = A.
A_MMS = 0.3
B_MMS = 0.5
R = 0.2
CENTRE = (0.5, 0.5)
AXIS = 2

# Rung 5: T = A_LIN + B_LIN * (x - X_WALL), constant (= A_LIN) on the plane.
X_WALL = 0.5
A_LIN = 0.0
B_LIN = 2.0

# verification plan §8: the sentence a step method owes an operator-level call.
STEP_METHOD_MSG = "does not support operator-level evaluation"

METHODS = ["noIbm", "directForcing", "ghostCell"]

# Generated from the registry so a new scheme cannot be added without
# entering the grid (verification plan §5). ``ddt`` is a time scheme — it has
# no operator-level ``evaluate`` — so it is not part of a spatial grid.
SCHEMES = [
    (op, name) for op, table in sorted(SCHEME_REGISTRY.items()) if op != "ddt" for name in table
]
SCHEME_IDS = [f"{op}-{name}" for op, name in SCHEMES]

# Operator -> the key its scheme is looked up under in ``Equation(schemes=...)``
# (``lookup_scheme`` tries the term's scheme_key then its class name).
SCHEME_DICT_KEY = {"laplacian": "Laplacian", "div": "Div", "grad": "Grad"}


# ---------------------------------------------------------------------------
# Helpers — mesh/field construction, analytic fills, result extraction
# ---------------------------------------------------------------------------


def _cylinder(centre=CENTRE):
    return {"cyl": Cylinder(centre=centre, radius=R, axis=AXIS)}


def _make_mesh(n=N, nz=NZ, max_size=None, bodies=None, periodic=(1, 1, 1)):
    """Mesh on the unit cube, ``n x n x nz`` cells, periodic by default.

    ``bodies`` is the patch-keyed immersed geometry dict; ``max_size`` picks
    the box decomposition (``None`` -> a single box). ``periodic`` is opened up
    for rung 5: a half-space body in a periodic box wraps its solid slab around
    the domain edge and manufactures a sub-cell gap there, which is a property
    of the test mesh, not of the method.
    """
    box = blockamr.Box([0, 0, 0], [n - 1, n - 1, nz - 1])
    rb = blockamr.RealBox([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    geom = blockamr.Geometry(box, rb, 0, list(periodic))
    ba = blockamr.BoxArray(box)
    ba.max_size(max(n, nz) if max_size is None else max_size)
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
    """Fill ``field`` from the analytic ``func(X, Y, Z)`` over the whole domain.

    Solid cells are seeded too: the IBM must reconstruct its near-surface
    stencil from its own BC, never lean on values found in the body.
    """
    mf = field.mf[0]
    for mfi in blockamr.MFIterator(mf):
        arr = mf.copy_to_host(mfi)
        X, Y, Z = _coords(mesh, mfi.valid_box().small_end(), arr.shape[:3])
        arr[:, :, :, 0] = func(X, Y, Z)
        mf.copy_from(mfi, arr)
    field.fill_patch(0, 0.0)


def _fill_halo(field, mesh, func):
    """Seed the *ghost* cells from the same analytic ``func``, after
    ``fill_patch``.

    Only needed on a non-periodic mesh: ``fill_boundary`` fills inter-box and
    periodic halos but leaves domain-exterior ghosts untouched, and an
    unfilled halo would contaminate every edge cell for reasons that have
    nothing to do with the IBM. Filling them analytically is an exact Dirichlet
    halo, and it lets the assertion cover the whole domain instead of an
    eroded interior.
    """
    mf = field.mf[0]
    ng = mf.n_grow()
    for mfi in blockamr.MFIterator(mf):
        arr = mf.copy_grown_to_host(mfi)
        lo = [c - ng for c in mfi.valid_box().small_end()]
        X, Y, Z = _coords(mesh, lo, arr.shape[:3])
        arr[:, :, :, 0] = func(X, Y, Z)
        mf.copy_grown_from(mfi, arr)


def _constant_field(mesh, value=CONST, ncomp=1, ngrow=1, ibm_bc=None):
    T = CellField(mesh, ncomp=ncomp, ngrow=ngrow, name="T", ibm_bc=ibm_bc or {})
    _fill(T, mesh, lambda X, Y, Z: np.full(X.shape, value))
    return T


def _flat(results):
    """All valid cells of level 0 as one flat array."""
    return np.concatenate([np.asarray(a).ravel() for a in results[0]])


def _assemble(field, results, n=N, nz=NZ):
    """Stitch the per-box level-0 result into one global ``(n, n, nz)`` array."""
    out = np.full((n, n, nz), np.nan)
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


def _valid_cells(field):
    """Bitwise snapshot of a field's valid cells, per box."""
    mf = field.mf[0]
    return [np.array(mf.copy_to_host(mfi), copy=True) for mfi in blockamr.MFIterator(mf)]


def _unit_velocity(x, y, z, t):
    return np.ones_like(x), np.ones_like(x), np.ones_like(x)


def _uniform_flux(mesh, ngrow):
    """Face flux field for the uniform, divergence-free velocity (1, 1, 1)."""
    ff = FaceField(mesh, ncomp=1, ngrow=ngrow, name="phi")
    update_face_fluxes(ff[0], _unit_velocity, mesh.geom(0), t=0.0)
    return ff


def _sol(method=None, backend=BACKEND):
    """The fvSolution block: no ``"ibm"`` key at all means no IBM."""
    return {"backend": backend} if method is None else {"ibm": method, "backend": backend}


# ---------------------------------------------------------------------------
# Rung 1 — the operator alone
# ---------------------------------------------------------------------------


def test_laplacian_of_a_constant_is_zero_without_a_body(blockamr_session):
    """Rung 1. If this fails, nothing below it means anything: the operator is
    broken without any IBM, so every lower rung would have two suspects."""
    mesh = _make_mesh()
    T = _constant_field(mesh)
    out = evaluate(Equation(exp.laplacian(1.0, T)), t=0.0, solution=_sol())
    np.testing.assert_allclose(_flat(out), 0.0, atol=1e-13)


# ---------------------------------------------------------------------------
# Rung 2 — degeneration: an empty band costs nothing and changes nothing
# ---------------------------------------------------------------------------


def test_body_outside_the_domain_is_bitwise_identical_to_no_ibm(blockamr_session):
    """Rung 2. A body with an empty band must not perturb the plain operator.

    ``assert_array_equal``, not ``allclose`` — a tolerance here would silently
    permit the IBM path to couple into cells it does not own (verification plan
    §10 anti-patterns). The ``"noIbm"`` method is held to the same bar: it is
    the degenerate member of the method axis, so it must reproduce the no-key
    path bit for bit.
    """
    mesh = _make_mesh(bodies=_cylinder(centre=(99.0, 99.0)))
    T = _constant_field(mesh, ibm_bc={"cyl": FixedValue(0.0)})
    eqn = Equation(exp.laplacian(1.0, T))

    plain = _flat(evaluate(eqn, t=0.0, solution=_sol()))
    far = _flat(evaluate(eqn, t=0.0, solution=_sol("ghostCell")))
    none = _flat(evaluate(eqn, t=0.0, solution=_sol("noIbm")))

    np.testing.assert_array_equal(far, plain)
    np.testing.assert_array_equal(none, plain)


# ---------------------------------------------------------------------------
# Rungs 3-4 — row consistency and the Neumann branch
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bc",
    [
        FixedValue(CONST),
        FixedGradient(0.0),
        Mixed(value=CONST, gradient=0.0, fraction=1.0),
    ],
    ids=["dirichlet", "neumann-zero", "mixed-as-dirichlet"],
)
def test_constant_field_is_annihilated_for_every_bc_type(blockamr_session, bc):
    """Rungs 3-4. The equation-level form of the row-consistency identity
    (``sum(w) + b*alpha == 1``): a constant field consistent with its wall BC
    must give an exactly zero laplacian, in the band as well as the bulk.

    Rung 3 is the Dirichlet datum, rung 4 the Neumann one; ``Mixed`` at
    ``fraction=1`` is the Dirichlet limit and must agree with rung 3.
    """
    mesh = _make_mesh(bodies=_cylinder())
    T = _constant_field(mesh, ibm_bc={"cyl": bc})
    out = evaluate(Equation(exp.laplacian(1.0, T)), t=0.0, solution=_sol("ghostCell"))
    np.testing.assert_allclose(_flat(out), 0.0, atol=1e-12)


# ---------------------------------------------------------------------------
# Rung 5 — linear exactness of the reconstruction
# ---------------------------------------------------------------------------


def test_linear_field_normal_to_a_plane_wall_is_reproduced_exactly(blockamr_session):
    """Rung 5 — the sharpest exact test available.

    ``T = a + b*(x - x_w)`` is linear, so a linear-exact reconstruction must
    give a machine-zero laplacian everywhere, band included.

    A **plane**, not a cylinder, because the scalar ``FixedValue`` datum can
    only express a wall trace that is *constant* on the surface. On a plane a
    linear field has exactly that. On a cylinder it would force a radial field
    (``r^2``, ``ln r``) — and those are not linear, so they cannot isolate
    first-order reconstruction error at all (verification plan §1).

    The mesh is non-periodic in x: a linear field is not periodic, and a
    half-space body in a periodic box would also wrap its solid slab back
    across the far edge. The two x-edge columns therefore carry an unfilled
    halo for reasons that have nothing to do with the IBM, and are excluded.
    y and z need no exclusion — T is invariant along both, so their periodic
    halos are exact.
    """
    mesh = _make_mesh(
        bodies={"wall": Plane(point=(X_WALL, 0.0, 0.0), normal=(1.0, 0.0, 0.0))},
        periodic=(0, 1, 1),
    )
    T = CellField(mesh, ncomp=1, ngrow=1, name="T", ibm_bc={"wall": FixedValue(A_LIN)})
    _fill(T, mesh, lambda X, Y, Z: A_LIN + B_LIN * (X - X_WALL))

    out = evaluate(Equation(exp.laplacian(1.0, T)), t=0.0, solution=_sol("ghostCell"))
    lap = _assemble(T, out)
    np.testing.assert_allclose(lap[1:-1, :, :], 0.0, atol=1e-12)


def test_linear_field_normal_to_a_tilted_plane_wall_is_reproduced_exactly(blockamr_session):
    """Rung 5, tilted — the version that can actually fail.

    A grid-aligned wall is a degenerate probe: the y/z trilinear fractions are
    exactly 0, the 8-donor stencil collapses to a 1-D 2-point stencil, and no
    donor is ever non-fluid — so the Invariant-D ladder never runs and the test
    above cannot see a first-order fallback in it. Tilting the normal to
    (1, 2, 3) makes every row a genuine 3-D interpolation, and a reconstruction
    that is only linear-*ish* shows up immediately as an O(dx) residual.

    The mesh is non-periodic in all three directions (a half-space is not
    periodic in any of them) and the ghost band is seeded analytically, so the
    assertion covers every cell rather than an eroded interior.
    """
    n_hat = np.array([1.0, 2.0, 3.0])
    n_hat /= np.linalg.norm(n_hat)
    point = (0.5, 0.5, 0.5)
    mesh = _make_mesh(
        n=24,
        nz=24,
        bodies={"wall": Plane(point=point, normal=tuple(n_hat))},
        periodic=(0, 0, 0),
    )

    def exact(X, Y, Z):
        return A_LIN + B_LIN * (
            (X - point[0]) * n_hat[0] + (Y - point[1]) * n_hat[1] + (Z - point[2]) * n_hat[2]
        )

    T = CellField(mesh, ncomp=1, ngrow=1, name="T", ibm_bc={"wall": FixedValue(A_LIN)})
    _fill(T, mesh, exact)
    _fill_halo(T, mesh, exact)

    out = evaluate(Equation(exp.laplacian(1.0, T)), t=0.0, solution=_sol("ghostCell"))
    lap = _assemble(T, out, n=24, nz=24)
    np.testing.assert_allclose(lap, 0.0, atol=1e-12)


# ---------------------------------------------------------------------------
# Still rung 5 tier, still exact: purity, determinism, decomposition, symmetry
# ---------------------------------------------------------------------------


def _mms(centre):
    """T(r) = A + B*(r^2 - R^2) about *centre*; laplacian(T) = 4B exactly."""

    def func(X, Y, Z):
        r2 = (X - centre[0]) ** 2 + (Y - centre[1]) ** 2
        return A_MMS + B_MMS * (r2 - R**2)

    return func


def _mms_case(max_size=None, centre=CENTRE, n=N):
    mesh = _make_mesh(n=n, max_size=max_size, bodies=_cylinder(centre=centre))
    T = CellField(mesh, ncomp=1, ngrow=1, name="T", ibm_bc={"cyl": FixedValue(A_MMS)})
    _fill(T, mesh, _mms(centre))
    return mesh, T, Equation(exp.laplacian(1.0, T))


def test_evaluate_does_not_mutate_the_input_field(blockamr_session):
    """Rung 5 tier. ``evaluate`` computes a source term; it must never write
    the reconstruction back into T — not even into the solid cells, which are
    valid cells of the field and the natural place for a ghost value to leak."""
    _mesh, T, eqn = _mms_case()
    before = _valid_cells(T)
    evaluate(eqn, t=0.0, solution=_sol("ghostCell"))
    after = _valid_cells(T)

    assert len(before) == len(after)
    for bi, (b, a) in enumerate(zip(before, after)):
        np.testing.assert_array_equal(a, b, err_msg=f"box {bi}: evaluate mutated T")


def test_repeated_evaluate_is_bitwise_reproducible(blockamr_session):
    """Rung 5 tier. Determinism: no race, no dependence on evaluation order,
    no state carried between calls. Bitwise, never allclose."""
    _mesh, _T, eqn = _mms_case()
    first = _flat(evaluate(eqn, t=0.0, solution=_sol("ghostCell")))
    second = _flat(evaluate(eqn, t=0.0, solution=_sol("ghostCell")))
    np.testing.assert_array_equal(second, first)


@pytest.mark.parametrize("max_size", [16, 32], ids=["4-boxes", "1-box"])
def test_result_is_independent_of_box_decomposition(blockamr_session, max_size):
    """Rung 5 tier. With no access to the wall table this is the **only** way
    to see a donor that reached across a box boundary incorrectly: the same
    equation, decomposed differently, must give bitwise identical cells.

    The reference (one box, no interior box boundaries at all) is recomputed
    per parametrization so the case is self-contained.
    """
    _m_ref, T_ref, eqn_ref = _mms_case(max_size=N)
    reference = _assemble(T_ref, evaluate(eqn_ref, t=0.0, solution=_sol("ghostCell")))

    _m, T, eqn = _mms_case(max_size=max_size)
    split = _assemble(T, evaluate(eqn, t=0.0, solution=_sol("ghostCell")))

    np.testing.assert_array_equal(split, reference)


def test_result_rotates_with_the_body(blockamr_session):
    """Rung 5 tier. Rotating body + field by 90 degrees about the domain axis
    must rotate the result and nothing else — an off-centre body, so the
    rotation is not the identity.

    The map is ``(x, y) -> (1 - y, x)``, i.e. cell ``(i, j) -> (n-1-j, i)``,
    which is exactly ``np.rot90(..., axes=(0, 1))``. The body centre
    ``(0.3, 0.5)`` goes to ``(0.5, 0.3)``. Every coordinate involved is a
    dyadic rational on this grid, so the rotated field is the rotated field
    exactly and any difference is anisotropy in the wall treatment.
    """
    base_centre = (0.3, 0.5)
    rot_centre = (0.5, 0.3)

    _m0, T0, eqn0 = _mms_case(centre=base_centre)
    base = _assemble(T0, evaluate(eqn0, t=0.0, solution=_sol("ghostCell")))

    _m1, T1, eqn1 = _mms_case(centre=rot_centre)
    rotated = _assemble(T1, evaluate(eqn1, t=0.0, solution=_sol("ghostCell")))

    np.testing.assert_allclose(rotated, np.rot90(base, k=1, axes=(0, 1)), rtol=1e-12, atol=1e-12)


# ---------------------------------------------------------------------------
# Rung 8 — div, with a nonzero tangential velocity at the wall
# ---------------------------------------------------------------------------

# Rigid rotation about the cylinder axis at OMEGA = 1/R, so the tangential
# speed on the surface is exactly |u| = OMEGA*R = 1: the first rung where the
# wall sees an O(1) velocity rather than none at all.
OMEGA = 1.0 / R


def _rotation_velocity(x, y, z, t):
    """``u = omega x r`` about the cylinder axis — solid-body rotation.

    Divergence-free analytically *and* discretely: ``u_x`` does not depend on
    x and ``u_y`` does not depend on y, so both face differences vanish
    identically, whatever the mesh.
    """
    return -OMEGA * (y - CENTRE[1]), OMEGA * (x - CENTRE[0]), np.zeros_like(x)


def _rotation_flux(mesh, ngrow):
    """Face flux field for the rigid rotation, built the way every other div
    test in this suite builds one."""
    ff = FaceField(mesh, ncomp=1, ngrow=ngrow, name="phi")
    update_face_fluxes(ff[0], _rotation_velocity, mesh.geom(0), t=0.0)
    return ff


def test_div_of_a_radial_scalar_on_a_tangential_flux_is_zero(blockamr_session):
    """Rung 8 — the div counterpart of rungs 3-6, and exact.

    ``u = omega x r`` is divergence-free and exactly tangential to a concentric
    cylinder, and ``T(r) = A + B(r^2 - R^2)`` is radial, so
    ``div(u T) = u . grad T == 0`` identically. The whole result is therefore
    error and there is nothing to subtract off — no analytic post-processing,
    no resolution study.

    It is exact discretely too, on the *linear* (central) flux interpolation:
    ``u_x`` is x-independent so the x-flux difference collapses to
    ``u_x (T_{i+1} - T_{i-1}) / 2dx = -omega*Y * 2X``, the y-difference to
    ``+omega*X * 2Y``, and the two cancel to the last bit. ``upwind`` would not
    — its O(dx) dissipation is a scheme property, not an IBM defect — so the
    scheme is named explicitly rather than left at the ``Div`` default.

    What this adds over every laplacian rung: the wall sees a **nonzero
    tangential velocity** (|u| = OMEGA*R = 1 on the surface). A wall treatment
    that couples into the face values — as opposed to the cell reconstruction
    the rungs above probe — can only show up here.

    The mesh is non-periodic and the ghost band is seeded analytically: neither
    ``T`` nor ``u`` is periodic, and a wrapped halo would contaminate the
    domain-edge cells for reasons that have nothing to do with the IBM.
    """
    mesh = _make_mesh(bodies=_cylinder(), periodic=(0, 0, 0))
    T = CellField(mesh, ncomp=1, ngrow=1, name="T", ibm_bc={"cyl": FixedValue(A_MMS)})
    _fill(T, mesh, _mms(CENTRE))
    _fill_halo(T, mesh, _mms(CENTRE))

    eqn = Equation(exp.div(_rotation_flux(mesh, T.ngrow), T), schemes={"Div": "linear"})
    out = evaluate(eqn, t=0.0, solution=_sol("ghostCell"))

    np.testing.assert_allclose(_assemble(T, out), 0.0, atol=1e-12)


# ---------------------------------------------------------------------------
# The scheme x method grid (verification plan §5)
# ---------------------------------------------------------------------------


def _grid_equation(op, scheme, mesh, T):
    """``Equation`` for one (operator, scheme) cell of the grid.

    The scheme is chosen the way this repo chooses schemes — a name in the
    equation's ``schemes`` dict, resolved through ``SCHEME_REGISTRY``.
    """
    if op == "laplacian":
        term = exp.laplacian(1.0, T)
    elif op == "div":
        term = exp.div(_uniform_flux(mesh, T.ngrow), T)
    elif op == "grad":
        term = exp.grad(T)
    else:  # pragma: no cover - the grid is generated, this is a guard
        raise AssertionError(f"grid has no builder for operator {op!r}")
    return Equation(term, schemes={SCHEME_DICT_KEY[op]: scheme})


@pytest.mark.parametrize("op, scheme", SCHEMES, ids=SCHEME_IDS)
@pytest.mark.parametrize("method", METHODS)
def test_every_scheme_runs_under_every_method_and_annihilates_a_constant(
    blockamr_session, op, scheme, method
):
    """The combinatorial smoke test, at equation level: its job is dispatch
    coverage, not numerics (verification plan §5).

    A constant field consistent with its wall BC is annihilated by every one of
    them: ``laplacian(C) = 0``, ``div(u C) = C div(u) = 0`` on a uniform flux,
    ``grad(C) = 0`` — including in the band, where the wall reconstruction
    reproduces C.

    ``directForcing`` is a *step* method: it restricts the field between steps
    and has no operator-level form, so the expectation for that column is the
    §8 refusal sentence, not a number.
    """
    mesh = _make_mesh(n=16, bodies=_cylinder())
    T = _constant_field(mesh, ngrow=2, ibm_bc={"cyl": FixedValue(CONST)})
    eqn = _grid_equation(op, scheme, mesh, T)
    solution = _sol(method)

    if method == "directForcing":
        with pytest.raises((ValueError, NotImplementedError), match=STEP_METHOD_MSG):
            evaluate(eqn, t=0.0, solution=solution)
        return

    out = evaluate(eqn, t=0.0, solution=solution)
    assert isinstance(eqn.spatial_ops[0].scheme, SCHEME_REGISTRY[op][scheme])
    np.testing.assert_allclose(_flat(out), 0.0, atol=1e-12)


# ---------------------------------------------------------------------------
# D1 — a scheme wider than the reconstructed depth degrades to width 1 in the
# band, and only there (verification plan §4, §5)
# ---------------------------------------------------------------------------

# ``ghostCell`` reconstructs exactly one solid layer (the cells with a fluid
# face-neighbour); every deeper solid cell carries a ``b = 0, ndonor = 0`` row
# and is pinned to zero. A scheme whose stencil is wider than that depth reads
# one of those pinned cells from the band — a number that is not a
# reconstruction of anything.
RECONSTRUCTED_SOLID_DEPTH = 1

# Split the div schemes by their own declared ``stencil_width`` rather than by a
# hand-written name list, so a new wide scheme joins these tests by existing
# (verification plan §10: both axes enter as parametrize data).
WIDE_DIV_SCHEMES = sorted(
    name for name, cls in SCHEME_REGISTRY["div"].items() if cls().stencil_width > 1
)
NARROW_DIV_SCHEMES = sorted(
    name for name, cls in SCHEME_REGISTRY["div"].items() if cls().stencil_width == 1
)


def _plane_linear_case(scheme, n=N, nz=NZ):
    """``div(u T)`` with ``u = (1,1,1)`` and ``T`` linear along a plane normal.

    The rung-5 geometry (a plane so the surface trace is a constant a scalar
    ``FixedValue`` can express), carried over to ``div``. ``ngrow=2`` because
    the wide schemes ask for it, and the x halo is seeded analytically because
    a linear field is not periodic.
    """
    mesh = _make_mesh(
        n=n,
        nz=nz,
        bodies={"wall": Plane(point=(X_WALL, 0.0, 0.0), normal=(1.0, 0.0, 0.0))},
        periodic=(0, 1, 1),
    )
    T = CellField(mesh, ncomp=1, ngrow=2, name="T", ibm_bc={"wall": FixedValue(A_LIN)})

    def exact(X, Y, Z):
        return A_LIN + B_LIN * (X - X_WALL)

    _fill(T, mesh, exact)
    _fill_halo(T, mesh, exact)
    eqn = Equation(exp.div(_uniform_flux(mesh, T.ngrow), T), schemes={"Div": scheme})
    return mesh, T, eqn


def _fluid_of_the_plane(n=N, nz=NZ):
    """Cells whose centre is on the fluid side of the ``X_WALL`` plane.

    Analytic, from the body — an independent oracle, never the implementation's
    own classification (verification plan §10).
    """
    xs = (np.arange(n) + 0.5) / n
    X, _Y, _Z = np.meshgrid(xs, np.arange(n), np.arange(nz), indexing="ij")
    return X > X_WALL


@pytest.mark.parametrize("scheme", NARROW_DIV_SCHEMES + WIDE_DIV_SCHEMES)
def test_every_div_scheme_is_exact_on_a_linear_field_at_a_plane_wall(blockamr_session, scheme):
    """D1, the sharp form. ``u = (1,1,1)`` is divergence-free and ``T`` is linear,
    so ``div(u T) = u . grad T = B_LIN`` exactly — and *every* div scheme in the
    registry reproduces a linear field exactly on a uniform flux (``upwind``
    included: its dissipation is proportional to the second derivative, which is
    zero here). So this is an exact probe with no tolerance to argue about, and
    it holds for a width-1 and a width-2 scheme alike.

    It holds in the **band** only if the wide schemes degrade there. ``ghostCell``
    reconstructs ``RECONSTRUCTED_SOLID_DEPTH == 1`` solid layer; a width-2 stencil
    reaches the second layer, which is pinned to zero, and the band result is then
    not the divergence of anything. The decision this test encodes: such a scheme
    must fall back to a width-1 scheme *for the band cells only*.

    **Red pending that fallback** for every entry of ``WIDE_DIV_SCHEMES``; green
    today for ``NARROW_DIV_SCHEMES``, which are in the parametrization precisely
    so the failure reads as "the wide schemes, and only the wide schemes".
    ``vanLeer`` matters more here, not less: on a *constant* field its limiter
    degenerates at the wall and hides the defect entirely, so the constant probe
    in the scheme x method grid passes for it. A linear field is the coarsest
    probe that sees through the limiter.

    The assertion covers the whole fluid region: the plane's x halo is seeded
    analytically, so no domain-edge column has to be eroded away.
    """
    _mesh, T, eqn = _plane_linear_case(scheme)
    out = _assemble(T, evaluate(eqn, t=0.0, solution=_sol("ghostCell")))
    fluid = _fluid_of_the_plane()
    np.testing.assert_allclose(out[fluid], B_LIN, atol=1e-12)


def _stencil_entirely_in_fluid(width, n=N, nz=NZ, centre=CENTRE):
    """Cells whose full ``width``-wide cross stencil lies in the fluid.

    Computed test-side from the analytic cylinder — with no access to the
    implementation's classification that is an independent oracle, and the plan
    (§4, §10) prefers it to asking the code which cells it believes are near the
    wall. ``np.roll`` wraps, which is harmless: the body sits well inside the
    domain, so every cell a wrap brings in is fluid anyway.
    """
    xs = (np.arange(n) + 0.5) / n
    zs = (np.arange(nz) + 0.5) / nz
    X, Y, _Z = np.meshgrid(xs, xs, zs, indexing="ij")
    fluid = np.hypot(X - centre[0], Y - centre[1]) > R
    mask = fluid.copy()
    for d in range(3):
        for s in range(1, width + 1):
            mask &= np.roll(fluid, s, axis=d) & np.roll(fluid, -s, axis=d)
    return mask


@pytest.mark.parametrize("scheme", WIDE_DIV_SCHEMES)
def test_degrading_a_wide_scheme_in_the_band_leaves_the_bulk_bitwise_unchanged(
    blockamr_session, scheme
):
    """D1, the other half — and the half that forbids the lazy fix.

    The fallback of the test above is confined to the band. A cell whose full
    width-2 stencil lies entirely in the fluid never reads a reconstructed or a
    pinned value, so under ``ghostCell`` it must produce the number the plain
    operator produces — **bitwise**, not to a tolerance, because a tolerance
    would permit exactly the coupling this forbids (verification plan §10).

    Degrading the whole field to width 1 would satisfy the constant probe and the
    linear probe above and fail here, which is the point: the bulk keeps the full
    wide scheme.

    The field is deliberately **non-constant** (the MMS quadratic) — a constant
    is annihilated by every scheme at every width, so it cannot tell a degraded
    stencil from an intact one. ``noIbm`` is the reference rather than the no-key
    path because rung 2 already pins those two together.

    One field, two evaluations: ``evaluate`` does not mutate its input (asserted
    at the rung-5 tier), so "the same data" is literally the same data.

    Green today and it must stay green — this is the constraint on the fix, not
    a prediction about it.
    """
    mesh = _make_mesh(bodies=_cylinder(), periodic=(0, 0, 0))
    T = CellField(mesh, ncomp=1, ngrow=2, name="T", ibm_bc={"cyl": FixedValue(A_MMS)})
    _fill(T, mesh, _mms(CENTRE))
    _fill_halo(T, mesh, _mms(CENTRE))
    eqn = Equation(exp.div(_uniform_flux(mesh, T.ngrow), T), schemes={"Div": scheme})

    with_ibm = _assemble(T, evaluate(eqn, t=0.0, solution=_sol("ghostCell")))
    without = _assemble(T, evaluate(eqn, t=0.0, solution=_sol("noIbm")))

    bulk = _stencil_entirely_in_fluid(SCHEME_REGISTRY["div"][scheme]().stencil_width)
    np.testing.assert_array_equal(with_ibm[bulk], without[bulk])


# ---------------------------------------------------------------------------
# Error paths (verification plan §8) — a sentence, never an AttributeError
# ---------------------------------------------------------------------------


def _laplacian_case(bodies=None, ibm_bc=None):
    mesh = _make_mesh(n=16, bodies=_cylinder() if bodies is None else bodies)
    T = _constant_field(mesh, ibm_bc=ibm_bc if ibm_bc is not None else {"cyl": FixedValue(CONST)})
    return Equation(exp.laplacian(1.0, T))


def test_unknown_ibm_name_names_the_bad_name_and_the_valid_list(blockamr_session):
    """§8. The message must carry both halves: what was asked for, and what
    could have been asked for. A bare 'unknown method' sends the reader to the
    source."""
    eqn = _laplacian_case()
    with pytest.raises(ValueError) as excinfo:
        evaluate(eqn, t=0.0, solution=_sol("noSuchMethod"))

    msg = str(excinfo.value)
    assert "noSuchMethod" in msg
    for name in METHODS:
        assert name in msg, f"valid method {name!r} missing from: {msg}"


@pytest.mark.parametrize(
    "ibm_bc, offending",
    [
        ({}, "cyl"),  # a body with no matching ibm_bc entry
        ({"cyl": FixedValue(CONST), "ghost": FixedValue(1.0)}, "ghost"),  # and the converse
    ],
    ids=["missing-bc-for-body", "bc-without-body"],
)
def test_ibm_bc_keys_that_do_not_match_bodies_name_the_offending_patch(
    blockamr_session, ibm_bc, offending
):
    """§8. ``ibm_bc`` and ``mesh.bodies`` are patch-keyed dicts over the same
    key set; a key in one and not the other is an error that names the key."""
    eqn = _laplacian_case(ibm_bc=ibm_bc)
    with pytest.raises(ValueError) as excinfo:
        evaluate(eqn, t=0.0, solution=_sol("ghostCell"))
    assert offending in str(excinfo.value)


def test_ibm_requested_with_no_bodies_says_bodies_is_empty(blockamr_session):
    """§8. Asking for IBM on a mesh with nothing immersed is a configuration
    error, and the message must point at ``mesh.bodies`` being empty rather
    than failing later on an empty band."""
    eqn = _laplacian_case(bodies={}, ibm_bc={})
    with pytest.raises(ValueError) as excinfo:
        evaluate(eqn, t=0.0, solution=_sol("ghostCell"))

    msg = str(excinfo.value)
    assert "mesh.bodies" in msg
    assert "empty" in msg.lower()


def test_step_method_rejected_for_operator_evaluation(blockamr_session):
    """§8. ``directForcing`` restricts the field between steps; it has no
    operator-level form. The refusal must say so by name, in a sentence."""
    eqn = _laplacian_case()
    with pytest.raises((ValueError, NotImplementedError)) as excinfo:
        evaluate(eqn, t=0.0, solution=_sol("directForcing"))

    msg = str(excinfo.value)
    assert "directForcing" in msg
    assert STEP_METHOD_MSG in msg
    assert "step method" in msg


def test_deferred_method_refuses_execution(blockamr_session):
    """§8. ``cutCell`` validates the fvSolution schema — the name is legal —
    and then refuses to run, naming itself. A missing capability fails with a
    sentence, not an AttributeError."""
    eqn = _laplacian_case()
    with pytest.raises((ValueError, NotImplementedError)) as excinfo:
        evaluate(eqn, t=0.0, solution=_sol("cutCell"))

    msg = str(excinfo.value)
    assert "cutCell" in msg
    assert "not implemented" in msg.lower() or "refus" in msg.lower()
