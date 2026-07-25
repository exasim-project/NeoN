# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""Test-first IBM (immersed boundary) explicit laplacian for blockamr.

This suite *defines* the target API — it is red by design (nothing is
implemented yet). Every test body spells out the agreed signature inline:

    mesh.bodies = {"cyl": Cylinder(centre=(0.5, 0.5), radius=0.2, axis=2)}
    T = CellField(mesh, ncomp=1, ngrow=1, name="T",
                  ibm_bc={"cyl": FixedValue(1.0)})
    lap = evaluate(Equation(exp.laplacian(1.0, T)),
                   t=0.0, solution={"ibm": "ghostCell"})

Both the geometry (``mesh.bodies``) and the surface BC (``ibm_bc``) are
patch-keyed dicts matched by name — this is what makes **more than one IBM
patch** expressible (OpenFOAM ``boundaryField`` semantics). The MMS uses a
single patch ``"cyl"``; ``test_two_patches_independent`` exercises two.

MMS: ``T(r) = a + b*(r**2 - R**2)`` with ``r`` the cylindrical radius about the
body axis. Then ``laplacian(T) = 4b`` exactly, ``T|_R = a`` and
``dT/dr|_R = 2bR``. The 7-point stencil is exact on quadratics, so bulk fluid
cells must be machine-precision exact (the IBM must not contaminate regular
cells); near-surface cells only need their error to shrink under refinement.
"""

import numpy as np
import pytest

import blockamr
from blockamr.dsl import Equation, evaluate, exp
from blockamr.field import CellField
from blockamr.ibm import Cylinder, FixedValue, FixedGradient, Mixed
from blockamr.mesh import Mesh

# MMS parameters: surface value a, curvature b -> laplacian(T) = 4b exactly.
A = 0.3
B = 0.5
LAP_EXACT = 4.0 * B  # = 2.0
R = 0.2
CENTRE = (0.5, 0.5)
AXIS = 2
NZ = 4  # thin in the axis direction; T is z-invariant so dz is irrelevant
GRAD_DATUM = 2.0 * B * R  # dT/dr|_R for the FixedGradient form of the same MMS


# ---------------------------------------------------------------------------
# Helpers (mesh building, MMS fill, result extraction, region masks)
# ---------------------------------------------------------------------------


def _default_bodies():
    return {"cyl": Cylinder(centre=CENTRE, radius=R, axis=AXIS)}


def _make_mesh(n, nz=NZ, bodies=None):
    """Periodic single-box Mesh on the unit cube, ``n x n x nz`` cells.

    ``mesh.bodies`` is the patch-keyed geometry dict (default one cylinder
    ``"cyl"``); ``max_size`` spans the domain so there is exactly one box.
    """
    box = blockamr.Box([0, 0, 0], [n - 1, n - 1, nz - 1])
    rb = blockamr.RealBox([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    geom = blockamr.Geometry(box, rb, 0, [1, 1, 1])
    ba = blockamr.BoxArray(box)
    ba.max_size(max(n, nz))
    dm = blockamr.DistributionMapping(ba)
    mesh = Mesh(ba, dm, geom)
    mesh.bodies = _default_bodies() if bodies is None else bodies
    return mesh


def _cell_centres(mesh, n, nz=NZ):
    geom = mesh.geom(0)
    dx = geom.cell_size()
    lo = geom.prob_lo()
    xs = np.array([lo[0] + (i + 0.5) * dx[0] for i in range(n)])
    ys = np.array([lo[1] + (j + 0.5) * dx[1] for j in range(n)])
    zs = np.array([lo[2] + (k + 0.5) * dx[2] for k in range(nz)])
    return np.meshgrid(xs, ys, zs, indexing="ij")


def _radius(mesh, body, n):
    """Cylindrical radius about ``body`` (axis=2) at each cell centre."""
    X, Y, _ = _cell_centres(mesh, n)
    return np.sqrt((X - body.centre[0]) ** 2 + (Y - body.centre[1]) ** 2)


def _init_mms(T, mesh, a=A, b=B):
    """Fill ``T`` with the MMS everywhere — including the solid interior.

    The IBM must reconstruct its near-surface stencil from its own BC, not
    lean on the (physically meaningless) values it happens to find in solid
    cells, so we seed the analytic field over the whole domain.
    """
    body = mesh.bodies["cyl"]
    geom = mesh.geom(0)
    dx = geom.cell_size()
    lo = geom.prob_lo()
    for mfi in blockamr.MFIterator(T.mf[0]):
        arr = T.mf[0].copy_to_host(mfi)
        blo = mfi.valid_box().small_end()
        nx, ny, nz = arr.shape[:3]
        xs = np.array([lo[0] + (blo[0] + i + 0.5) * dx[0] for i in range(nx)])
        ys = np.array([lo[1] + (blo[1] + j + 0.5) * dx[1] for j in range(ny)])
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        r2 = (X - body.centre[0]) ** 2 + (Y - body.centre[1]) ** 2
        plane = a + b * (r2 - body.radius**2)
        for k in range(nz):
            arr[:, :, k, 0] = plane
        T.mf[0].copy_from(mfi, arr)
    T.fill_patch(0, 0.0)


def _first_box(results):
    """Extract the (single) box array from an ``evaluate`` result."""
    return np.asarray(results[0][0])


def _regions(mesh, n):
    """(solid, band, bulk) boolean masks for the ``"cyl"`` body.

    - solid: cell centre inside the body (r < R)
    - band:  fluid cells within 3*dx of the surface (the ones whose stencil
             reaches into the solid and so depend on the IBM treatment)
    - bulk:  fluid cells more than 3*dx from the surface

    Domain-edge cells in x/y are excluded from band/bulk: the mesh is periodic
    but the MMS is not, so their periodic ghosts are wrong.
    """
    body = mesh.bodies["cyl"]
    r = _radius(mesh, body, n)
    dx = float(mesh.geom(0).cell_size()[0])
    edge = np.zeros(r.shape, dtype=bool)
    edge[0, :, :] = edge[-1, :, :] = True
    edge[:, 0, :] = edge[:, -1, :] = True
    solid = r < body.radius
    band = (r >= body.radius) & (r < body.radius + 3.0 * dx) & ~edge
    bulk = (r >= body.radius + 3.0 * dx) & ~edge
    return solid, band, bulk


def _laplacian_ibm(mesh, bc, n, ibm="ghostCell"):
    """Fill the MMS and evaluate the IBM laplacian, returning the box array."""
    T = CellField(mesh, ncomp=1, ngrow=1, name="T", ibm_bc={"cyl": bc})
    _init_mms(T, mesh)
    eqn = Equation(exp.laplacian(1.0, T))
    return _first_box(evaluate(eqn, t=0.0, solution={"ibm": ibm}))


# ---------------------------------------------------------------------------
# Validation / wiring
# ---------------------------------------------------------------------------


def test_unknown_ibm_name_raises(blockamr_session):
    """An unregistered ``solution["ibm"]`` name is rejected."""
    mesh = _make_mesh(32)
    T = CellField(mesh, ncomp=1, ngrow=1, name="T", ibm_bc={"cyl": FixedValue(A)})
    _init_mms(T, mesh)
    eqn = Equation(exp.laplacian(1.0, T))
    with pytest.raises(ValueError, match="Unknown IBM method"):
        evaluate(eqn, t=0.0, solution={"ibm": "noSuchMethod"})


def test_ibm_requires_body(blockamr_session):
    """Requesting IBM with no immersed bodies on the mesh is an error."""
    mesh = _make_mesh(32, bodies={})
    T = CellField(mesh, ncomp=1, ngrow=1, name="T", ibm_bc={})
    eqn = Equation(exp.laplacian(1.0, T))
    with pytest.raises(ValueError, match="mesh.bodies"):
        evaluate(eqn, t=0.0, solution={"ibm": "ghostCell"})


@pytest.mark.parametrize(
    "make_bc, offending",
    [
        # a body ("cyl") with no matching ibm_bc entry
        (lambda: {}, "cyl"),
        # an ibm_bc entry ("ghost") with no matching body
        (lambda: {"cyl": FixedValue(A), "ghost": FixedValue(1.0)}, "ghost"),
    ],
)
def test_ibm_bc_keys_must_match_bodies(blockamr_session, make_bc, offending):
    """``ibm_bc`` keys must match ``mesh.bodies`` exactly; a key present in one
    but not the other is an error naming the offending patch. This pins the
    patch-keyed dict shape on both sides (the multi-patch contract)."""
    mesh = _make_mesh(32)  # bodies = {"cyl": ...}
    T = CellField(mesh, ncomp=1, ngrow=1, name="T", ibm_bc=make_bc())
    eqn = Equation(exp.laplacian(1.0, T))
    with pytest.raises(ValueError, match=offending):
        evaluate(eqn, t=0.0, solution={"ibm": "ghostCell"})


def test_ibm_is_opt_in(blockamr_session):
    """No ``"ibm"`` key -> plain laplacian, bitwise identical to a field with
    no ``ibm_bc`` at all (the IBM datum must not perturb the opt-out path)."""
    mesh = _make_mesh(32)
    T = CellField(mesh, ncomp=1, ngrow=1, name="T", ibm_bc={"cyl": FixedValue(A)})
    _init_mms(T, mesh)
    with_bc = _first_box(evaluate(Equation(exp.laplacian(1.0, T)), t=0.0))

    plain_mesh = _make_mesh(32)
    P = CellField(plain_mesh, ncomp=1, ngrow=1, name="T")
    _init_mms(P, plain_mesh)
    plain = _first_box(evaluate(Equation(exp.laplacian(1.0, P)), t=0.0))

    assert np.array_equal(with_bc, plain)


# ---------------------------------------------------------------------------
# MMS: laplacian(T) = 4b
# ---------------------------------------------------------------------------


def test_solid_cells_are_masked(blockamr_session):
    """The result is exactly zero inside the body."""
    n = 64
    mesh = _make_mesh(n)
    lap = _laplacian_ibm(mesh, FixedValue(A), n)
    solid, _band, _bulk = _regions(mesh, n)
    assert np.all(lap[solid] == 0.0)


def test_mms_fixed_value_bulk_exact(blockamr_session):
    """Bulk fluid cells recover 4b to machine precision (Dirichlet datum)."""
    n = 64
    mesh = _make_mesh(n)
    lap = _laplacian_ibm(mesh, FixedValue(A), n)
    _solid, _band, bulk = _regions(mesh, n)
    assert np.max(np.abs(lap[bulk] - LAP_EXACT)) < 1e-8


def test_mms_fixed_value_band_converges(blockamr_session):
    """Near-surface error shrinks under refinement (first-order treatment)."""

    def band_max_err(n):
        mesh = _make_mesh(n)
        lap = _laplacian_ibm(mesh, FixedValue(A), n)
        _solid, band, _bulk = _regions(mesh, n)
        return float(np.max(np.abs(lap[band] - LAP_EXACT)))

    err_32 = band_max_err(32)
    err_64 = band_max_err(64)
    assert err_64 < 0.7 * err_32


def test_mms_fixed_gradient_bulk_exact(blockamr_session):
    """Same MMS via its Neumann datum (dT/dr|_R = 2bR): bulk exact."""
    n = 64
    mesh = _make_mesh(n)
    lap = _laplacian_ibm(mesh, FixedGradient(GRAD_DATUM), n)
    _solid, _band, bulk = _regions(mesh, n)
    assert np.max(np.abs(lap[bulk] - LAP_EXACT)) < 1e-8


@pytest.mark.parametrize(
    "fraction, pure_bc",
    [
        (1.0, lambda: FixedValue(A)),  # fraction=1 -> pure FixedValue
        (0.0, lambda: FixedGradient(GRAD_DATUM)),  # fraction=0 -> pure FixedGradient
    ],
)
def test_mixed_limits(blockamr_session, fraction, pure_bc):
    """Mixed at fraction 1/0 is bitwise equal to the corresponding pure BC."""
    n = 48
    mesh_mixed = _make_mesh(n)
    mixed = _laplacian_ibm(
        mesh_mixed, Mixed(value=A, gradient=GRAD_DATUM, fraction=fraction), n
    )
    mesh_pure = _make_mesh(n)
    pure = _laplacian_ibm(mesh_pure, pure_bc(), n)
    assert np.array_equal(mixed, pure)


# ---------------------------------------------------------------------------
# More than one IBM patch
# ---------------------------------------------------------------------------


def test_two_patches_independent(blockamr_session):
    """Two patches carry independent per-patch BCs. A second body placed
    entirely outside the domain (with a wildly different BC) must not leak into
    the first patch's result, which stays bitwise equal to the single-patch
    run — proving ``ibm_bc`` binds each BC to its own patch by key."""
    n = 64

    ref_mesh = _make_mesh(n)
    ref = _laplacian_ibm(ref_mesh, FixedValue(A), n)

    two_mesh = _make_mesh(
        n,
        bodies={
            "cyl": Cylinder(centre=CENTRE, radius=R, axis=AXIS),
            "far": Cylinder(centre=(5.0, 5.0), radius=R, axis=AXIS),
        },
    )
    T = CellField(
        two_mesh,
        ncomp=1,
        ngrow=1,
        name="T",
        ibm_bc={"cyl": FixedValue(A), "far": FixedGradient(999.0)},
    )
    _init_mms(T, two_mesh)
    two = _first_box(
        evaluate(Equation(exp.laplacian(1.0, T)), t=0.0, solution={"ibm": "ghostCell"})
    )

    assert np.array_equal(two, ref)
