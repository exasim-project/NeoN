# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""Flow around a cylinder via direct-forcing immersed boundary (IBM).

The cylinder is not cut into the grid; instead the engine pins the velocity to
zero in the solid cells each step (mask from centre/radius/axis). The projection
then deflects the free stream around the zero-velocity zone. Checks the mask, the
no-slip forcing, and a physically sane wake (upstream stagnation, accelerated
flanks, low-velocity wake) without asserting literature drag (that is Spec 03).

Also covers the mesh-owned IBM restructure (API doc §6, plan 04):
``mesh.body`` / ``mesh.build_ibm`` / ``mesh.ibm_data`` and the ``IBM`` registry.
"""

import numpy as np
import pytest

import blockamr
from blockamr.bc import VectorBC, fixedValue, NeumannBC, slip
from blockamr.incompressible import build_incompressible, step
from blockamr.ibm import IBM, Cylinder, DirectForcing
from blockamr.mesh import AmrMesh, Mesh

U0 = 1.0
D = 0.2
RADIUS = D / 2.0
NU = U0 * D / 20.0  # Re = 20
NX, NY, NZ = 48, 24, 8
LX, LY, LZ = 2.0, 1.0, 0.25
CENTER = [0.5, 0.5, 0.125]


def _make_cylinder_solver():
    box = blockamr.Box([0, 0, 0], [NX - 1, NY - 1, NZ - 1])
    rb = blockamr.RealBox([0.0, 0.0, 0.0], [LX, LY, LZ])
    geom = blockamr.Geometry(box, rb, 0, [0, 0, 1])
    ba = blockamr.BoxArray(box)
    ba.max_size(max(NX, NY, NZ))
    dm = blockamr.DistributionMapping(ba)
    mesh = Mesh(ba, dm, geom)
    mesh.body = Cylinder(centre=CENTER, radius=RADIUS, axis=2)

    u_bc = VectorBC(
        xlo=fixedValue([U0, 0.0, 0.0]),
        xhi=NeumannBC(),
        ylo=slip(),
        yhi=slip(),
    )
    dt = 0.2 * float(geom.cell_size()[0]) / U0
    solver = build_incompressible(mesh, NU, dt, U_bc=u_bc, sol_U={"ibm": "directForcing"})

    ng = solver.U.mf[0].n_grow()
    g = solver.U.mf[0].grown_arrays()[0]
    g = g.at[:, :, :, 0].set(U0).at[:, :, :, 1:].set(0.0)
    solver.U.mf[0].copy_grown_arrays([g])
    return solver, ng


def _valid_u(solver, ng):
    arr = np.array(solver.U.mf[0].arrays()[0])
    return arr[ng : ng + NX, ng : ng + NY, ng : ng + NZ, :], arr


def test_cylinder_mask_matches_geometry(blockamr_session):
    """The solid-cell count matches the analytic disc area (per z-layer)."""
    solver, ng = _make_cylinder_solver()
    data = solver.mesh.ibm_data(DirectForcing)
    mask = np.array(data.masks[0][0])
    dx, dy = float(LX / NX), float(LY / NY)
    expected = np.pi * RADIUS**2 / (dx * dy) * NZ
    # a cell-centre-in-disc mask staircases the boundary — at this coarse
    # resolution (radius ≈ 2.4 cells) it under-counts the smooth area by ~12%.
    assert mask.sum() == pytest.approx(expected, rel=0.2)
    assert mask.sum() > 0


def test_cylinder_wake_and_noslip(blockamr_session):
    """Direct forcing holds U=0 in the body and produces a sane wake."""
    solver, ng = _make_cylinder_solver()
    for _ in range(150):
        step(solver)

    u, arr = _valid_u(solver, ng)
    ux = u[..., 0]
    data = solver.mesh.ibm_data(DirectForcing)
    mask = np.array(data.masks[0][0])

    # bounded & finite
    assert np.all(np.isfinite(arr))
    assert float(np.max(np.abs(arr))) < 2.0 * U0

    # no-slip: velocity pinned to zero inside the body
    assert float(np.max(np.abs(u[mask]))) < 1e-6

    # sane wake structure around the body
    ci, cj = int(CENTER[0] / LX * NX), int(CENTER[1] / LY * NY)
    rc = int(RADIUS / LX * NX)
    u_wake = float(np.mean(ux[ci + rc + 2 : ci + rc + 6, cj - 2 : cj + 2, :]))
    u_flank = float(np.mean(ux[ci - 2 : ci + 2, cj + rc + 1 : cj + rc + 3, :]))
    u_stag = float(np.mean(ux[ci - rc - 4 : ci - rc - 1, cj - 2 : cj + 2, :]))

    assert u_wake < 0.6 * U0  # wake deficit behind the body
    assert u_flank > 1.05 * U0  # accelerated over the flanks
    assert u_stag < 0.9 * U0  # decelerated at the upstream stagnation


# ---------------------------------------------------------------------------
# mesh.body / build_ibm / ibm_data (API doc §6)
# ---------------------------------------------------------------------------


def _make_plain_mesh():
    box = blockamr.Box([0, 0, 0], [NX - 1, NY - 1, NZ - 1])
    rb = blockamr.RealBox([0.0, 0.0, 0.0], [LX, LY, LZ])
    geom = blockamr.Geometry(box, rb, 0, [0, 0, 1])
    ba = blockamr.BoxArray(box)
    ba.max_size(max(NX, NY, NZ))
    dm = blockamr.DistributionMapping(ba)
    return Mesh(ba, dm, geom)


def test_build_ibm_ibm_data_round_trip(blockamr_session):
    """``build_ibm([DirectForcing])`` precomputes data that ``ibm_data`` returns."""
    mesh = _make_plain_mesh()
    mesh.body = Cylinder(centre=CENTER, radius=RADIUS, axis=2)

    mesh.build_ibm([DirectForcing])
    data = mesh.ibm_data(DirectForcing)

    dx, dy = float(LX / NX), float(LY / NY)
    expected = np.pi * RADIUS**2 / (dx * dy) * NZ
    solid_count = sum(int(np.array(m).sum()) for m in data.masks[0])
    assert solid_count == pytest.approx(expected, rel=0.2)
    assert data.force_history == []
    # Name-based lookup resolves to the same class used for the class-list
    # form of build_ibm — both paths must coexist (plan 04).
    assert IBM.lookup("directForcing") is DirectForcing


def test_ibm_data_before_build_raises(blockamr_session):
    mesh = _make_plain_mesh()
    mesh.body = Cylinder(centre=CENTER, radius=RADIUS, axis=2)
    with pytest.raises(RuntimeError, match="not built"):
        mesh.ibm_data(DirectForcing)


def test_build_ibm_without_body_raises(blockamr_session):
    mesh = _make_plain_mesh()
    with pytest.raises(ValueError, match="mesh.body"):
        mesh.build_ibm([DirectForcing])


def test_ibm_lookup_unknown_method_raises():
    with pytest.raises(ValueError, match="Unknown IBM method"):
        IBM.lookup("bogus")


@pytest.mark.parametrize("name", ["cutCell"])
def test_ibm_lookup_not_implemented_methods_raise(name):
    """``ghostCell`` used to be listed here; it is implemented now (task T10)
    and its lookup is covered by ``test_ibm_rungs.py``."""
    with pytest.raises(NotImplementedError):
        IBM.lookup(name)


def _tag_all(lev, tags, time, ngrow):
    """Tag every cell on the level for refinement."""
    for tbi in blockamr.TagBoxIterator(tags):
        bx = tbi.valid_box()
        lo = bx.small_end()
        hi = bx.big_end()
        nx, ny, nz = (hi[d] - lo[d] + 1 for d in range(3))
        tbi.set_tags(np.ones((nx, ny, nz), dtype=np.int32))


def test_build_ibm_rebuilds_on_regrid(blockamr_session):
    """Regrid recomputes the masks for the new box arrays but keeps the
    accumulated force history (a time series, not spatial data)."""
    box = blockamr.Box([0, 0, 0], [15, 15, 15])
    rb = blockamr.RealBox([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    geom = blockamr.Geometry(box, rb, 0, [1, 1, 1])
    info = blockamr.AmrInfo()
    info.max_level = 1
    info.set_ref_ratio(0, 2)
    info.set_max_grid_size(0, 16)
    info.set_blocking_factor(0, 8)
    mesh = AmrMesh(geom, info)
    mesh.init_from_scratch(0.0)
    mesh.body = Cylinder(centre=[0.5, 0.5, 0.5], radius=0.2, axis=2)

    mesh.build_ibm([DirectForcing])
    data_before = mesh.ibm_data(DirectForcing)
    assert len(data_before.masks) == 1
    data_before.force_history.append((0.0, 1.0, 2.0, 3.0))

    mesh.regrid(0.0, tag=_tag_all)
    assert mesh.n_levels() == 2

    data_after = mesh.ibm_data(DirectForcing)
    assert data_after is not data_before  # masks rebuilt for the new box arrays
    assert len(data_after.masks) == 2  # data now covers the new fine level
    assert data_after.force_history == [(0.0, 1.0, 2.0, 3.0)]  # preserved


# ---------------------------------------------------------------------------
# Force-history numerics parity with the pre-refactor engine
# ---------------------------------------------------------------------------

# Captured from the pre-refactor engine (``eb=`` ctor kwarg, ``_force_solid``)
# on this exact case — 20 steps from a uniform U0 initial field. Locks the
# restructure's numerics: same solid-cell classification, same apply order
# (after correct(), each step()).
_BASELINE_FORCE_HISTORY = [
    (0.0, 0.8333333333333333, 0.0, 0.0),
    (0.008333333333333333, 0.4535420390239231, -2.8912057932946782e-18, -9.676504389433457e-18),
    (0.016666666666666666, 0.29267605036797906, -3.989863994746656e-16, -3.027082132865158e-20),
    (0.025, 0.21204097007008477, 2.255140518769849e-16, -4.160062323890525e-19),
    (0.03333333333333333, 0.16913018507740363, -1.7058114180438602e-16, 5.314723564248453e-19),
    (0.041666666666666664, 0.1434825159770522, 1.7925475918427006e-16, 1.4855231346826116e-19),
    (0.049999999999999996, 0.12786589077657806, -1.1564823173178715e-16, -1.0741626499879733e-20),
    (0.05833333333333333, 0.11714182896910713, 1.0986582014519779e-16, 2.210261957063631e-19),
    (0.06666666666666667, 0.10987373809555466, -8.673617379884034e-17, -3.935678956884304e-19),
    (0.075, 0.10431134640987269, 5.204170427930421e-17, 7.054855292977845e-19),
    (0.08333333333333333, 0.10019898073258984, -6.938893903907228e-17, 1.6724178342031352e-18),
    (0.09166666666666666, 0.09680572270883377, 4.336808689942017e-17, -1.1670280503689134e-19),
    (0.09999999999999999, 0.09413211667318691, -4.336808689942017e-17, 3.1846015262721814e-19),
    (0.10833333333333332, 0.09185996630731189, 8.673617379884035e-18, 3.5579311874433765e-19),
    (0.11666666666666665, 0.08999997111718563, -4.625929269271485e-17, -1.1111260201718109e-18),
    (0.12499999999999999, 0.0883553635114706, 1.1564823173178713e-17, 1.4884728606291724e-18),
    (0.13333333333333333, 0.08695842890724291, -1.734723475976807e-17, 5.0697855819599863e-20),
    (0.14166666666666666, 0.08570778089225531, 5.7824115865893565e-18, 8.257775823281798e-19),
    (0.15, 0.08462624403484457, -1.1564823173178713e-17, 1.8324168855217586e-19),
    (0.15833333333333333, 0.08365226506614806, -1.1564823173178713e-17, -1.189455618760562e-19),
]


def test_force_history_matches_pre_refactor_baseline(blockamr_session):
    """The restructured DirectForcing reproduces the pre-refactor force
    history exactly — the Cd/Cl/St acceptance oracle depends on this."""
    solver, _ng = _make_cylinder_solver()
    for _ in range(len(_BASELINE_FORCE_HISTORY)):
        step(solver)

    data = solver.mesh.ibm_data(DirectForcing)
    hist = DirectForcing.force_history(data)
    assert len(hist) == len(_BASELINE_FORCE_HISTORY)
    for (t, fx, fy, fz), (bt, bfx, bfy, bfz) in zip(hist, _BASELINE_FORCE_HISTORY):
        assert t == pytest.approx(bt, abs=1e-12)
        assert fx == pytest.approx(bfx, rel=1e-6, abs=1e-9)
        assert fy == pytest.approx(bfy, abs=1e-8)
        assert fz == pytest.approx(bfz, abs=1e-8)
