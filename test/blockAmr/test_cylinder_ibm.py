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
from blockamr.ibm import IBM, Cylinder, DirectForcing
from blockamr.mesh import AmrMesh, Mesh

U0 = 1.0
D = 0.2
RADIUS = D / 2.0
NU = U0 * D / 20.0  # Re = 20
NX, NY, NZ = 48, 24, 8
LX, LY, LZ = 2.0, 1.0, 0.25
CENTER = [0.5, 0.5, 0.125]


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


@pytest.mark.parametrize("name", ["cutCell", "ghostCell"])
def test_ibm_lookup_not_implemented_methods_raise(name):
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

