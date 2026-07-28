# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

import os

import numpy as np

import blockamr
from blockamr.mesh import AmrMesh
from blockamr.field import CellField


def _make_geom_and_info(ncell=32, max_level=0):
    box = blockamr.Box([0, 0, 0], [ncell - 1, ncell - 1, ncell - 1])
    rb = blockamr.RealBox([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    geom = blockamr.Geometry(box, rb, 0, [1, 1, 1])
    info = blockamr.AmrInfo()
    info.max_level = max_level
    info.set_ref_ratio(0, 2)
    info.set_max_grid_size(0, 32)
    info.set_blocking_factor(0, 8)
    return geom, info


def _tag_all(lev, tags, time, ngrow):
    """Tag every cell on the level for refinement."""
    for tbi in blockamr.TagBoxIterator(tags):
        bx = tbi.valid_box()
        lo = bx.small_end()
        hi = bx.big_end()
        nx = hi[0] - lo[0] + 1
        ny = hi[1] - lo[1] + 1
        nz = hi[2] - lo[2] + 1
        mask = np.ones((nx, ny, nz), dtype=np.int32)
        tbi.set_tags(mask)


def test_amr_mesh_init_single_level():
    geom, info = _make_geom_and_info(ncell=32, max_level=0)
    mesh = AmrMesh(geom, info)
    mesh.init_from_scratch(0.0)
    assert mesh.n_levels() == 1
    assert mesh.finest_level() == 0
    assert mesh.max_level == 0


def test_amr_mesh_field_allocation():
    """AmrField allocates MultiFab on init_from_scratch."""
    geom, info = _make_geom_and_info(ncell=32, max_level=0)
    mesh = AmrMesh(geom, info)
    phi = CellField(mesh, name="phi", ncomp=1, ngrow=1)
    mesh.init_from_scratch(0.0)

    assert phi.mf[0] is not None
    assert phi.mf[0].num_comp() == 1
    assert phi.mf[0].n_grow() == 1


def test_amr_mesh_metadata():
    geom, info = _make_geom_and_info(ncell=32, max_level=0)
    mesh = AmrMesh(geom, info)
    mesh.init_from_scratch(0.0)

    g = mesh.geom(0)
    assert g.cell_size()[0] > 0
    ba = mesh.box_array(0)
    dm = mesh.dm(0)
    assert ba is not None
    assert dm is not None


def test_amr_mesh_ref_ratio():
    """ref_ratio returns the configured refinement ratio."""
    geom, info = _make_geom_and_info(ncell=16, max_level=1)
    mesh = AmrMesh(geom, info)
    rr = mesh.ref_ratio(0)
    assert rr[0] == 2


def test_amr_mesh_regrid_calls_error_est():
    """Regrid invokes the tag function."""
    geom, info = _make_geom_and_info(ncell=32, max_level=1)
    mesh = AmrMesh(geom, info)
    phi = CellField(mesh, name="phi", ncomp=1, ngrow=0)
    mesh.init_from_scratch(0.0)

    tag_called = [False]

    def tag_func(lev, tags, time, ngrow):
        tag_called[0] = True

    mesh.regrid(0.0, tag=tag_func)
    assert tag_called[0]


def test_amr_mesh_regrid_creates_fine_level():
    """Tagging all cells during regrid creates level 1."""
    geom, info = _make_geom_and_info(ncell=16, max_level=1)
    mesh = AmrMesh(geom, info)
    phi = CellField(mesh, name="phi", ncomp=1, ngrow=0)
    mesh.init_from_scratch(0.0)
    assert mesh.n_levels() == 1

    mesh.regrid(0.0, tag=_tag_all)
    assert mesh.n_levels() == 2
    assert mesh.finest_level() == 1


def test_amr_mesh_regrid_allocates_field_on_fine_level():
    """After regrid creates level 1, AmrField.mf[1] is allocated."""
    geom, info = _make_geom_and_info(ncell=16, max_level=1)
    mesh = AmrMesh(geom, info)
    phi = CellField(mesh, name="phi", ncomp=1, ngrow=0)
    mesh.init_from_scratch(0.0)
    assert phi.mf[1] is None

    mesh.regrid(0.0, tag=_tag_all)
    assert phi.mf[1] is not None
    assert phi.mf[1].num_comp() == 1


def test_amr_mesh_write_plotfile(tmp_path):
    """AmrMesh.write_plotfile produces valid directory structure."""
    geom, info = _make_geom_and_info(ncell=16, max_level=0)
    mesh = AmrMesh(geom, info)
    phi = CellField(mesh, name="phi", ncomp=1, ngrow=0)
    mesh.init_from_scratch(0.0)

    plotdir = str(tmp_path / "plt_mesh")
    mesh.write_plotfile(plotdir, phi, 0.0)

    assert os.path.isdir(plotdir)
    assert os.path.isfile(os.path.join(plotdir, "Header"))
    assert os.path.isdir(os.path.join(plotdir, "Level_0"))


def test_amr_mesh_multiple_fields():
    """Multiple AmrFields are all allocated on init."""
    geom, info = _make_geom_and_info(ncell=16, max_level=0)
    mesh = AmrMesh(geom, info)
    phi = CellField(mesh, name="phi", ncomp=1, ngrow=0)
    rho = CellField(mesh, name="rho", ncomp=1, ngrow=1)
    mesh.init_from_scratch(0.0)

    assert phi.mf[0] is not None
    assert rho.mf[0] is not None
    assert phi.mf[0].n_grow() == 0
    assert rho.mf[0].n_grow() == 1
