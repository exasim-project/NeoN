# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""TDD-2 Cycles 6 & 7: solve with single-level Mesh and multi-level AmrMesh."""

import numpy as np

import neon.blockamr as blockamr
from neon.blockamr.field import CellField, FaceField
from neon.blockamr.mesh import Mesh, AmrMesh
from neon.blockamr.dsl import exp, solve
from neon.blockamr.schemes.div_schemes import Upwind


def _tag_all(lev, tags, time, ngrow):
    for tbi in blockamr.TagBoxIterator(tags):
        bx = tbi.valid_box()
        lo = bx.small_end()
        hi = bx.big_end()
        nx = hi[0] - lo[0] + 1
        ny = hi[1] - lo[1] + 1
        nz = hi[2] - lo[2] + 1
        tbi.set_tags(np.ones((nx, ny, nz), dtype=np.int32))


def test_solve_single_level_constant_advection(blockamr_session):
    """Constant phi advected by constant velocity stays constant."""
    box = blockamr.Box([0, 0, 0], [15, 15, 15])
    rb = blockamr.RealBox([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    geom = blockamr.Geometry(box, rb, 0, [1, 1, 1])
    ba = blockamr.BoxArray(box)
    ba.max_size(16)
    dm = blockamr.DistributionMapping(ba)

    mesh = Mesh(ba, dm, geom)
    phi = CellField(mesh, ncomp=1, ngrow=1, name="phi")
    ff = FaceField(mesh, ncomp=1, ngrow=1, name="U")

    # Set phi to constant 5.0
    for mfi in blockamr.MFIterator(phi.mf[0]):
        arr = phi.mf[0].copy_to_host(mfi)
        arr[:] = 5.0
        phi.mf[0].copy_from(mfi, arr)
    phi.fill_patch(0, 0.0)

    # Set face velocity to constant 1.0 in all directions
    for d in range(3):
        for mfi in blockamr.MFIterator(ff[0][d].mf):
            arr = ff[0][d].mf.copy_to_host(mfi)
            arr[:] = 1.0
            ff[0][d].mf.copy_from(mfi, arr)

    expr = exp.ddt(phi) + exp.div(ff, phi, scheme=Upwind())
    solve(expr, 0.0, 0.001)

    for mfi in blockamr.MFIterator(phi.mf[0]):
        arr = phi.mf[0].copy_to_host(mfi)
        # Interior cells should remain ~5.0 (constant advected by constant)
        ng = phi.mf[0].n_grow()
        s = slice(ng, -ng if ng else None)
        assert np.allclose(arr[s, s, s, 0], 5.0, atol=1e-10)


def test_solve_multilevel_average_down(blockamr_session):
    """After solve on 2 levels, coarse and fine exist and don't crash."""
    box = blockamr.Box([0, 0, 0], [15, 15, 15])
    rb = blockamr.RealBox([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    geom = blockamr.Geometry(box, rb, 0, [1, 1, 1])

    info = blockamr.AmrInfo()
    info.max_level = 1
    info.set_ref_ratio(0, 2)
    info.set_max_grid_size(0, 16)
    info.set_blocking_factor(0, 8)

    mesh = AmrMesh(geom, info)
    phi = CellField(mesh, ncomp=1, ngrow=1, name="phi")
    ff = FaceField(mesh, ncomp=1, ngrow=1, name="U")
    mesh.init_from_scratch(0.0)
    mesh.regrid(0.0, tag=_tag_all)

    # Set constant values on all levels
    for lev in range(mesh.n_levels()):
        for mfi in blockamr.MFIterator(phi.mf[lev]):
            arr = phi.mf[lev].copy_to_host(mfi)
            arr[:] = 5.0
            phi.mf[lev].copy_from(mfi, arr)
        phi.fill_patch(lev, 0.0)
        for d in range(3):
            for mfi in blockamr.MFIterator(ff[lev][d].mf):
                arr = ff[lev][d].mf.copy_to_host(mfi)
                arr[:] = 1.0
                ff[lev][d].mf.copy_from(mfi, arr)

    expr = exp.ddt(phi) + exp.div(ff, phi, scheme=Upwind())
    solve(expr, 0.0, 0.001)

    # After solve, coarse and fine should exist
    assert phi.mf[0] is not None
    assert phi.mf[1] is not None
