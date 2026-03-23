# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

import numpy as np

import neon.blockamr as blockamr
from neon.blockamr.mesh import AmrMesh
from neon.blockamr.field import CellField


def _tag_all(lev, tags, time, ngrow):
    for tbi in blockamr.TagBoxIterator(tags):
        bx = tbi.valid_box()
        lo = bx.small_end()
        hi = bx.big_end()
        nx = hi[0] - lo[0] + 1
        ny = hi[1] - lo[1] + 1
        nz = hi[2] - lo[2] + 1
        tbi.set_tags(np.ones((nx, ny, nz), dtype=np.int32))


def test_average_down_constant(blockamr_session):
    """Fine set to constant 42 -> coarse cells match after average_down."""
    box = blockamr.Box([0, 0, 0], [15, 15, 15])
    rb = blockamr.RealBox([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    geom = blockamr.Geometry(box, rb, 0, [1, 1, 1])

    info = blockamr.AmrInfo()
    info.max_level = 1
    info.set_ref_ratio(0, 2)
    info.set_max_grid_size(0, 16)
    info.set_blocking_factor(0, 8)

    mesh = AmrMesh(geom, info)
    phi = CellField(mesh, ncomp=1, ngrow=0, name="phi")
    mesh.init_from_scratch(0.0)
    mesh.regrid(0.0, tag=_tag_all)

    assert phi.mf[1] is not None

    # Set coarse to 0, fine to 42
    for mfi in blockamr.MFIterator(phi.mf[0]):
        arr = phi.mf[0].copy_to_host(mfi)
        arr[:] = 0.0
        phi.mf[0].copy_from(mfi, arr)
    for mfi in blockamr.MFIterator(phi.mf[1]):
        arr = phi.mf[1].copy_to_host(mfi)
        arr[:] = 42.0
        phi.mf[1].copy_from(mfi, arr)

    blockamr.average_down(
        phi.mf[1], phi.mf[0],
        mesh.geom(1), mesh.geom(0),
        0, 1, mesh.ref_ratio(0),
    )

    for mfi in blockamr.MFIterator(phi.mf[0]):
        arr = phi.mf[0].copy_to_host(mfi)
        assert np.allclose(arr, 42.0)
