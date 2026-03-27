# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""TDD-2 Cycles 2 & 8: Div with bucket-based dispatch and update_face_fluxes with _FaceFieldLevel."""

import jax.numpy as jnp
import numpy as np

import neon.blockamr as blockamr
from neon.blockamr.field import CellField, FaceField
from neon.blockamr.mesh import Mesh
from neon.blockamr.operators.div import Div, update_face_fluxes
from neon.blockamr.schemes.div_schemes import Upwind
from neon.blockamr.flattened_boxes import flattened_boxes_from_mf, build_buckets


def _make_mesh(n_cell=16, max_size=16):
    """Create a periodic Mesh on [0,1]^3."""
    box = blockamr.Box([0, 0, 0], [n_cell - 1, n_cell - 1, n_cell - 1])
    rb = blockamr.RealBox([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    geom = blockamr.Geometry(box, rb, 0, [1, 1, 1])
    ba = blockamr.BoxArray(box)
    ba.max_size(max_size)
    dm = blockamr.DistributionMapping(ba)
    mesh = Mesh(ba, dm, geom)
    return mesh


def test_div_build_kernel_lev0(blockamr_session):
    """Div.build_kernel(bucket, t) uses level-0 face data."""
    mesh = _make_mesh()
    phi = CellField(mesh, ncomp=1, ngrow=1, name="phi")
    ff = FaceField(mesh, ncomp=1, ngrow=1, name="U")

    # Set face velocity to constant 1 in all directions
    for d in range(3):
        for mfi in blockamr.MFIterator(ff[0][d].mf):
            arr = ff[0][d].mf.copy_to_host(mfi)
            arr[:] = 1.0
            ff[0][d].mf.copy_from(mfi, arr)

    div_op = Div(ff, phi, scheme=Upwind())
    mf = phi.mf[0]
    fb = flattened_boxes_from_mf(mf)
    dh = tuple(float(d) for d in mesh.geom(0).cell_size())
    buckets = build_buckets(fb, dh, lev=0)
    for bucket in buckets:
        if bucket.n_valid == 0:
            continue
        kernel = div_op.build_kernel(bucket, 0.0)
        assert kernel is not None
        assert hasattr(kernel, '__call__')


def test_div_build_kernel_returns_callable_result(blockamr_session):
    """Div kernel applied to uniform field returns near-zero divergence."""
    from neon.blockamr.bucket_dispatch import process_bucket

    mesh = _make_mesh()
    phi = CellField(mesh, ncomp=1, ngrow=1, name="phi")
    ff = FaceField(mesh, ncomp=1, ngrow=1, name="U")

    # Constant velocity → divergence of constant field = 0
    for d in range(3):
        for mfi in blockamr.MFIterator(ff[0][d].mf):
            arr = ff[0][d].mf.copy_to_host(mfi)
            arr[:] = 1.0
            ff[0][d].mf.copy_from(mfi, arr)

    # Set phi to constant
    for mfi in blockamr.MFIterator(phi.mf[0]):
        arr = phi.mf[0].copy_to_host(mfi)
        arr[:] = 1.0
        phi.mf[0].copy_from(mfi, arr)
    phi.fill_patch(0, 0.0)

    div_op = Div(ff, phi, scheme=Upwind())
    mf = phi.mf[0]
    fb = flattened_boxes_from_mf(mf)
    dh = tuple(float(d) for d in mesh.geom(0).cell_size())
    buckets = build_buckets(fb, dh, lev=0)
    for bucket in buckets:
        if bucket.n_valid == 0:
            continue
        kernel = div_op.build_kernel(bucket, 0.0)
        # Use process_bucket to evaluate the kernel on the bucket
        result = process_bucket(bucket, 1.0, (kernel,))
        # process_bucket returns phi - dt*div(phi).
        # For constant phi=1, div=0, so result ≈ 1.0 (unchanged)
        valid = result[:bucket.n_valid]
        assert float(jnp.abs(valid - 1.0).max()) < 1e-10


def test_update_face_fluxes_with_face_field_level(blockamr_session):
    """update_face_fluxes works with _FaceFieldLevel from FaceField[0]."""
    mesh = _make_mesh()
    ff = FaceField(mesh, ncomp=1, ngrow=0, name="U")

    def const_vel(x, y, z, t):
        return jnp.ones_like(x), jnp.zeros_like(x), jnp.zeros_like(x)

    update_face_fluxes(ff[0], const_vel, mesh.geom(0), 0.0)

    for mfi in blockamr.MFIterator(ff[0][0].mf):
        arr = ff[0][0].mf.copy_to_host(mfi)
        assert np.all(arr > 0)  # x-velocity = 1
