# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""Integration tests for scheme selection with the bucket dispatch pipeline."""

import math

import neon.blockamr as blockamr
import jax.numpy as jnp
import numpy as np
from neon.blockamr.field import CellField, FaceField
from neon.blockamr.mesh import Mesh
from neon.blockamr.dsl import exp, solve
from neon.blockamr.operators.div import Div, build_face_fluxes
from neon.blockamr.operators.grad import Grad
from neon.blockamr.operators.laplacian import Laplacian
from neon.blockamr.schemes.div_schemes import QUICK, Linear, Upwind, VanLeer
from neon.blockamr.schemes.grad_schemes import CentralDiffGrad
from neon.blockamr.schemes.laplacian_schemes import CentralDiffLaplacian
from neon.blockamr.flattened_boxes import flattened_boxes_from_mf, build_buckets
from neon.blockamr.bucket_dispatch import process_bucket


def _make_mesh(n_cell=64, max_size=32):
    """Create a periodic Mesh on [0,1]^3."""
    box = blockamr.Box([0, 0, 0], [n_cell - 1, n_cell - 1, n_cell - 1])
    rb = blockamr.RealBox([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    geom = blockamr.Geometry(box, rb, 0, [1, 1, 1])
    ba = blockamr.BoxArray(box)
    ba.max_size(max_size)
    dm = blockamr.DistributionMapping(ba)
    mesh = Mesh(ba, dm, geom)
    return mesh, box, dm, geom


def _uniform_vel(x, y, z, t):
    return np.ones_like(x), np.ones_like(x), np.ones_like(x)


def _x_vel(x, y, z, t):
    return np.ones_like(x), np.zeros_like(x), np.zeros_like(x)


def test_div_default_unchanged(blockamr_session):
    """Div without explicit scheme matches old (Upwind) behaviour."""
    mesh, box, dm, geom = _make_mesh(n_cell=64, max_size=32)
    phi = CellField(mesh, ncomp=1, ngrow=1, name="phi")

    for mfi in blockamr.MFIterator(phi.mf[0]):
        arr = phi.mf[0].copy_to_host(mfi)
        arr[:, :, :, 0] = 1.0
        phi.mf[0].copy_from(mfi, arr)
    phi.fill_patch(0, 0.0)

    ff = build_face_fluxes(_uniform_vel, box, dm, geom, ngrow=1, t=0.0)
    div_op = Div(ff, phi)
    assert isinstance(div_op.scheme, Upwind)

    # Use bucket dispatch to verify divergence of constant = 0
    mf = phi.mf[0]
    fb = flattened_boxes_from_mf(mf)
    dh = tuple(float(d) for d in geom.cell_size())
    buckets = build_buckets(fb, dh, lev=0)
    for bucket in buckets:
        if bucket.n_valid == 0:
            continue
        kernel = div_op.build_kernel(bucket, 0.0)
        result = process_bucket(bucket, 1.0, (kernel,))
        # result = phi - dt*div(phi); for constant phi=1, div=0, result=1
        valid = result[:bucket.n_valid]
        assert np.allclose(valid, 1.0, atol=1e-12)


def test_div_with_linear_scheme(blockamr_session):
    """Div with Linear scheme gives different (2nd-order) results vs Upwind."""
    n_cell = 32
    mesh, box, dm, geom = _make_mesh(n_cell=n_cell, max_size=n_cell)
    phi = CellField(mesh, ncomp=1, ngrow=1, name="phi")
    dx = geom.cell_size()

    for mfi in blockamr.MFIterator(phi.mf[0]):
        arr = phi.mf[0].copy_to_host(mfi)
        bx = mfi.valid_box()
        lo = bx.small_end()
        nx = arr.shape[0]
        for i in range(nx):
            x = (lo[0] + i + 0.5) * dx[0]
            arr[i, :, :, 0] = math.sin(2 * math.pi * x)
        phi.mf[0].copy_from(mfi, arr)
    phi.fill_patch(0, 0.0)

    ff_up = build_face_fluxes(_x_vel, box, dm, geom, ngrow=1, t=0.0)
    div_upwind = Div(ff_up, phi)

    ff_lin = build_face_fluxes(_x_vel, box, dm, geom, ngrow=1, t=0.0)
    div_linear = Div(ff_lin, phi, scheme=Linear())

    mf = phi.mf[0]
    fb = flattened_boxes_from_mf(mf)
    dh = tuple(float(d) for d in geom.cell_size())
    buckets = build_buckets(fb, dh, lev=0)

    for bucket in buckets:
        if bucket.n_valid == 0:
            continue
        k_up = div_upwind.build_kernel(bucket, 0.0)
        k_lin = div_linear.build_kernel(bucket, 0.0)
        r_up = process_bucket(bucket, 1.0, (k_up,))
        r_lin = process_bucket(bucket, 1.0, (k_lin,))
        # They should differ (Linear is 2nd-order, Upwind is 1st-order)
        assert not np.allclose(r_up[:bucket.n_valid], r_lin[:bucket.n_valid], atol=1e-6)


def test_laplacian_default_scheme(blockamr_session):
    """Laplacian defaults to CentralDiffLaplacian."""
    mesh, *_ = _make_mesh(n_cell=32, max_size=32)
    phi = CellField(mesh, ncomp=1, ngrow=1, name="phi")

    def gamma_one(x, y, z, t):
        return np.ones_like(x)

    lap_op = Laplacian(gamma_one, phi)
    assert isinstance(lap_op.scheme, CentralDiffLaplacian)


def test_grad_default_scheme(blockamr_session):
    """Grad defaults to CentralDiffGrad."""
    mesh, *_ = _make_mesh(n_cell=32, max_size=32)
    phi = CellField(mesh, ncomp=1, ngrow=1, name="phi")
    grad_op = Grad(phi)
    assert isinstance(grad_op.scheme, CentralDiffGrad)


def test_solve_with_schemes_dict(blockamr_session):
    """solve() with schemes dict overrides operator's default scheme."""
    mesh, box, dm, geom = _make_mesh(n_cell=32, max_size=32)
    phi = CellField(mesh, ncomp=1, ngrow=1, name="phi")
    dx = geom.cell_size()

    for mfi in blockamr.MFIterator(phi.mf[0]):
        arr = phi.mf[0].copy_to_host(mfi)
        bx = mfi.valid_box()
        lo = bx.small_end()
        nx = arr.shape[0]
        for i in range(nx):
            x = (lo[0] + i + 0.5) * dx[0]
            arr[i, :, :, 0] = math.sin(2 * math.pi * x)
        phi.mf[0].copy_from(mfi, arr)
    phi.fill_patch(0, 0.0)
    phi_before = {}
    for mfi in blockamr.MFIterator(phi.mf[0]):
        bx = mfi.valid_box()
        lo = bx.small_end()
        phi_before[tuple(lo)] = phi.mf[0].copy_to_host(mfi)[:, :, :, 0].copy()

    ff = build_face_fluxes(_x_vel, box, dm, geom, ngrow=1, t=0.0)
    expr = exp.ddt(phi) + exp.div(ff, phi)
    solve(expr, t=0.0, dt=1e-4, schemes={"Div": Linear()})

    # The field should have changed (not zero div for non-uniform field)
    for mfi in blockamr.MFIterator(phi.mf[0]):
        bx = mfi.valid_box()
        lo = bx.small_end()
        arr_new = phi.mf[0].copy_to_host(mfi)[:, :, :, 0]
        assert not np.allclose(arr_new, phi_before[tuple(lo)])


def test_div_with_vanleer_scheme(blockamr_session):
    """Div with VanLeer scheme runs without error (requires wider stencil, ngrow>=2)."""
    n_cell = 32
    mesh, box, dm, geom = _make_mesh(n_cell=n_cell, max_size=n_cell)
    phi = CellField(mesh, ncomp=1, ngrow=2, name="phi")
    dx = geom.cell_size()

    for mfi in blockamr.MFIterator(phi.mf[0]):
        arr = phi.mf[0].copy_to_host(mfi)
        bx = mfi.valid_box()
        lo = bx.small_end()
        nx = arr.shape[0]
        for i in range(nx):
            x = (lo[0] + i + 0.5) * dx[0]
            arr[i, :, :, 0] = math.sin(2 * math.pi * x)
        phi.mf[0].copy_from(mfi, arr)
    phi.fill_patch(0, 0.0)

    ff = build_face_fluxes(_x_vel, box, dm, geom, ngrow=2, t=0.0)
    div_vanleer = Div(ff, phi, scheme=VanLeer())

    mf = phi.mf[0]
    fb = flattened_boxes_from_mf(mf)
    dh = tuple(float(d) for d in geom.cell_size())
    buckets = build_buckets(fb, dh, lev=0)
    for bucket in buckets:
        if bucket.n_valid == 0:
            continue
        kernel = div_vanleer.build_kernel(bucket, 0.0)
        result = process_bucket(bucket, 1.0, (kernel,))
        valid = result[:bucket.n_valid]
        assert np.all(np.isfinite(valid))


def test_factory_div_with_scheme(blockamr_session):
    """exp.div() accepts optional scheme parameter."""
    mesh, box, dm, geom = _make_mesh(n_cell=32, max_size=32)
    phi = CellField(mesh, ncomp=1, ngrow=1, name="phi")
    ff = build_face_fluxes(_x_vel, box, dm, geom, ngrow=1, t=0.0)
    div_op = exp.div(ff, phi, scheme=Linear())
    assert isinstance(div_op.scheme, Linear)


def test_solve_backward_compat(blockamr_session):
    """solve() without schemes kwarg works as before."""
    mesh, box, dm, geom = _make_mesh(n_cell=64, max_size=32)
    phi = CellField(mesh, ncomp=1, ngrow=1, name="phi")

    for mfi in blockamr.MFIterator(phi.mf[0]):
        arr = phi.mf[0].copy_to_host(mfi)
        arr[:, :, :, 0] = 5.0
        phi.mf[0].copy_from(mfi, arr)

    def zero_vel(x, y, z, t):
        return np.zeros_like(x), np.zeros_like(x), np.zeros_like(x)

    ff = build_face_fluxes(zero_vel, box, dm, geom, ngrow=1, t=0.0)
    expr = exp.ddt(phi) + exp.div(ff, phi)
    solve(expr, t=0.0, dt=0.01)

    for mfi in blockamr.MFIterator(phi.mf[0]):
        arr = phi.mf[0].copy_to_host(mfi)
        assert np.allclose(arr[:, :, :, 0], 5.0)
