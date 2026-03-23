# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

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


def test_div_default_unchanged():
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

    for mfi in blockamr.MFIterator(phi.mf[0]):
        phi_arr = phi.mf[0].grown_array(mfi)
        kernel = div_op.build_kernel(mfi, t=0.0)
        result = kernel(phi_arr)
        assert np.allclose(result, 0.0, atol=1e-12)


def test_div_with_linear_scheme():
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
    results_upwind = []
    for mfi in blockamr.MFIterator(phi.mf[0]):
        phi_arr = phi.mf[0].grown_array(mfi)
        results_upwind.append(div_upwind.build_kernel(mfi, t=0.0)(phi_arr))

    ff_lin = build_face_fluxes(_x_vel, box, dm, geom, ngrow=1, t=0.0)
    div_linear = Div(ff_lin, phi, scheme=Linear())
    results_linear = []
    for mfi in blockamr.MFIterator(phi.mf[0]):
        phi_arr = phi.mf[0].grown_array(mfi)
        results_linear.append(div_linear.build_kernel(mfi, t=0.0)(phi_arr))

    for r_up, r_lin in zip(results_upwind, results_linear):
        # They should differ (Linear is 2nd-order, Upwind is 1st-order)
        assert not np.allclose(r_up, r_lin, atol=1e-6)


def test_laplacian_default_scheme():
    """Laplacian defaults to CentralDiffLaplacian."""
    mesh, *_ = _make_mesh(n_cell=32, max_size=32)
    phi = CellField(mesh, ncomp=1, ngrow=1, name="phi")

    def gamma_one(x, y, z, t):
        return np.ones_like(x)

    lap_op = Laplacian(gamma_one, phi)
    assert isinstance(lap_op.scheme, CentralDiffLaplacian)


def test_grad_default_scheme():
    """Grad defaults to CentralDiffGrad."""
    mesh, *_ = _make_mesh(n_cell=32, max_size=32)
    phi = CellField(mesh, ncomp=1, ngrow=1, name="phi")
    grad_op = Grad(phi)
    assert isinstance(grad_op.scheme, CentralDiffGrad)


def test_solve_with_schemes_dict():
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


def test_div_with_vanleer_scheme():
    """Div with VanLeer scheme runs without error (requires wider stencil, ngrow>=2)."""
    n_cell = 32
    # VanLeer needs 2 ghost cells
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

    for mfi in blockamr.MFIterator(phi.mf[0]):
        phi_arr = phi.mf[0].grown_array(mfi)
        kernel = div_vanleer.build_kernel(mfi, t=0.0)
        result = kernel(phi_arr)
        # Should produce finite results
        assert np.all(np.isfinite(result))


def test_factory_div_with_scheme():
    """exp.div() accepts optional scheme parameter."""
    mesh, box, dm, geom = _make_mesh(n_cell=32, max_size=32)
    phi = CellField(mesh, ncomp=1, ngrow=1, name="phi")
    ff = build_face_fluxes(_x_vel, box, dm, geom, ngrow=1, t=0.0)
    div_op = exp.div(ff, phi, scheme=Linear())
    assert isinstance(div_op.scheme, Linear)


def test_solve_backward_compat():
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
