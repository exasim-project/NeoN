# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""Integration tests for scheme selection with the explicit C++ evaluation path."""

import math

import blockamr
import numpy as np
from blockamr.field import CellField
from blockamr.mesh import Mesh
from blockamr.dsl import evaluate, exp, solve
from blockamr.dsl.equation import Equation
from blockamr.operators.div import Div, build_face_fluxes
from blockamr.operators.grad import Grad
from blockamr.operators.laplacian import Laplacian
from blockamr.schemes.div_schemes import Linear, Upwind, VanLeer
from blockamr.schemes.grad_schemes import CentralDiffGrad
from blockamr.schemes.laplacian_schemes import CentralDiffLaplacian


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

    # Divergence of a constant field is zero in every box.
    for div in evaluate(div_op, t=0.0)[0]:
        assert np.allclose(div, 0.0, atol=1e-12)


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

    r_up = evaluate(div_upwind, t=0.0)[0]
    r_lin = evaluate(div_linear, t=0.0)[0]
    for up, lin in zip(r_up, r_lin):
        # They should differ (Linear is 2nd-order, Upwind is 1st-order)
        assert not np.allclose(up, lin, atol=1e-6)


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
    """Equation(schemes=...) overrides an operator's default scheme.

    schemes is bound at Equation construction (fvSchemes); solve() no
    longer takes a schemes= kwarg (plan 02).
    """
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
    expr = Equation(exp.ddt(phi) + exp.div(ff, phi), schemes={"Div": Linear()})
    solve(expr, t=0.0, dt=1e-4)

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

    for div in evaluate(div_vanleer, t=0.0)[0]:
        assert np.all(np.isfinite(div))


def test_factory_div_with_scheme(blockamr_session):
    """Div() accepts an explicit scheme object (exp.div() resolves by name only)."""
    mesh, box, dm, geom = _make_mesh(n_cell=32, max_size=32)
    phi = CellField(mesh, ncomp=1, ngrow=1, name="phi")
    ff = build_face_fluxes(_x_vel, box, dm, geom, ngrow=1, t=0.0)
    div_op = Div(ff, phi, scheme=Linear())
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
