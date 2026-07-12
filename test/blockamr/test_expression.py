# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

import math

import neon.blockamr as blockamr
import numpy as np
from neon.blockamr.field import CellField
from neon.blockamr.mesh import Mesh
from neon.blockamr.dsl import exp, solve, Equation
from neon.blockamr.operators.div import build_face_fluxes
from neon.blockamr.schemes.div_schemes import Linear


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


def _init_sin3d(phi, geom):
    """Set field to sin(2*pi*x)*sin(2*pi*y)*sin(2*pi*z)."""
    dx = geom.cell_size()
    for mfi in blockamr.MFIterator(phi.mf[0]):
        arr = phi.mf[0].copy_to_host(mfi)
        bx = mfi.valid_box()
        lo = bx.small_end()
        nx, ny, nz = arr.shape[:3]
        for i in range(nx):
            x = (lo[0] + i + 0.5) * dx[0]
            for j in range(ny):
                y = (lo[1] + j + 0.5) * dx[1]
                for k in range(nz):
                    z = (lo[2] + k + 0.5) * dx[2]
                    arr[i, j, k, 0] = (
                        math.sin(2 * math.pi * x)
                        * math.sin(2 * math.pi * y)
                        * math.sin(2 * math.pi * z)
                    )
        phi.mf[0].copy_from(mfi, arr)
    phi.fill_patch(0, 0.0)


def test_ddt_plus_div_creates_equation():
    """ddt(phi) + div(face_fluxes, phi) creates an Equation with 1 temporal + 1 spatial op."""
    mesh, box, dm, geom = _make_mesh()
    phi = CellField(mesh, ncomp=1, ngrow=1, name="phi")

    def vel(x, y, z, t):
        return np.ones_like(x), np.zeros_like(x), np.zeros_like(x)

    ff = build_face_fluxes(vel, box, dm, geom, ngrow=1, t=0.0)
    eqn = exp.ddt(phi) + exp.div(ff, phi)
    assert isinstance(eqn, Equation)
    assert len(eqn.temporal_ops) == 1
    assert len(eqn.spatial_ops) == 1


def test_scalar_mul_operator():
    """Scalar * operator sets the coefficient."""
    mesh, box, dm, geom = _make_mesh()
    phi = CellField(mesh, ncomp=1, ngrow=1, name="phi")

    def vel(x, y, z, t):
        return np.ones_like(x), np.zeros_like(x), np.zeros_like(x)

    ff = build_face_fluxes(vel, box, dm, geom, ngrow=1, t=0.0)
    div_op = 2.0 * exp.div(ff, phi)
    assert div_op.coeff == 2.0


def test_equation_subtraction():
    """ddt(phi) - div(face_fluxes, phi) negates the spatial op coefficient on a copy."""
    mesh, box, dm, geom = _make_mesh()
    phi = CellField(mesh, ncomp=1, ngrow=1, name="phi")

    def vel(x, y, z, t):
        return np.ones_like(x), np.zeros_like(x), np.zeros_like(x)

    div_op = exp.div(build_face_fluxes(vel, box, dm, geom, ngrow=1, t=0.0), phi)
    eqn = exp.ddt(phi) - div_op
    assert isinstance(eqn, Equation)
    assert len(eqn.spatial_ops) == 1
    assert eqn.spatial_ops[0].coeff == -1.0
    assert div_op.coeff == 1.0  # original term is not mutated


def test_solve_constant_field_unchanged():
    """Solving ddt(phi) + div(U=0, phi) = 0 leaves a constant field unchanged."""
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


def test_equation_with_named_schemes_solves():
    """Equation(terms, schemes={names}) resolves schemes via the registry in .solve()."""
    mesh, box, dm, geom = _make_mesh(n_cell=32, max_size=32)
    T = CellField(mesh, ncomp=1, ngrow=1, name="T")

    for mfi in blockamr.MFIterator(T.mf[0]):
        arr = T.mf[0].copy_to_host(mfi)
        arr[:, :, :, 0] = 5.0
        T.mf[0].copy_from(mfi, arr)

    def zero_vel(x, y, z, t):
        return np.zeros_like(x), np.zeros_like(x), np.zeros_like(x)

    ff = build_face_fluxes(zero_vel, box, dm, geom, ngrow=1, t=0.0)
    ff.name = "phi"

    div_term = exp.div(ff, T)
    eqn = Equation(exp.ddt(T) + div_term, schemes={"ddt": "Euler", "div(phi,T)": "linear"})
    eqn.solve(t=0.0, dt=0.01)

    assert isinstance(div_term.scheme, Linear)  # name resolved at discretise time
    for mfi in blockamr.MFIterator(T.mf[0]):
        arr = T.mf[0].copy_to_host(mfi)
        assert np.allclose(arr[:, :, :, 0], 5.0)


def test_diffusion_single_step():
    """One forward-Euler step of ddt(phi) - laplacian(1, phi) = 0.

    Verify: phi_new = phi_old + dt * laplacian(phi_old).
    """
    n_cell = 32
    mesh, box, dm, geom = _make_mesh(n_cell=n_cell, max_size=n_cell)
    phi = CellField(mesh, ncomp=1, ngrow=1, name="phi")
    _init_sin3d(phi, geom)

    phi_old = {}
    for mfi in blockamr.MFIterator(phi.mf[0]):
        bx = mfi.valid_box()
        lo = bx.small_end()
        phi_old[tuple(lo)] = phi.mf[0].copy_to_host(mfi)[:, :, :, 0].copy()

    def gamma_one(x, y, z, t):
        return np.ones_like(x)

    dt = 1e-5
    expr = exp.ddt(phi) - exp.laplacian(gamma_one, phi)
    solve(expr, t=0.0, dt=dt)

    pi = math.pi
    decay = 1.0 + dt * (-12.0 * pi**2)

    for mfi in blockamr.MFIterator(phi.mf[0]):
        bx = mfi.valid_box()
        lo = bx.small_end()
        arr_new = phi.mf[0].copy_to_host(mfi)[:, :, :, 0]
        arr_old = phi_old[tuple(lo)]
        expected = arr_old * decay
        assert np.allclose(arr_new, expected, atol=1e-4), (
            f"Max diff: {np.abs(arr_new - expected).max()}"
        )
