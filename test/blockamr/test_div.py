# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

import math

import neon.blockamr as blockamr
import jax
import jax.numpy as jnp
import numpy as np
from neon.blockamr.field import CellField, FaceField
from neon.blockamr.mesh import Mesh
from neon.blockamr.operators.div import Div, build_face_fluxes, _fill_face_component


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


def _init_sin3d(phi, mesh):
    """Set field to sin(2*pi*x)*sin(2*pi*y)*sin(2*pi*z)."""
    dx = mesh.geom(0).cell_size()
    for mfi in blockamr.MFIterator(phi.mf[0]):
        bx = mfi.valid_box()
        lo = bx.small_end()
        hi = bx.big_end()
        nx, ny, nz = hi[0] - lo[0] + 1, hi[1] - lo[1] + 1, hi[2] - lo[2] + 1
        xs = jnp.array([(lo[0] + i + 0.5) * dx[0] for i in range(nx)])
        ys = jnp.array([(lo[1] + j + 0.5) * dx[1] for j in range(ny)])
        zs = jnp.array([(lo[2] + k + 0.5) * dx[2] for k in range(nz)])
        X, Y, Z = jnp.meshgrid(xs, ys, zs, indexing="ij")
        vals = jnp.sin(2 * jnp.pi * X) * jnp.sin(2 * jnp.pi * Y) * jnp.sin(2 * jnp.pi * Z)
        phi.mf[0].copy_from(mfi, vals)
    phi.fill_patch(0, 0.0)


def test_div_uniform_field_is_zero():
    """Divergence of a uniform field should be zero."""
    mesh, box, dm, geom = _make_mesh(n_cell=64, max_size=32)
    phi = CellField(mesh, ncomp=1, ngrow=1, name="phi")

    for mfi in blockamr.MFIterator(phi.mf[0]):
        bx = mfi.valid_box()
        lo = bx.small_end()
        hi = bx.big_end()
        nx, ny, nz = hi[0] - lo[0] + 1, hi[1] - lo[1] + 1, hi[2] - lo[2] + 1
        phi.mf[0].copy_from(mfi, jnp.ones((nx, ny, nz)))
    phi.fill_patch(0, 0.0)

    def uniform_vel(x, y, z, t):
        return jnp.ones_like(x), jnp.ones_like(x), jnp.ones_like(x)

    ff = build_face_fluxes(uniform_vel, box, dm, geom, ngrow=1, t=0.0)
    div_op = Div(ff, phi)

    for mfi in blockamr.MFIterator(phi.mf[0]):
        phi_arr = phi.mf[0].grown_array(mfi)
        kernel = div_op.build_kernel(mfi, t=0.0)
        result = kernel(phi_arr)
        assert jnp.allclose(result, 0.0, atol=1e-12), f"max div = {jnp.abs(result).max()}"


def test_div_sin_field():
    """Divergence of sin(2*pi*x) with u=1 approximates 2*pi*cos(2*pi*x)."""
    mesh, box, dm, geom = _make_mesh(n_cell=64, max_size=32)
    phi = CellField(mesh, ncomp=1, ngrow=1, name="phi")
    dx = geom.cell_size()

    for mfi in blockamr.MFIterator(phi.mf[0]):
        bx = mfi.valid_box()
        lo = bx.small_end()
        hi = bx.big_end()
        nx, ny, nz = hi[0] - lo[0] + 1, hi[1] - lo[1] + 1, hi[2] - lo[2] + 1
        xs = jnp.array([(lo[0] + i + 0.5) * dx[0] for i in range(nx)])
        vals = jnp.sin(2 * jnp.pi * xs)
        phi.mf[0].copy_from(mfi, (vals[:, None, None] * jnp.ones((nx, ny, nz))))
    phi.fill_patch(0, 0.0)

    def x_vel(x, y, z, t):
        return jnp.ones_like(x), jnp.zeros_like(x), jnp.zeros_like(x)

    ff = build_face_fluxes(x_vel, box, dm, geom, ngrow=1, t=0.0)
    div_op = Div(ff, phi)

    for mfi in blockamr.MFIterator(phi.mf[0]):
        phi_arr = phi.mf[0].grown_array(mfi)
        kernel = div_op.build_kernel(mfi, t=0.0)
        result = kernel(phi_arr)
        lo = mfi.valid_box().small_end()
        nx = result.shape[0]
        for i in range(nx):
            x = (lo[0] + i + 0.5) * dx[0]
            analytic = 2 * math.pi * math.cos(2 * math.pi * x)
            assert abs(float(result[i, 0, 0]) - analytic) < 0.6, (
                f"At x={x:.3f}: got {float(result[i, 0, 0]):.4f}, expected {analytic:.4f}"
            )


def _compute_div_error(n_cell):
    """Compute max error of div(U*phi) with U=(1,0,0) vs analytical on sin3d."""
    mesh, box, dm, geom = _make_mesh(n_cell=n_cell, max_size=n_cell)
    phi = CellField(mesh, ncomp=1, ngrow=1, name="phi")
    _init_sin3d(phi, mesh)

    def x_vel(x, y, z, t):
        return jnp.ones_like(x), jnp.zeros_like(x), jnp.zeros_like(x)

    ff = build_face_fluxes(x_vel, box, dm, geom, ngrow=1, t=0.0, max_size=n_cell)
    div_op = Div(ff, phi)
    pi = math.pi

    max_err = 0.0
    for mfi in blockamr.MFIterator(phi.mf[0]):
        phi_arr = phi.mf[0].grown_array(mfi)
        kernel = div_op.build_kernel(mfi, t=0.0)
        result = kernel(phi_arr)
        lo = mfi.valid_box().small_end()
        dx = geom.cell_size()
        prob_lo = geom.prob_lo()
        nx, ny, nz = result.shape[:3]
        xs = jnp.array([prob_lo[0] + (lo[0] + i + 0.5) * dx[0] for i in range(nx)])
        ys = jnp.array([prob_lo[1] + (lo[1] + j + 0.5) * dx[1] for j in range(ny)])
        zs = jnp.array([prob_lo[2] + (lo[2] + k + 0.5) * dx[2] for k in range(nz)])
        X, Y, Z = jnp.meshgrid(xs, ys, zs, indexing="ij")
        exact = 2 * pi * jnp.cos(2 * pi * X) * jnp.sin(2 * pi * Y) * jnp.sin(2 * pi * Z)
        err = float(jnp.max(jnp.abs(result - exact)))
        max_err = max(max_err, err)
    return max_err


def test_div_sin_convergence():
    """First-order upwind div converges at O(dx) on sin3d with U=(1,0,0).

    Analytical: div(U*phi) = dphi/dx = 2*pi*cos(2*pi*x)*sin(2*pi*y)*sin(2*pi*z).
    Error ratio should be ~2 (first-order).
    """
    errors = []
    for n in [16, 32, 64]:
        err = _compute_div_error(n)
        errors.append(err)

    ratio_1 = errors[0] / errors[1]
    ratio_2 = errors[1] / errors[2]
    assert ratio_1 > 1.8, f"Ratio 16->32: {ratio_1:.2f}, expected ~2"
    assert ratio_2 > 1.8, f"Ratio 32->64: {ratio_2:.2f}, expected ~2"


def test_fill_face_component_passes_jax_arrays():
    """_fill_face_component should pass JAX arrays to vel_func (GPU-ready path)."""
    n_cell = 32
    box = blockamr.Box([0, 0, 0], [n_cell - 1, n_cell - 1, n_cell - 1])
    rb = blockamr.RealBox([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    geom = blockamr.Geometry(box, rb, 0, [1, 1, 1])
    ba = blockamr.BoxArray(box)
    ba.max_size(32)
    dm = blockamr.DistributionMapping(ba)

    mesh = Mesh(ba, dm, geom)
    ff = FaceField(mesh, ncomp=1, ngrow=1)

    received_types = []

    def spy_vel(x, y, z, t):
        received_types.append(type(x))
        return jnp.ones_like(x), jnp.zeros_like(x), jnp.zeros_like(x)

    dx = geom.cell_size()
    prob_lo = geom.prob_lo()
    _fill_face_component(ff[0][0], 0, spy_vel, dx, prob_lo, 0.0)

    assert len(received_types) > 0, "vel_func was never called"
    for tp in received_types:
        assert issubclass(tp, jax.Array), (
            f"vel_func received {tp.__name__}, expected jax.Array"
        )
