# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""Tests for the interpolate operator: cell-centred U → face-centred φ."""

import jax.numpy as jnp
import numpy as np

import blockamr
from blockamr.field import CellField, FaceField
from blockamr.mesh import Mesh
from blockamr.operators.interpolate import interpolate


def _make_mesh(N, max_size=None):
    ms = max_size or N
    box = blockamr.Box([0, 0, 0], [N - 1, N - 1, N - 1])
    rb = blockamr.RealBox([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    geom = blockamr.Geometry(box, rb, 0, [1, 1, 1])
    ba = blockamr.BoxArray(box)
    ba.max_size(ms)
    dm = blockamr.DistributionMapping(ba)
    return Mesh(ba, dm, geom), geom


def _set_cellfield(field, geom, func):
    dx = geom.cell_size()
    prob_lo = geom.prob_lo()
    for mfi in blockamr.MFIterator(field.mf[0]):
        bx = mfi.valid_box()
        lo = bx.small_end()
        hi = bx.big_end()
        nx, ny, nz = hi[0]-lo[0]+1, hi[1]-lo[1]+1, hi[2]-lo[2]+1
        xs = jnp.array([prob_lo[0] + (lo[0]+i+0.5)*dx[0] for i in range(nx)])
        ys = jnp.array([prob_lo[1] + (lo[1]+j+0.5)*dx[1] for j in range(ny)])
        zs = jnp.array([prob_lo[2] + (lo[2]+k+0.5)*dx[2] for k in range(nz)])
        X, Y, Z = jnp.meshgrid(xs, ys, zs, indexing="ij")
        vals = func(X, Y, Z)
        field.mf[0].copy_from(mfi, vals)
    field.fill_patch(0, 0.0)


def test_interpolate_linear_field_exact(blockamr_session):
    """Linear interpolation of U_x = x is exact at interior face centres."""
    N = 16
    mesh, geom = _make_mesh(N)
    U = CellField(mesh, ncomp=3, ngrow=1, name="U")
    phi = FaceField(mesh, ncomp=1, ngrow=1, name="phi")

    def linear_x(X, Y, Z):
        return jnp.stack([X, jnp.zeros_like(X), jnp.zeros_like(X)], axis=-1)

    _set_cellfield(U, geom, linear_x)
    interpolate(U, phi)

    # Check: interior x-faces should be exact average of adjacent cells
    cell_grown = np.array(U.mf[0].grown_arrays()[0][:, :, :, 0])
    ng = U.ngrow
    face_mf = phi[0][0].mf
    for mfi in blockamr.MFIterator(face_mf):
        face_valid = np.array(face_mf.copy_to_host(mfi)[:, :, :, 0])
        break

    jy, jz = N // 2, N // 2
    for i in range(1, N):
        expected = 0.5 * (cell_grown[ng+i-1, ng+jy, ng+jz]
                          + cell_grown[ng+i, ng+jy, ng+jz])
        got = face_valid[i, jy, jz]
        assert abs(got - expected) < 1e-12, \
            f"Face {i}: got {got:.6f}, expected {expected:.6f}"


def test_interpolate_constant_field(blockamr_session):
    """Interpolate of constant U = (1, 2, 3) gives constant face fluxes."""
    N = 16
    mesh, geom = _make_mesh(N)
    U = CellField(mesh, ncomp=3, ngrow=1, name="U")
    phi = FaceField(mesh, ncomp=1, ngrow=1, name="phi")

    def const_vel(X, Y, Z):
        return jnp.stack([jnp.ones_like(X), 2*jnp.ones_like(X),
                          3*jnp.ones_like(X)], axis=-1)

    _set_cellfield(U, geom, const_vel)
    interpolate(U, phi)

    # x-faces should have flux = 1, y-faces = 2, z-faces = 3
    for d, expected_val in enumerate([1.0, 2.0, 3.0]):
        face_arr = phi[0][d].mf.arrays()[0]
        max_err = float(jnp.max(jnp.abs(face_arr[:, :, :, 0] - expected_val)))
        assert max_err < 1e-12, \
            f"Direction {d}: max_err={max_err}, expected constant {expected_val}"


def test_interpolate_multi_box(blockamr_session):
    """Interpolate works with multiple boxes."""
    N = 32
    mesh, geom = _make_mesh(N, max_size=16)
    U = CellField(mesh, ncomp=3, ngrow=1, name="U")
    phi = FaceField(mesh, ncomp=1, ngrow=1, name="phi")

    def const_vel(X, Y, Z):
        return jnp.stack([jnp.ones_like(X), jnp.zeros_like(X),
                          jnp.zeros_like(X)], axis=-1)

    _set_cellfield(U, geom, const_vel)
    interpolate(U, phi)

    # All x-face fluxes should be 1.0 across all boxes
    for arr in phi[0][0].mf.arrays():
        max_err = float(jnp.max(jnp.abs(arr[:, :, :, 0] - 1.0)))
        assert max_err < 1e-12, f"Multi-box x-face error: {max_err}"


def test_interpolate_analytical_multi_box(blockamr_session):
    """Interpolate sin(2πx) velocity on multi-box grid, verify against analytical."""
    import math
    N = 32
    mesh, geom = _make_mesh(N, max_size=8)
    U = CellField(mesh, ncomp=3, ngrow=1, name="U")
    phi = FaceField(mesh, ncomp=1, ngrow=1, name="phi")
    dx = geom.cell_size()

    def sin_vel(X, Y, Z):
        return jnp.stack([
            jnp.sin(2 * math.pi * X),
            jnp.zeros_like(X),
            jnp.zeros_like(X),
        ], axis=-1)

    _set_cellfield(U, geom, sin_vel)
    interpolate(U, phi)

    # Verify x-faces against analytical: face_x = 0.5*(sin(2π*x_lo) + sin(2π*x_hi))
    face_mf = phi[0][0].mf
    ng_face = face_mf.n_grow()
    max_err = 0.0
    for mfi in blockamr.MFIterator(face_mf):
        bx = mfi.valid_box()
        lo = bx.small_end()
        hi = bx.big_end()
        nfx = hi[0] - lo[0] + 1  # valid face count in x
        nfy = hi[1] - lo[1] + 1
        nfz = hi[2] - lo[2] + 1

        # Face centres in x: lo[0]*dx to (lo[0]+nfx-1)*dx (node positions)
        fx = jnp.array([(lo[0] + i) * dx[0] for i in range(nfx)])
        fy = jnp.array([(lo[1] + j + 0.5) * dx[1] for j in range(nfy)])
        fz = jnp.array([(lo[2] + k + 0.5) * dx[2] for k in range(nfz)])

        # Analytical: face_x at node x_face = 0.5*(sin(2π*(x_face-dx/2)) + sin(2π*(x_face+dx/2)))
        # = sin(2π*x_face)*cos(π*dx)  (trig identity)
        analytical = jnp.sin(2 * math.pi * fx)[:, None, None] * np.cos(math.pi * dx[0])

        valid = np.array(face_mf.copy_to_host(mfi)[:nfx, :nfy, :nfz, 0])
        err = np.max(np.abs(valid - np.array(analytical * jnp.ones((1, nfy, nfz)))))
        max_err = max(max_err, err)

    assert max_err < 1e-10, f"x-face interpolation error: {max_err:.2e}"

    # y-faces and z-faces should be 0 (v=0, w=0)
    for d in [1, 2]:
        face_d = phi[0][d].mf
        for mfi in blockamr.MFIterator(face_d):
            valid = np.array(face_d.copy_to_host(mfi))
            bx = mfi.valid_box()
            lo = bx.small_end(); hi = bx.big_end()
            vn = [hi[ax]-lo[ax]+1 for ax in range(3)]
            max_val = np.max(np.abs(valid[:vn[0], :vn[1], :vn[2], 0]))
            assert max_val < 1e-12, f"Face dir {d} should be 0, max={max_val}"
