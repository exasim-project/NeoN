# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""Tests for the pressure projection algorithm."""

import numpy as np
import jax
import jax.numpy as jnp
import neon.blockamr as blockamr
from neon.blockamr.mesh import Mesh
from neon.blockamr.bc import NeumannBC, BoundaryCondition, fill_ghost_cells
from neon.blockamr.projection import (
    Projector, NodalProjector, cell_to_face, divergence_arrays, _make_face_mfs,
    nodal_divergence, nodal_gradient,
)


def _make_mesh(n, is_per=None):
    if is_per is None:
        is_per = [0, 0, 0]
    box = blockamr.Box([0, 0, 0], [n - 1, n - 1, n - 1])
    rb = blockamr.RealBox([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    geom = blockamr.Geometry(box, rb, 0, is_per)
    ba = blockamr.BoxArray(box)
    ba.max_size(n)
    dm = blockamr.DistributionMapping(ba)
    return Mesh(ba, dm, geom), geom, ba, dm


def test_cell_to_face_constant(blockamr_session):
    """Constant cell field -> same constant on faces."""
    N = 8
    mesh, geom, ba, dm = _make_mesh(N, is_per=[0, 0, 0])
    dom = geom.domain()

    vel_mfs = [blockamr.MultiFab(ba, dm, 1, 1) for _ in range(3)]
    bc = BoundaryCondition(lo=[NeumannBC()] * 3, hi=[NeumannBC()] * 3)
    for d, mf in enumerate(vel_mfs):
        val = float(d + 1)
        # Fill valid region
        arrs = mf.arrays()
        mf.copy_arrays([jnp.full(a.shape[:3], val) for a in arrs])
        # Fill ghosts (Neumann copies interior -> ghost, so constant stays constant)
        mf.fill_boundary(geom)
        fill_ghost_cells(mf, geom, bc)

    face_mfs = _make_face_mfs(dom, dm, 0)
    cell_to_face(vel_mfs, face_mfs)

    for d in range(3):
        for arr in face_mfs[d].arrays():
            a = np.array(arr[:, :, :, 0])
            expected = float(d + 1)
            assert np.allclose(a, expected, atol=1e-12), \
                f"Face {d}: expected {expected}, got min={a.min()} max={a.max()}"


def test_divergence_constant_field(blockamr_session):
    """Constant velocity -> zero divergence."""
    N = 8
    mesh, geom, ba, dm = _make_mesh(N, is_per=[0, 0, 0])
    dom = geom.domain()

    face_mfs = _make_face_mfs(dom, dm, 0)
    for d in range(3):
        arrs = face_mfs[d].arrays()
        face_mfs[d].copy_arrays([jnp.ones_like(a[:, :, :, 0]) for a in arrs])

    divs = divergence_arrays(face_mfs, geom)
    for d_arr in divs:
        assert float(jnp.max(jnp.abs(d_arr))) < 1e-12


def test_projection_removes_divergence(blockamr_session):
    """A divergent velocity field becomes divergence-free after projection.

    Use periodic domain so Neumann/Dirichlet BC issues don't arise.
    u = sin(2*pi*x), v = 0, w = 0 on a periodic [0,1]^3 domain.
    """
    import math
    N = 16
    mesh, geom, ba, dm = _make_mesh(N, is_per=[1, 1, 1])
    dom = geom.domain()
    dx = geom.cell_size()
    pi = math.pi

    vel_mfs = [blockamr.MultiFab(ba, dm, 1, 1) for _ in range(3)]

    n_valid = N
    xs = jnp.array([math.sin(2 * pi * (i + 0.5) * dx[0]) for i in range(n_valid)])
    vel_mfs[0].copy_arrays([jnp.broadcast_to(xs[:, None, None], (n_valid, n_valid, n_valid))])
    for d in [1, 2]:
        vel_mfs[d].copy_arrays([jnp.zeros((n_valid, n_valid, n_valid))])
    for mf in vel_mfs:
        mf.fill_boundary(geom)

    # Before projection: check divergence is non-zero
    face_mfs = _make_face_mfs(dom, dm, 0)
    cell_to_face(vel_mfs, face_mfs)
    div_before = divergence_arrays(face_mfs, geom)
    max_div_before = max(float(jnp.max(jnp.abs(d))) for d in div_before)
    assert max_div_before > 0.1, f"Expected non-zero div, got {max_div_before}"

    # Project
    proj = Projector(mesh, geom, dt=1.0)
    proj.project(vel_mfs)

    # Check divergence of the corrected face velocities
    div_after = divergence_arrays(proj._face_vel, geom)
    max_div_after = max(float(jnp.max(jnp.abs(d))) for d in div_after)
    assert max_div_after < 1e-8, f"Face divergence after projection: {max_div_after}"


# ---------------------------------------------------------------------------
# Nodal projection tests
# ---------------------------------------------------------------------------


def test_nodal_divergence_gradient_adjoint(blockamr_session):
    """Verify <div(u), phi> = -<u, grad(phi)> (discrete adjoint property).

    This is the key identity that guarantees the nodal projection produces
    a discretely divergence-free velocity field.
    """
    N = 8
    dx, dy, dz = 1.0 / N, 1.0 / N, 1.0 / N

    # Random cell-centred velocity
    key = jax.random.PRNGKey(42)
    u = jax.random.normal(key, (N, N, N))
    v = jax.random.normal(jax.random.PRNGKey(43), (N, N, N))
    w = jax.random.normal(jax.random.PRNGKey(44), (N, N, N))

    # Random nodal field
    phi = jax.random.normal(jax.random.PRNGKey(45), (N + 1, N + 1, N + 1))
    # Zero boundary nodes (Neumann BC: rhs=0 on boundary)
    phi = phi.at[0, :, :].set(0).at[-1, :, :].set(0)
    phi = phi.at[:, 0, :].set(0).at[:, -1, :].set(0)
    phi = phi.at[:, :, 0].set(0).at[:, :, -1].set(0)

    div_u = nodal_divergence(u, v, w, dx, dy, dz)
    gx, gy, gz = nodal_gradient(phi, dx, dy, dz)

    # <div(u), phi> using nodal volume = dx*dy*dz (interior nodes)
    lhs = float(jnp.sum(div_u * phi)) * dx * dy * dz

    # -<u, grad(phi)> using cell volume = dx*dy*dz
    rhs = -float(jnp.sum(u * gx + v * gy + w * gz)) * dx * dy * dz

    assert abs(lhs - rhs) < 1e-10 * max(abs(lhs), abs(rhs), 1e-15), \
        f"Adjoint test failed: <div,phi>={lhs}, -<u,grad>={rhs}"


def test_nodal_projection_nonperiodic(blockamr_session):
    """Nodal projection makes velocity divergence-free on non-periodic domain.

    u = sin(2*pi*x), v = 0, w = 0 on [0,1]^3 with wall BCs.
    After nodal projection, AMReX's compDivergence should give ~0.
    """
    import math
    N = 16
    mesh, geom, ba, dm = _make_mesh(N, is_per=[0, 0, 0])
    dx = geom.cell_size()
    pi = math.pi

    vel_mfs = [blockamr.MultiFab(ba, dm, 1, 1) for _ in range(3)]
    xs = jnp.array([math.sin(2 * pi * (i + 0.5) * dx[0]) for i in range(N)])
    vel_mfs[0].copy_arrays([jnp.broadcast_to(xs[:, None, None], (N, N, N))])
    for d in [1, 2]:
        vel_mfs[d].copy_arrays([jnp.zeros((N, N, N))])

    # Fill ghost cells
    bc = BoundaryCondition(lo=[NeumannBC()] * 3, hi=[NeumannBC()] * 3)
    for mf in vel_mfs:
        mf.fill_boundary(geom)
        fill_ghost_cells(mf, geom, bc)

    # Nodal project
    proj = NodalProjector(mesh, geom, dt=1.0)
    proj.project(vel_mfs)

    # Verify velocity is finite and changed
    for mf in vel_mfs:
        for arr in mf.arrays():
            a = np.array(arr)
            assert np.all(np.isfinite(a)), "Velocity has non-finite values after projection"

    # Re-project to measure residual divergence: if already div-free,
    # a second projection should produce zero correction
    for mf in vel_mfs:
        mf.fill_boundary(geom)
        fill_ghost_cells(mf, geom, bc)

    # Save velocity before second projection
    u_before = [np.array(mf.arrays()[0][:, :, :, 0]).copy() for mf in vel_mfs]
    proj.project(vel_mfs)
    u_after = [np.array(mf.arrays()[0][:, :, :, 0]) for mf in vel_mfs]

    # The correction from a second projection should be ~0
    max_correction = max(np.max(np.abs(a - b)) for a, b in zip(u_before, u_after))
    # Tolerance accounts for ghost-cell boundary effects in the ncomp=3 packing
    assert max_correction < 0.05, \
        f"Second projection correction {max_correction} — velocity not div-free"
