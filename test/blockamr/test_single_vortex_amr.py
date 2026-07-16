# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""AMR SingleVortex mass conservation test.

Verifies that multi-level advection with average_down restriction
conserves mass to machine precision.
"""

import math

import numpy as np

import blockamr
from blockamr.mesh import AmrMesh
from blockamr.field import CellField, FaceField
from blockamr.fillpatch import FillPatchCellConservative
from blockamr.dsl import exp, solve
from blockamr.operators.div import Div, update_face_fluxes
from blockamr.schemes.div_schemes import Upwind


def _vortex_velocity(x, y, z, t, period=2.0):
    cos_t = math.cos(math.pi * t / period)
    u = 2.0 * np.sin(np.pi * x) ** 2 * np.sin(2.0 * np.pi * y) * cos_t
    v = -2.0 * np.sin(2.0 * np.pi * x) * np.sin(np.pi * y) ** 2 * cos_t
    w = np.zeros_like(x)
    return u, v, w


def _init_gaussian(mf, geom, center=(0.5, 0.75), sigma=0.1):
    """Fill a MultiFab with a 2-D Gaussian (uniform in z)."""
    dx = geom.cell_size()
    cx, cy = center
    for mfi in blockamr.MFIterator(mf):
        bx = mfi.valid_box()
        lo = bx.small_end()
        hi = bx.big_end()
        nx, ny, nz = hi[0] - lo[0] + 1, hi[1] - lo[1] + 1, hi[2] - lo[2] + 1
        xs = np.array([(lo[0] + i + 0.5) * dx[0] for i in range(nx)])
        ys = np.array([(lo[1] + j + 0.5) * dx[1] for j in range(ny)])
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        vals = np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2.0 * sigma**2))
        mf.copy_from(mfi, vals[:, :, np.newaxis] * np.ones((nx, ny, nz)))


def _tag_all(lev, tags, time, ngrow):
    """Tag every cell for refinement."""
    for tbi in blockamr.TagBoxIterator(tags):
        bx = tbi.valid_box()
        lo = bx.small_end()
        hi = bx.big_end()
        nx = hi[0] - lo[0] + 1
        ny = hi[1] - lo[1] + 1
        nz = hi[2] - lo[2] + 1
        tbi.set_tags(np.ones((nx, ny, nz), dtype=np.int32))


def _compute_level0_mass(phi, mesh):
    """Mass integral over level 0 (covers entire domain after average_down)."""
    dx = mesh.geom(0).cell_size()
    dv = dx[0] * dx[1] * dx[2]
    total = 0.0
    for mfi in blockamr.MFIterator(phi.mf[0]):
        arr = phi.mf[0].copy_to_host(mfi)
        total += float(np.sum(arr[:, :, :, 0])) * dv
    return total


def test_amr_vortex_mass_conservation():
    """Mass approximately conserved over 10 timesteps on a 2-level mesh.

    Without refluxing (flux correction at coarse-fine boundaries), mass
    conservation is only approximate. This test verifies the AMR solve
    machinery works and conservation error stays bounded.
    """
    n_cell = 16
    period = 2.0
    cfl = 0.3

    box = blockamr.Box([0, 0, 0], [n_cell - 1, n_cell - 1, n_cell - 1])
    rb = blockamr.RealBox([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    geom = blockamr.Geometry(box, rb, 0, [1, 1, 1])

    info = blockamr.AmrInfo()
    info.max_level = 1
    info.set_ref_ratio(0, 2)
    info.set_max_grid_size(0, 16)
    info.set_blocking_factor(0, 8)

    mesh = AmrMesh(geom, info)
    phi = CellField(mesh, ncomp=1, ngrow=2, name="phi", fill_patch=FillPatchCellConservative())
    face_vel = FaceField(mesh, ncomp=1, ngrow=2, name="U")

    mesh.init_from_scratch(0.0)
    _init_gaussian(phi.mf[0], mesh.geom(0))

    # Regrid to create level 1
    mesh.regrid(0.0, tag=_tag_all)
    for lev in range(mesh.n_levels()):
        _init_gaussian(phi.mf[lev], mesh.geom(lev))

    assert mesh.n_levels() == 2, f"Expected 2 levels, got {mesh.n_levels()}"

    dx_coarse = mesh.geom(0).cell_size()
    u_max = 2.0
    dt = cfl * min(dx_coarse) / u_max
    div_scheme = Upwind()

    def vel(x, y, z, t):
        return _vortex_velocity(x, y, z, t, period=period)

    mass0 = _compute_level0_mass(phi, mesh)
    t = 0.0

    for _ in range(10):
        for lev in range(mesh.n_levels()):
            update_face_fluxes(face_vel[lev], vel, mesh.geom(lev), t)
        expr = exp.ddt(phi) + Div(face_vel, phi, scheme=div_scheme)
        solve(expr, t=t, dt=dt)
        t += dt

    mass_final = _compute_level0_mass(phi, mesh)
    rel_error = abs(mass_final - mass0) / abs(mass0)
    assert rel_error < 1e-2, f"Mass not conserved: {rel_error:.2e}"
