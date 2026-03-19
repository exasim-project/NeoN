# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""End-to-end SingleVortex advection test.

Solves ddt(phi) + div(U, phi) = 0 with a time-reversing vortex velocity
on a periodic domain. After one full period T, the solution should return
close to its initial condition.

Uses OpenFOAM-style loop: face fluxes are updated each step so the
time-dependent velocity is captured correctly.
"""

import math

import blockamr
from blockamr.field import Field
from blockamr.dsl import exp, solve
from blockamr.operators.div import build_face_fluxes, update_face_fluxes

import numpy as np


def _vortex_velocity(x, y, z, t, period=2.0):
    """2D SingleVortex velocity field (uniform in z), divergence-free.

    u =  2 * sin^2(pi*x) * sin(2*pi*y) * cos(pi*t/T)
    v = -2 * sin(2*pi*x) * sin^2(pi*y) * cos(pi*t/T)
    w = 0
    """
    cos_t = math.cos(math.pi * t / period)
    u = 2.0 * np.sin(np.pi * x) ** 2 * np.sin(2.0 * np.pi * y) * cos_t
    v = -2.0 * np.sin(2.0 * np.pi * x) * np.sin(np.pi * y) ** 2 * cos_t
    w = np.zeros_like(x)
    return u, v, w


def test_single_vortex_advection():
    """Advect a Gaussian with the SingleVortex field for one full period."""
    n_cell = 64
    max_size = 32
    ngrow = 1
    period = 2.0
    cfl = 0.3
    sigma = 0.1

    box = blockamr.Box([0, 0, 0], [n_cell - 1, n_cell - 1, n_cell - 1])
    rb = blockamr.RealBox([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    geom = blockamr.Geometry(box, rb, 0, [1, 1, 1])

    ba = blockamr.BoxArray(box)
    ba.max_size(max_size)
    dm = blockamr.DistributionMapping(ba)
    mf = blockamr.MultiFab(ba, dm, 1, ngrow)

    field = Field(mf, geom, name="phi", box=box, dm=dm, max_size=max_size)
    dx = field.dx

    for mfi in blockamr.MFIterator(mf):
        arr = mf.array(mfi)
        bx = mfi.valid_box()
        lo = bx.small_end()
        nx, ny, nz = arr.shape[:3]
        for i in range(nx):
            for j in range(ny):
                x = (lo[0] + i + 0.5) * dx[0]
                y = (lo[1] + j + 0.5) * dx[1]
                val = math.exp(-((x - 0.5) ** 2 + (y - 0.75) ** 2) / (2 * sigma**2))
                arr[i, j, :, 0] = val

    phi0 = {}
    for mfi in blockamr.MFIterator(mf):
        arr = mf.array(mfi)
        bx = mfi.valid_box()
        lo = tuple(bx.small_end())
        phi0[lo] = np.array(arr[:, :, :, 0], copy=True)

    def vel(x, y, z, t):
        return _vortex_velocity(x, y, z, t, period=period)

    # OpenFOAM-style: build face fluxes, update each step
    face_fluxes = build_face_fluxes(vel, box, dm, geom, ngrow=ngrow, t=0.0,
                                    max_size=max_size)

    u_max = 2.0
    dt = cfl * min(dx) / u_max
    t = 0.0
    nsteps = 0

    while t < period - 1e-12:
        if t + dt > period:
            dt = period - t
        update_face_fluxes(face_fluxes, vel, geom, t)
        expr = exp.ddt(field) + exp.div(face_fluxes, field)
        solve(expr, t, dt)
        t += dt
        nsteps += 1

    l2_err_sq = 0.0
    l2_norm_sq = 0.0
    for mfi in blockamr.MFIterator(mf):
        arr = mf.array(mfi)
        bx = mfi.valid_box()
        lo = tuple(bx.small_end())
        diff = arr[:, :, :, 0] - phi0[lo]
        l2_err_sq += np.sum(diff**2) * dx[0] * dx[1] * dx[2]
        l2_norm_sq += np.sum(phi0[lo] ** 2) * dx[0] * dx[1] * dx[2]

    l2_error = math.sqrt(l2_err_sq / l2_norm_sq)
    assert l2_error < 1.0, f"L2 error too large: {l2_error}"


def test_conservation():
    """Total mass should be conserved (up to roundoff) during advection."""
    n_cell = 32
    max_size = 16
    ngrow = 1
    cfl = 0.3

    box = blockamr.Box([0, 0, 0], [n_cell - 1, n_cell - 1, n_cell - 1])
    rb = blockamr.RealBox([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    geom = blockamr.Geometry(box, rb, 0, [1, 1, 1])

    ba = blockamr.BoxArray(box)
    ba.max_size(max_size)
    dm = blockamr.DistributionMapping(ba)
    mf = blockamr.MultiFab(ba, dm, 1, ngrow)

    field = Field(mf, geom, name="phi", box=box, dm=dm, max_size=max_size)
    dx = field.dx

    for mfi in blockamr.MFIterator(mf):
        arr = mf.array(mfi)
        bx = mfi.valid_box()
        lo = bx.small_end()
        nx, ny, nz = arr.shape[:3]
        for i in range(nx):
            for j in range(ny):
                x = (lo[0] + i + 0.5) * dx[0]
                y = (lo[1] + j + 0.5) * dx[1]
                arr[i, j, :, 0] = math.exp(-((x - 0.5) ** 2 + (y - 0.5) ** 2) / 0.02)

    def compute_mass():
        total = 0.0
        for mfi in blockamr.MFIterator(mf):
            arr = mf.array(mfi)
            total += np.sum(arr[:, :, :, 0]) * dx[0] * dx[1] * dx[2]
        return total

    mass0 = compute_mass()

    def vel(x, y, z, t):
        return _vortex_velocity(x, y, z, t, period=2.0)

    face_fluxes = build_face_fluxes(vel, box, dm, geom, ngrow=ngrow, t=0.0,
                                    max_size=max_size)

    dt = cfl * min(dx) / 2.0
    t = 0.0
    for _ in range(20):
        update_face_fluxes(face_fluxes, vel, geom, t)
        expr = exp.ddt(field) + exp.div(face_fluxes, field)
        solve(expr, t, dt)
        t += dt

    mass_final = compute_mass()
    rel_error = abs(mass_final - mass0) / abs(mass0)
    assert rel_error < 1e-10, f"Mass not conserved: {rel_error}"
