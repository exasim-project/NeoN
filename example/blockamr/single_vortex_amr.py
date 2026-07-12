# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""Multi-level AMR advection of a Gaussian in a time-reversing vortex.

Same physics as single_vortex.py but with 2-level AMR. Fine level tracks
the Gaussian via gradient-based tagging. All levels advance with the same
global dt (no subcycling). average_down synchronises after each step.

Usage:
    python example/blockamr/single_vortex_amr.py
    python example/blockamr/single_vortex_amr.py --ncell 64 --max-level 2
"""

import argparse
import os

os.environ.setdefault("AMREX_THE_ARENA_INIT_SIZE", "0")

import jax.numpy as jnp
import numpy as np

import neon.blockamr as blockamr
from neon.blockamr.mesh import AmrMesh
from neon.blockamr.field import CellField, FaceField
from neon.blockamr.fillpatch import FillPatchCellConservative
from neon.blockamr.dsl import exp, solve
from neon.blockamr.operators.div import AmrFaceFluxUpdater, Div
from neon.blockamr.schemes.div_schemes import VanLeer


def vortex_velocity(x, y, z, t, period=2.0):
    cos_t = jnp.cos(jnp.pi * t / period)
    u = 2.0 * jnp.sin(jnp.pi * x) ** 2 * jnp.sin(2.0 * jnp.pi * y) * cos_t
    v = -2.0 * jnp.sin(2.0 * jnp.pi * x) * jnp.sin(jnp.pi * y) ** 2 * cos_t
    w = jnp.zeros_like(x)
    return u, v, w


def init_gaussian(mf, geom, center=(0.5, 0.75), sigma=0.1):
    """Fill a MultiFab with a 2-D Gaussian (uniform in z)."""
    dx = geom.cell_size()
    cx, cy = center
    for mfi in blockamr.MFIterator(mf):
        bx = mfi.valid_box()
        lo = bx.small_end()
        hi = bx.big_end()
        nx, ny, nz = hi[0] - lo[0] + 1, hi[1] - lo[1] + 1, hi[2] - lo[2] + 1
        xs = jnp.array([(lo[0] + i + 0.5) * dx[0] for i in range(nx)])
        ys = jnp.array([(lo[1] + j + 0.5) * dx[1] for j in range(ny)])
        vals = jnp.exp(-((xs[:, None] - cx) ** 2 + (ys[None, :] - cy) ** 2) / (2.0 * sigma**2))
        mf.copy_from(mfi, vals[:, :, None] * jnp.ones((nx, ny, nz)))


def tag_gradient(phi, threshold=1.5):
    """Return a tagging function based on gradient of phi (GPU-resident)."""

    def _tag(lev, tags, time, ngrow):
        if phi.mf[lev] is None:
            return
        dx = phi.mesh.geom(lev).cell_size()
        ng = phi.mf[lev].n_grow()
        for mfi in blockamr.MFIterator(phi.mf[lev]):
            phi_4d = phi.mf[lev].array(mfi)
            data = phi_4d[:, :, :, 0]
            # Compute gradient on valid region (set_tags expects valid-sized mask)
            bx = mfi.valid_box()
            lo = bx.small_end()
            hi = bx.big_end()
            vn = [hi[d] - lo[d] + 1 for d in range(3)]
            gx = jnp.abs(
                data[ng + 1 : ng + 1 + vn[0], ng : ng + vn[1], ng : ng + vn[2]]
                - data[ng - 1 : ng - 1 + vn[0], ng : ng + vn[1], ng : ng + vn[2]]
            ) / (2 * dx[0])
            gy = jnp.abs(
                data[ng : ng + vn[0], ng + 1 : ng + 1 + vn[1], ng : ng + vn[2]]
                - data[ng : ng + vn[0], ng - 1 : ng - 1 + vn[1], ng : ng + vn[2]]
            ) / (2 * dx[1])
            mask = ((gx + gy) > threshold).astype(jnp.int32)
            tags.set_tags(mfi, mask)

    return _tag


def compute_level0_mass(phi, mesh):
    """Mass integral over level 0 (full domain after average_down)."""
    dx = mesh.geom(0).cell_size()
    dv = dx[0] * dx[1] * dx[2]
    total = 0.0
    for mfi in blockamr.MFIterator(phi.mf[0]):
        arr = phi.mf[0].copy_to_host(mfi)
        total += float(np.sum(arr[:, :, :, 0])) * dv
    return total


def run(
    n_cell=32,
    max_level=1,
    cfl=0.3,
    period=2.0,
    scheme="Upwind",
    plotfile=True,
    write_interval=0.1,
    max_grid_size=32,
):
    box = blockamr.Box([0, 0, 0], [n_cell - 1, n_cell - 1, n_cell - 1])
    rb = blockamr.RealBox([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    geom = blockamr.Geometry(box, rb, 0, [1, 1, 1])

    info = blockamr.AmrInfo()
    info.max_level = max_level
    info.set_ref_ratio(0, 2)
    info.set_max_grid_size(0, max_grid_size)
    info.set_blocking_factor(0, 16)

    mesh = AmrMesh(geom, info)
    ngrow = 2
    phi = CellField(mesh, ncomp=1, ngrow=ngrow, name="phi", fill_patch=FillPatchCellConservative())
    face_vel = FaceField(mesh, ncomp=1, ngrow=ngrow, name="U")

    # --- initialise ---
    mesh.init_from_scratch(0.0)
    init_gaussian(phi.mf[0], mesh.geom(0))

    # regrid to create fine levels
    for lev in range(max_level + 1):
        mesh.regrid(0.0, tag=tag_gradient(phi, threshold=1.0))
        for lev in range(mesh.n_levels()):
            init_gaussian(phi.mf[lev], mesh.geom(lev))

        print(f"Levels: {mesh.n_levels()}, finest_level: {mesh.finest_level()}")

    dx_coarse = mesh.geom(0).cell_size()
    u_max = 2.0
    dt = cfl * min(dx_coarse) / u_max
    t = 0.0
    nsteps = 0

    mass0 = compute_level0_mass(phi, mesh)
    div_scheme = VanLeer()  # Upwind()

    def vel(x, y, z, t):
        return vortex_velocity(x, y, z, t, period=period)

    # write initial plotfile
    plot_count = 0
    next_write = write_interval
    if plotfile:
        mesh.write_plotfile(f"plt_vortex_amr_{plot_count:04d}", phi, 0.0)
        print(f"  Wrote plt_vortex_amr_{plot_count:04d}  (t = 0.0000)")
        plot_count += 1

    print(f"Grid: {n_cell}^3, dt = {dt:.4e}, write every {write_interval}s")
    print()

    # --- time loop (DSL identical to single-level) ---
    flux_updater = AmrFaceFluxUpdater(face_vel, vel, mesh)

    while t < (period - 1e-12):
        if t + dt > period:
            dt = period - t

        mesh.regrid(t, tag=tag_gradient(phi, threshold=1.0))
        flux_updater.update(t)

        expr = exp.ddt(phi) + Div(face_vel, phi, scheme=div_scheme)
        solve(expr, t=t, dt=dt)

        t += dt
        nsteps += 1

        if plotfile and t >= next_write - 1e-12:
            # mesh.write_plotfile(f"plt_vortex_amr_{plot_count:04d}", phi, t)
            print(f"  Wrote plt_vortex_amr_{plot_count:04d}  (t = {t:.4f})")
            plot_count += 1
            next_write += write_interval

    mass_final = compute_level0_mass(phi, mesh)
    mass_err = abs(mass_final - mass0) / abs(mass0)

    print()
    print(f"Completed {nsteps} steps, final t = {t:.6f}")
    print(f"Mass conservation error: {mass_err:.2e}")
    if plotfile:
        print(f"Wrote {plot_count} plotfiles")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ncell", type=int, default=32)
    parser.add_argument("--max-level", type=int, default=2)
    parser.add_argument("--cfl", type=float, default=0.3)
    parser.add_argument("--period", type=float, default=2.0)
    parser.add_argument("--max-size", type=int, default=32, help="max_grid_size for AMR regridding")
    parser.add_argument(
        "--write-interval", type=float, default=0.1, help="plotfile write interval in seconds"
    )
    parser.add_argument("--no-plot", action="store_true", help="skip plotfile output")
    parser.add_argument(
        "--backend", choices=["jax", "pallas", "triton"], default="jax", help="dispatch backend"
    )
    args = parser.parse_args()

    with blockamr.runtime():
        blockamr.set_backend(args.backend)
        run(
            n_cell=args.ncell,
            max_level=args.max_level,
            cfl=args.cfl,
            period=args.period,
            plotfile=not args.no_plot,
            write_interval=args.write_interval,
            max_grid_size=args.max_size,
        )
