# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""Single-level advection of a Gaussian profile in a time-reversing vortex.

Solves the advection equation on a periodic [0,1]^3 domain:

    ddt(phi) + div(U, phi) = 0

using the explicit DSL with configurable divergence scheme and forward-Euler
time stepping.  The prescribed velocity field is a 2-D divergence-free
vortex that reverses direction at t = T/2, so the Gaussian returns to its
initial position after one full period T.

Uses an OpenFOAM-style loop: the expression is assembled inside the time loop
with current velocity, so time-dependent fields are captured correctly.

Based on the AMReX AmrAdvection SingleVortex tutorial:
https://amrex-codes.github.io/amrex/docs_html/AmrCore.html#the-advection-equation

Usage:
    python example/blockAmr/single_vortex.py                    # default 64^3, Upwind
    python example/blockAmr/single_vortex.py --ncell 128        # finer grid
    python example/blockAmr/single_vortex.py --scheme Linear    # central scheme
    python example/blockAmr/single_vortex.py --device cpu       # CPU baseline
"""

import argparse
import math
import os
import shutil

# Parse --device early so env vars are set before JAX/AMReX import
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--device", choices=["cpu", "gpu"], default=None)
_early, _ = _pre.parse_known_args()
if _early.device == "cpu":
    os.environ["JAX_PLATFORMS"] = "cpu"
else:
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.25")
os.environ.setdefault("AMREX_THE_ARENA_INIT_SIZE", "0")

import jax.numpy as jnp
import numpy as np

import blockamr
from blockamr.field import CellField
from blockamr.mesh import Mesh
from blockamr.dsl import exp, solve
from blockamr.operators.div import build_face_fluxes, Div, FaceFluxUpdater
from blockamr.schemes.div_schemes import QUICK, Linear, Upwind, VanLeer

DIV_SCHEMES = {
    "Upwind": Upwind,
    "Linear": Linear,
    "VanLeer": VanLeer,
    "QUICK": QUICK,
}


# ---------------------------------------------------------------------------
# Velocity field (JAX-native — works on both CPU and GPU)
# ---------------------------------------------------------------------------
def vortex_velocity(x, y, z, t, period=2.0):
    """Divergence-free 2-D vortex (uniform in z) that reverses at t = T/2.

    u =  2 sin^2(pi x) sin(2 pi y) cos(pi t / T)
    v = -2 sin(2 pi x) sin^2(pi y) cos(pi t / T)
    w =  0
    """
    cos_t = jnp.cos(jnp.pi * t / period)
    u = 2.0 * jnp.sin(jnp.pi * x) ** 2 * jnp.sin(2.0 * jnp.pi * y) * cos_t
    v = -2.0 * jnp.sin(2.0 * jnp.pi * x) * jnp.sin(jnp.pi * y) ** 2 * cos_t
    w = jnp.zeros_like(x)
    return u, v, w


# ---------------------------------------------------------------------------
# Initial condition (JAX-native — works on both CPU and GPU)
# ---------------------------------------------------------------------------
def init_gaussian(phi, geom, center=(0.5, 0.75), sigma=0.1):
    """Fill *phi* with a 2-D Gaussian (uniform in z)."""
    dx = geom.cell_size()
    cx, cy = center
    for mfi in blockamr.MFIterator(phi.mf[0]):
        bx = mfi.valid_box()
        lo = bx.small_end()
        hi = bx.big_end()
        nx, ny, nz = hi[0] - lo[0] + 1, hi[1] - lo[1] + 1, hi[2] - lo[2] + 1
        xs = jnp.array([(lo[0] + i + 0.5) * dx[0] for i in range(nx)])
        ys = jnp.array([(lo[1] + j + 0.5) * dx[1] for j in range(ny)])
        vals = jnp.exp(-((xs[:, None] - cx) ** 2 + (ys[None, :] - cy) ** 2) / (2.0 * sigma**2))
        phi.mf[0].copy_from(mfi, vals[:, :, None] * jnp.ones((nx, ny, nz)))


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
def compute_mass(mf, dx):
    """Integrate phi over the domain."""
    total = 0.0
    dv = dx[0] * dx[1] * dx[2]
    for mfi in blockamr.MFIterator(mf):
        host = mf.copy_to_host(mfi)
        total += float(jnp.sum(host[:, :, :, 0])) * dv
    return total


def compute_l2_error(mf, phi0, dx):
    """Relative L2 error between current field and stored reference."""
    err_sq = 0.0
    ref_sq = 0.0
    dv = dx[0] * dx[1] * dx[2]
    for mfi in blockamr.MFIterator(mf):
        bx = mfi.valid_box()
        lo = bx.small_end()
        hi = bx.big_end()
        lo_key = tuple(lo)
        nx, ny, nz = hi[0] - lo[0] + 1, hi[1] - lo[1] + 1, hi[2] - lo[2] + 1
        host = mf.copy_to_host(mfi)
        current = host[:, :, :, 0]
        diff = current - phi0[lo_key]
        err_sq += float(np.sum(diff**2)) * dv
        ref_sq += float(np.sum(phi0[lo_key] ** 2)) * dv
    return math.sqrt(err_sq / ref_sq)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _write_plotfile(mf, geom, t, plot_count):
    """Write a plotfile and return the incremented counter."""
    pltdir = f"plt_vortex_{plot_count:04d}"
    if os.path.exists(pltdir):
        shutil.rmtree(pltdir)
    blockamr.write_single_level_plotfile(pltdir, mf, ["phi"], geom, t, 0)
    print(f"  Wrote {pltdir}  (t = {t:.4f})")
    return plot_count + 1


def run(
    n_cell=128,
    max_size=32,
    cfl=0.3,
    period=2.0,
    sigma=0.1,
    plotfile=True,
    write_interval=0.1,
    scheme="Upwind",
    memory="default",
):
    """Run the SingleVortex advection problem and return the relative L2 error."""

    # ---- domain setup ----
    div_scheme = DIV_SCHEMES[scheme]()
    ngrow = div_scheme.stencil_width

    box = blockamr.Box([0, 0, 0], [n_cell - 1, n_cell - 1, n_cell - 1])
    rb = blockamr.RealBox([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    geom = blockamr.Geometry(box, rb, 0, [1, 1, 1])

    ba = blockamr.BoxArray(box)
    ba.max_size(max_size)
    dm = blockamr.DistributionMapping(ba)

    mesh = Mesh(ba, dm, geom)
    phi = CellField(mesh, ncomp=1, ngrow=ngrow, name="phi", memory=memory)
    dx = geom.cell_size()

    # ---- initial condition ----
    init_gaussian(phi, geom, sigma=sigma)

    # save reference for error computation
    phi0 = {}
    for mfi in blockamr.MFIterator(phi.mf[0]):
        bx = mfi.valid_box()
        host = phi.mf[0].copy_to_host(mfi)
        phi0[tuple(bx.small_end())] = np.array(host[:, :, :, 0], copy=True)

    mass0 = compute_mass(phi.mf[0], dx)

    # write initial plotfile
    plot_count = 0
    if plotfile:
        plot_count = _write_plotfile(phi.mf[0], geom, 0.0, plot_count)

    # ---- build face fluxes ----
    def vel(x, y, z, t):
        return vortex_velocity(x, y, z, t, period=period)

    ff = build_face_fluxes(vel, box, dm, geom, ngrow=ngrow, t=0.0, max_size=max_size, memory=memory)

    # ---- time loop (OpenFOAM-style) ----
    u_max = 2.0  # max velocity magnitude
    dt = cfl * min(dx) / u_max
    t = 0.0
    nsteps = 0
    next_write = write_interval

    print(f"Grid: {n_cell}^3, dx = {dx[0]:.4e}, dt = {dt:.4e}")
    print(f"Scheme: {scheme}, Period T = {period}, CFL = {cfl}, write every {write_interval}s")
    print()

    flux_updater = FaceFluxUpdater(ff[0], vel, geom)

    while t < (period - 1e-12):
        if t + dt > period:
            dt = period - t
        flux_updater.update(t)
        expr = exp.ddt(phi) + Div(ff, phi, scheme=div_scheme)
        solve(expr, t=t, dt=dt)
        t += dt
        nsteps += 1

        # write plotfile at regular intervals
        if plotfile and t >= next_write - 1e-12:
            plot_count = _write_plotfile(phi.mf[0], geom, t, plot_count)
            next_write += write_interval

    # ---- diagnostics ----
    mass_final = compute_mass(phi.mf[0], dx)
    mass_err = abs(mass_final - mass0) / abs(mass0)
    l2_error = compute_l2_error(phi.mf[0], phi0, dx)

    print()
    print(f"Completed {nsteps} steps, final t = {t:.6f}")
    print(f"Mass conservation error: {mass_err:.2e}")
    print(f"Relative L2 error vs IC: {l2_error:.6f}")
    if plotfile:
        print(f"Wrote {plot_count} plotfiles")

    return l2_error


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SingleVortex advection example")
    parser.add_argument("--ncell", type=int, default=64, help="cells per dimension")
    parser.add_argument("--max-size", type=int, default=32, help="max grid size")
    parser.add_argument("--cfl", type=float, default=0.3, help="CFL number")
    parser.add_argument("--period", type=float, default=2.0, help="vortex period")
    parser.add_argument(
        "--write-interval", type=float, default=0.1, help="plotfile write interval in seconds"
    )
    parser.add_argument("--no-plot", action="store_true", help="skip plotfile output")
    parser.add_argument("--device", choices=["cpu", "gpu"], default=None, help="force cpu or gpu")
    parser.add_argument(
        "--backend",
        choices=["jax", "pallas", "triton"],
        default="jax",
        help="dispatch backend (default: jax)",
    )
    parser.add_argument(
        "--scheme",
        choices=list(DIV_SCHEMES),
        default="Upwind",
        help="divergence scheme (default: Upwind)",
    )
    args = parser.parse_args()

    with blockamr.runtime():
        blockamr.set_backend(args.backend)
        memory = "pinned" if args.device == "cpu" else "default"
        run(
            n_cell=args.ncell,
            max_size=args.max_size,
            cfl=args.cfl,
            period=args.period,
            plotfile=not args.no_plot,
            write_interval=args.write_interval,
            scheme=args.scheme,
            memory=memory,
        )
