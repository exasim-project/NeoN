# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""Lid-driven cavity at Re=100 — OpenFOAM-style DSL.

Solves the incompressible Navier-Stokes equations on a [0,1]^3 domain
(periodic in z) using explicit advection + diffusion with forward Euler
and a nodal pressure projection.

    ddt(U) + div(phi, U) - laplacian(nu, U) = 0
    laplacian(dt, p) = div(U*)
    U -= dt * grad(p)

Usage:
    python example/blockamr/lid_driven_cavity.py
    python example/blockamr/lid_driven_cavity.py --ncell 64 --re 1000
"""

import argparse
import os
import shutil

os.environ.setdefault("AMREX_THE_ARENA_INIT_SIZE", "0")

import jax.numpy as jnp
import numpy as np

import neon.blockamr as blockamr
from neon.blockamr.field import CellField, FaceField
from neon.blockamr.mesh import Mesh
from neon.blockamr.dsl import exp, solve, imp
from neon.blockamr.bc import VectorBC, fixedValue, noSlip
from neon.blockamr.fillpatch import FillPatchWithBC
from neon.blockamr.operators.interpolate import interpolate
from neon.blockamr.operators.correct import correct


def run(N_cells=64, Re=100, cfl=0.25, n_steps=5000, plot_interval=500, max_size=64, plotfile=True):
    nu = 1.0 / Re
    dt = cfl / N_cells

    # --- mesh ---
    box = blockamr.Box([0, 0, 0], [N_cells - 1, N_cells - 1, N_cells - 1])
    rb = blockamr.RealBox([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    geom = blockamr.Geometry(box, rb, 0, [0, 0, 1])  # periodic in z
    ba = blockamr.BoxArray(box)
    ba.max_size(max_size)
    dm = blockamr.DistributionMapping(ba)
    mesh = Mesh(ba, dm, geom)

    # --- fields (cf. createFields.H) ---
    U = CellField(mesh, ncomp=3, ngrow=1, name="U", fill_patch=FillPatchWithBC(
        VectorBC(
            xlo=noSlip(), xhi=noSlip(),
            ylo=noSlip(), yhi=fixedValue([1, 0, 0]),  # lid
        )
    ))
    p = CellField(mesh, ncomp=1, ngrow=0, name="p")  # pressure (nodal solve internal)
    phi = FaceField(mesh, ncomp=1, ngrow=1, name="phi")

    nu_func = lambda x, y, z, t: nu * jnp.ones_like(x)

    # --- scheme dictionaries (cf. fvSchemes / fvSolution) ---
    schemes_p = {"rtol": 1e-10, "atol": 1e-8, "max_iter": 200, "verbose": 0}

    print(f"Lid-driven cavity: N={N_cells}, Re={Re}, CFL={cfl}, dt={dt:.6f}, nu={nu:.6f}")
    print(f"  Viscous CFL: nu*dt/dx^2 = {nu * dt * N_cells**2:.4f}")

    # --- plotfile helper ---
    plot_count = 0

    def write_plotfile(step_num, time):
        nonlocal plot_count
        # Write U (velocity, 3 components)
        u_dir = f"plt_cavity_{plot_count:04d}"
        if os.path.exists(u_dir):
            shutil.rmtree(u_dir)
        blockamr.write_single_level_plotfile(
            u_dir, U.mf[0], ["Ux", "Uy", "Uz"], geom, time, step_num)
        print(f"  Step {step_num:6d}, t = {time:.4f}  → wrote {u_dir}")
        plot_count += 1

    # --- time loop (cf. icoFoam) ---
    t = 0.0
    for step in range(1, n_steps + 1):

        # Face flux from cell velocity
        print(f"Step {step:6d}, t = {t:.4f}")
        interpolate(U, phi)

        # Momentum predictor (explicit):
        #   ddt(U) + div(phi, U) - laplacian(nu, U) = 0
        solve(exp.ddt(U) + exp.div(phi, U) - exp.laplacian(nu_func, U), t, dt)

        # Fill BCs on U* before pressure solve
        U.fill_patch(0, t)

        # Pressure correction (implicit nodal Poisson):
        #   laplacian(dt, p) = div(U*)
        solve(imp.laplacian(dt, p) == exp.div(U), schemes=schemes_p)

        # Velocity correction: U -= dt * grad(p)
        correct(U, -dt * exp.grad(p))

        t += dt

        if plotfile and step % plot_interval == 0:
            write_plotfile(step, t)
            

    # --- extract centreline ---
    dx = geom.cell_size()
    ix, iz = N_cells // 2, N_cells // 2
    U_arrs = U.mf[0].arrays()[0]
    ng = U.mf[0].n_grow()
    u_profile = np.array(U_arrs[ng + ix, ng:ng + N_cells, ng + iz, 0])
    y_coords = np.array([(j + 0.5) * dx[1] for j in range(N_cells)])

    print("\n  y-coordinate    u-velocity")
    for y, u in zip(y_coords, u_profile):
        print(f"  {y:12.6f}  {u:12.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lid-driven cavity")
    parser.add_argument("--ncell", type=int, default=32)
    parser.add_argument("--re", type=int, default=100)
    parser.add_argument("--cfl", type=float, default=0.25, help="CFL number (default: 0.25)")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--plot-interval", type=int, default=100)
    parser.add_argument("--max-size", type=int, default=64, help="Max block size for AMR (default: 64)")
    parser.add_argument("--no-plot", type=bool, default=False, help="Skip plotfile output")
    parser.add_argument("--backend", choices=["jax", "pallas", "triton"],
                        default="jax", help="dispatch backend")
    args = parser.parse_args()

    with blockamr.runtime():
        blockamr.set_backend(args.backend)
        run(N_cells=args.ncell, Re=args.re, cfl=args.cfl, n_steps=args.steps,
            plot_interval=args.plot_interval, max_size=args.max_size,
            plotfile=not args.no_plot)
