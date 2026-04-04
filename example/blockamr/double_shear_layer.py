# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""Double shear layer — incompressible Navier-Stokes benchmark.

Solves the incompressible Navier-Stokes equations on a fully periodic
[0,1]^3 domain (quasi-2D, thin in z) using the DSL solver:

    ddt(U) + div(phi, U) - laplacian(nu, U) = 0
    laplacian(dt, p) = div(U*)
    U -= dt * grad(p)

The initial condition consists of two horizontal shear layers with a
small vertical perturbation that triggers Kelvin-Helmholtz roll-up:

    u = tanh(rho * (y - 0.25))     for y <= 0.5
        tanh(rho * (0.75 - y))     for y >  0.5
    v = delta * sin(2 * pi * x)
    w = 0

Supports both single-level and multi-level AMR (with --max-level).

Usage:
    python example/blockamr/double_shear_layer.py
    python example/blockamr/double_shear_layer.py --ncell 64 --max-level 1
    python example/blockamr/double_shear_layer.py --ncell 256 --re 100000
"""

import argparse
import os
import shutil

os.environ.setdefault("AMREX_THE_ARENA_INIT_SIZE", "0")
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(0.25)

import jax.numpy as jnp

import neon.blockamr as blockamr
from neon.blockamr.mesh import Mesh, AmrMesh
from neon.blockamr.field import CellField
from neon.blockamr.dsl_solver import DSLIncompressibleSolver
from neon.blockamr.fillpatch import FillPatchCellConservative
from neon.blockamr.schemes.div_schemes import VanLeer, QUICK, Linear, Upwind


def init_double_shear_layer(mf, geom, rho=80.0, delta=0.05):
    """Set the double shear layer initial condition on a MultiFab."""
    dx = geom.cell_size()
    for mfi in blockamr.MFIterator(mf):
        bx = mfi.valid_box()
        lo = bx.small_end()
        hi = bx.big_end()
        nx = hi[0] - lo[0] + 1
        ny = hi[1] - lo[1] + 1
        nz = hi[2] - lo[2] + 1

        xs = jnp.array([(lo[0] + i + 0.5) * dx[0] for i in range(nx)])
        ys = jnp.array([(lo[1] + j + 0.5) * dx[1] for j in range(ny)])

        x2d = xs[:, None] * jnp.ones((1, ny))
        y2d = jnp.ones((nx, 1)) * ys[None, :]

        u = jnp.where(
            y2d <= 0.5,
            jnp.tanh(rho * (y2d - 0.25)),
            jnp.tanh(rho * (0.75 - y2d)),
        )
        v = delta * jnp.sin(2.0 * jnp.pi * x2d)
        w = jnp.zeros_like(u)

        vals = jnp.stack([u, v, w], axis=-1)  # (nx, ny, 3)
        vals = vals[:, :, None, :] * jnp.ones((1, 1, nz, 1))  # (nx, ny, nz, 3)
        mf.copy_from(mfi, vals)


def tag_vorticity(U, threshold=50.0, tag_field=None):
    """Tagging function based on vorticity magnitude (dv/dx - du/dy).

    If *tag_field* is a CellField, the 0/1 mask is also written to it
    for diagnostic output.
    """

    def _tag(lev, tags, time, ngrow):
        if U.mf[lev] is None:
            return
        dx = U.mesh.geom(lev).cell_size()
        ng = U.mf[lev].n_grow()

        # Zero the tag field before writing
        if tag_field is not None and tag_field.mf[lev] is not None:
            tag_field.mf[lev].set_val(0.0)

        tag_results = []
        for mfi in blockamr.MFIterator(U.mf[lev]):
            u4d = U.mf[lev].array(mfi)
            ux = u4d[:, :, :, 0]
            uy = u4d[:, :, :, 1]
            bx = mfi.valid_box()
            lo = bx.small_end()
            hi = bx.big_end()
            vn = [hi[d] - lo[d] + 1 for d in range(3)]

            dvdx = (uy[ng+1:ng+1+vn[0], ng:ng+vn[1], ng:ng+vn[2]]
                    - uy[ng-1:ng-1+vn[0], ng:ng+vn[1], ng:ng+vn[2]]) / (2 * dx[0])
            dudy = (ux[ng:ng+vn[0], ng+1:ng+1+vn[1], ng:ng+vn[2]]
                    - ux[ng:ng+vn[0], ng-1:ng-1+vn[1], ng:ng+vn[2]]) / (2 * dx[1])
            vort = jnp.abs(dvdx - dudy)
            mask = (vort > threshold).astype(jnp.int32)
            tags.set_tags(mfi, mask)
            tag_results.append(mask.astype(jnp.float64))

        if tag_field is not None and tag_field.mf[lev] is not None:
            tag_field.mf[lev].copy_arrays(tag_results)

    return _tag


def compute_vorticity(U, omega):
    """Compute z-vorticity (dv/dx - du/dy) into CellField omega at all levels."""
    mesh = U.mesh
    for lev in range(mesh.n_levels()):
        if U.mf[lev] is None or omega.mf[lev] is None:
            continue
        dx = mesh.geom(lev).cell_size()
        ng = U.mf[lev].n_grow()
        results = []
        for arr in U.mf[lev].arrays():
            vn = [int(arr.shape[ax]) - 2 * ng for ax in range(3)]
            ux = arr[:, :, :, 0]
            uy = arr[:, :, :, 1]
            dvdx = (uy[ng + 1:ng + 1 + vn[0], ng:ng + vn[1], ng:ng + vn[2]]
                    - uy[ng - 1:ng - 1 + vn[0], ng:ng + vn[1], ng:ng + vn[2]]) / (2 * dx[0])
            dudy = (ux[ng:ng + vn[0], ng + 1:ng + 1 + vn[1], ng:ng + vn[2]]
                    - ux[ng:ng + vn[0], ng - 1:ng - 1 + vn[1], ng:ng + vn[2]]) / (2 * dx[1])
            results.append(dvdx - dudy)
        omega.mf[lev].copy_arrays(results)


def run(
    N_cells=128,
    Nz=4,
    Re=10000,
    cfl=0.25,
    n_steps=2000,
    plot_interval=200,
    max_size=64,
    blocking_factor=4,
    rho=80.0,
    delta=0.05,
    plotfile=True,
    max_level=0,
    tag_threshold=50.0,
):
    nu = 1.0 / Re
    ref_ratio_product = 2 ** max_level
    dt = cfl / (N_cells * ref_ratio_product)

    # --- mesh ---
    box = blockamr.Box([0, 0, 0], [N_cells - 1, N_cells - 1, Nz - 1])
    # box = blockamr.Box([0, 0, 0], [N_cells - 1, N_cells - 1, N_cells - 1])
    rb = blockamr.RealBox([0.0, 0.0, 0.0], [1.0, 1.0, Nz / N_cells])
    geom = blockamr.Geometry(box, rb, 0, [1, 1, 1])  # periodic in all dirs

    if max_level == 0:
        ba = blockamr.BoxArray(box)
        ba.max_size(max_size)
        dm = blockamr.DistributionMapping(ba)
        mesh = Mesh(ba, dm, geom)
    else:
        info = blockamr.AmrInfo()
        info.max_level = max_level
        for lev in range(max_level):
            info.set_ref_ratio(lev, 2)  # MLNodeLaplacian requires isotropic ratio (2 or 4)
        info.set_max_grid_size(0, max_size)
        info.set_blocking_factor(0, blocking_factor)
        mesh = AmrMesh(geom, info)

    # --- solver ---
    schemes_p = {"rtol": 0, "atol": 1e-8, "max_iter": 200, "verbose": 0}
    div_scheme = VanLeer() # VanLeer()
    solver = DSLIncompressibleSolver(
        mesh, nu, dt, fill_patch=FillPatchCellConservative(), schemes_p=schemes_p,
        div_scheme=div_scheme, cfl=cfl,
    )

    # --- diagnostic output fields ---
    omega = CellField(mesh, ncomp=1, ngrow=0, name="vorticity",
                      fill_patch=FillPatchCellConservative())
    tag_field = CellField(mesh, ncomp=1, ngrow=0, name="tagged",
                          fill_patch=FillPatchCellConservative())

    # --- initial condition ---
    tag_func = tag_vorticity(solver.U, threshold=tag_threshold, tag_field=tag_field)
    if max_level == 0:
        init_double_shear_layer(solver.U.mf[0], geom, rho=rho, delta=delta)
    else:
        mesh.init_from_scratch(0.0)
        init_double_shear_layer(solver.U.mf[0], mesh.geom(0), rho=rho, delta=delta)
        for _ in range(max_level + 1):
            mesh.regrid(0.0, tag=tag_func)
            for lev in range(mesh.n_levels()):
                init_double_shear_layer(solver.U.mf[lev], mesh.geom(lev), rho=rho, delta=delta)

    total_cells = 0
    for lev in range(mesh.n_levels()):
        mf = solver.U.mf[lev]
        if mf is not None:
            lev_cells = sum(
                (m[1] - 2 * mf.n_grow()) * (m[2] - 2 * mf.n_grow()) * (m[3] - 2 * mf.n_grow())
                for m in mf.fab_metadata()
            )
            total_cells += lev_cells

    from neon.blockamr.dsl.solve import BF

    print(f"Double shear layer: N={N_cells}, Re={Re}, CFL={cfl}, dt={dt:.6f}, nu={nu:.8f}")
    print(f"  rho={rho}, delta={delta}, max_level={max_level}, max_size={max_size}")
    print(f"  Levels: {mesh.n_levels()}, total cells: {total_cells:,}")
    for lev in range(mesh.n_levels()):
        mf = solver.U.mf[lev]
        if mf is not None:
            n_boxes = len(mf.fab_metadata())
            ng = mf.n_grow()
            lev_cells = sum(
                (m[1]-2*ng)*(m[2]-2*ng)*(m[3]-2*ng) for m in mf.fab_metadata())
            layout = blockamr.build_tile_layout(mf, BF)
            print(f"    lev {lev}: {lev_cells:,} cells, {n_boxes} boxes, "
                  f"tiles={layout.n_tiles} (padded={layout.n_tiles_padded}), bf={BF}")
    print(f"  Viscous CFL: nu*dt/dx^2 = {nu * dt * N_cells**2:.4f}")

    # --- plotfile helper ---
    plot_count = 0

    def write_plot(step_num):
        nonlocal plot_count
        pdir = f"plt_shear_{plot_count:04d}"
        for lev in range(mesh.n_levels()):
            solver.U.fill_patch(lev, solver.time)
        compute_vorticity(solver.U, omega)
        solver.write_plotfile(pdir, fields=[solver.U, omega, tag_field])
        print(f"  -> wrote {pdir}")
        plot_count += 1

    # --- time loop ---
    for step in range(1, n_steps + 1):
        print(f"Step {step:6d}, t = {solver.time:.4f}, dt = {solver.dt:.6f}, "
              f"max|U| = {solver._max_velocity():.6f}")

        if step % 10 == 0:
            solver.regrid(tag=tag_func)

        solver.step()

        if plotfile and step % plot_interval == 0:
            write_plot(step)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Double shear layer")
    parser.add_argument("--ncell", type=int, default=128)
    parser.add_argument("--nz", type=int, default=4, help="Number of cells in z-direction (default: 4)")
    parser.add_argument("--re", type=int, default=10000)
    parser.add_argument("--cfl", type=float, default=0.25, help="CFL number (default: 0.25)")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--plot-interval", type=int, default=50)
    parser.add_argument("--rho", type=float, default=80.0, help="Shear layer thickness (default: 80)")
    parser.add_argument("--delta", type=float, default=0.05, help="Perturbation amplitude (default: 0.05)")
    parser.add_argument("--max-size", type=int, default=64, help="Max block size (default: 64)")
    parser.add_argument("--blocking-factor", type=int, default=4, help="AMR blocking factor (default: 4)")
    parser.add_argument("--tile-size", type=int, default=8, help="Pallas tile size (default: 8)")
    parser.add_argument("--max-level", type=int, default=0, help="AMR max refinement level (default: 0)")
    parser.add_argument("--tag-threshold", type=float, default=1.0, help="Vorticity tagging threshold")
    parser.add_argument("--no-plot", action="store_true", default=False, help="Skip plotfile output")
    parser.add_argument("--backend", choices=["jax", "pallas", "triton"],
                        default="jax", help="dispatch backend")
    args = parser.parse_args()

    with blockamr.runtime():
        blockamr.set_backend(args.backend)
        blockamr.set_tile_size(args.tile_size)
        run(
            N_cells=args.ncell,
            Nz=args.nz,
            Re=args.re,
            cfl=args.cfl,
            n_steps=args.steps,
            plot_interval=args.plot_interval,
            max_size=args.max_size,
            blocking_factor=args.blocking_factor,
            rho=args.rho,
            delta=args.delta,
            plotfile=not args.no_plot,
            max_level=args.max_level,
            tag_threshold=args.tag_threshold,
        )
