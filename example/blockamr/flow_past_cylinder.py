# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""Flow past a cylinder — incompressible Navier-Stokes with embedded boundary.

Validates the EB infrastructure end-to-end:

- ``EB2_CylinderIF`` builds an implicit-function geometry.
- ``make_eb_factory`` creates an ``EBFArrayBoxFactory`` honouring the
  PaddedArena single-chunk allocation, so JAX/Pallas kernels still read
  field data zero-copy.
- ``Mesh(..., eb_factory=ebf)`` flips ``mesh.has_eb=True``; this is the
  *only* difference from ``double_shear_layer.py`` at the user level.
- The same ``DSLIncompressibleSolver`` runs without an alternate code path:
  ``CellField.fill_patch`` zeros covered cells, the MAC projection uses
  ``MLEBABecLaplacian`` with EB-Dirichlet wall, the pressure correction
  is a cell-centred Chorin-style projection, and ``parallel_for``
  multiplies stencil output by ``mesh.vol_frac(lev)``.

Geometry: a cylinder of radius ``D/2`` aligned with the z-axis, centred at
(cx, 0.5, 0.5) inside a periodic [0, Lx] × [0, 1] × [0, Lz] box. The flow
is initialised to U = (1, 0, 0); the no-slip wall on the cylinder forces a
boundary layer and (eventually) wake / vortex shedding.

Quantitative drag verification (Cd at Re=20 / Re=100) requires inflow /
outflow boundary conditions which are not yet plumbed for cell-centred
EB; this example is a runtime / smoke / regression test for the EB
infrastructure rather than a published-coefficient validation.

Usage:
    python example/blockamr/flow_past_cylinder.py
    python example/blockamr/flow_past_cylinder.py --re 100 --steps 200
    python example/blockamr/flow_past_cylinder.py --ncell 64 --plot-interval 20
"""

import argparse
import os

os.environ.setdefault("AMREX_THE_ARENA_INIT_SIZE", "0")
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(0.25)

import jax.numpy as jnp

import neon.blockamr as blockamr
from neon.blockamr.bc import VectorBC, fixedValue, NeumannBC, slipWall
from neon.blockamr.mesh import Mesh
from neon.blockamr.field import CellField
from neon.blockamr.dsl_solver import DSLIncompressibleSolver
from neon.blockamr.fillpatch import FillPatchWithBC
from neon.blockamr.schemes.div_schemes import VanLeer


def init_uniform_flow(mf, geom, U_inf=1.0):
    """U = (U_inf, 0, 0) everywhere. eb_set_covered will zero solid cells."""
    for mfi in blockamr.MFIterator(mf):
        bx = mfi.valid_box()
        lo = bx.small_end()
        hi = bx.big_end()
        nx = hi[0] - lo[0] + 1
        ny = hi[1] - lo[1] + 1
        nz = hi[2] - lo[2] + 1
        u = jnp.full((nx, ny, nz), U_inf)
        v = jnp.zeros((nx, ny, nz))
        w = jnp.zeros((nx, ny, nz))
        vals = jnp.stack([u, v, w], axis=-1)
        mf.copy_from(mfi, vals)


def wake_drag_coefficient(solver, x_outflow, U_inf=1.0, D=1.0, rho=1.0):
    """Canonical wake-deficit drag estimator (no pressure required).

    For incompressible flow with uniform inflow U_inf, the streamwise
    force on the cylinder equals the wake momentum deficit measured
    far downstream:

        F_x = ρ U_inf · ∫_y (U_inf − U_x(x_outflow, y)) dy dz

    This is the textbook "wake survey" formula. It assumes (i) the
    measurement station is far enough downstream that p → p_inf and
    U_y is small, (ii) the cross-section is wide enough to capture
    the entire deficit. The result is independent of the second-order
    term ∫(U_inf − U_x)² which is small for thin wakes.

    A previous version of this function used
    F_x = ρ ∫(U_inf² − U_x²) dy = ρ ∫(U_inf − U_x)(U_inf + U_x) dy
    which overestimates the canonical formula by ~(U_inf + U_x_avg) /
    (2 U_inf) ≈ 1 + (1 − Δ)/2 ≈ 2× for typical small deficits.

    Lift is computed from the y-momentum flux through the same
    outflow line: F_y = ρ ∫_y U_x · U_y dy dz (zero for symmetric
    flow, finite for an asymmetric wake).

    Cd = F_x / (0.5 ρ U_inf² D)
    Cl = F_y / (0.5 ρ U_inf² D)
    """
    import numpy as np
    mesh = solver.mesh
    geom = mesh.geom(0)
    dx = np.array(geom.cell_size())
    plo = np.array(geom.prob_lo())
    phi_arr = solver.U.mf[0]
    ng = phi_arr.n_grow()

    Lz = float(geom.prob_hi()[2] - geom.prob_lo()[2])

    deficit = 0.0
    fy_flux = 0.0
    for mfi in blockamr.MFIterator(phi_arr):
        bx = mfi.valid_box()
        lo = bx.small_end()
        hi = bx.big_end()
        u4 = phi_arr.array(mfi)
        ux = np.asarray(u4[:, :, :, 0])
        uy = np.asarray(u4[:, :, :, 1])
        nx = hi[0] - lo[0] + 1
        ny = hi[1] - lo[1] + 1
        nz = hi[2] - lo[2] + 1

        xc = plo[0] + (np.arange(nx) + lo[0] + 0.5) * dx[0]
        ix = int(np.argmin(np.abs(xc - x_outflow)))
        if abs(xc[ix] - x_outflow) <= 0.5 * dx[0]:
            ux_v = ux[ng + ix, ng:ng + ny, ng:ng + nz]
            uy_v = uy[ng + ix, ng:ng + ny, ng:ng + nz]
            # Canonical wake-deficit: F_x = ρ U_inf · ∫(U_inf - U_x) dy dz
            deficit += float(rho * U_inf * np.sum(U_inf - ux_v) * dx[1] * dx[2])
            fy_flux += float(rho * np.sum(ux_v * uy_v) * dx[1] * dx[2])

    Fx = deficit
    Fy = -fy_flux  # force on cylinder is opposite of CV efflux
    q = 0.5 * rho * U_inf * U_inf * D * Lz
    return Fx / q, Fy / q


# Backwards-compat alias
def cv_drag_lift(solver, x_lo, x_hi, y_lo, y_hi, U_inf=1.0, D=1.0, rho=1.0):
    return wake_drag_coefficient(solver, x_hi, U_inf=U_inf, D=D, rho=rho)


def compute_vorticity(U, omega):
    """z-vorticity dv/dx - du/dy on every level (per-fab JAX)."""
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
    N_cells=64,
    Nz=4,
    Re=100,
    cfl=0.2,
    n_steps=100,
    plot_interval=20,
    # KNOWN LIMITATION: AMReX MLNodeLaplacian + EB + multi-box decomposition
    # produces a structural residual plateau (~1.5e-4 relative) on the second
    # solve and beyond when one fab carries cut cells and another doesn't.
    # The plateau is robust against bottom-solver choice, multigrid coarsening
    # depth, warm-start state, and explicit solvability projection. Default
    # max_size is therefore set large enough to keep the layout single-box at
    # the example's default resolution. Override --max-size only if you have
    # tested multi-box with your specific cylinder placement.
    max_size=256,
    Lx=2.0,
    cyl_diameter=0.1,
    cyl_x=0.4,
    plotfile=True,
):
    U_inf = 1.0
    D = cyl_diameter
    nu = U_inf * D / Re

    Nx = int(N_cells * Lx)
    Ny = N_cells
    dx = 1.0 / N_cells
    dt = cfl * dx / U_inf

    # --- geometry: inflow/outflow in x, slip walls in y, periodic in z ---
    box = blockamr.Box([0, 0, 0], [Nx - 1, Ny - 1, Nz - 1])
    rb = blockamr.RealBox([0.0, 0.0, 0.0], [Lx, 1.0, Nz / N_cells])
    geom = blockamr.Geometry(box, rb, 0, [0, 0, 1])

    # --- EB: cylinder oriented along z, centred at (cyl_x, 0.5, 0) ---
    cyl = blockamr.EB2_CylinderIF(
        D / 2, 2, [cyl_x, 0.5, 0.0], False)
    blockamr.eb2_build_cylinder(cyl, geom, 0, 100)

    ba = blockamr.BoxArray(box)
    ba.max_size(max_size)
    dm = blockamr.DistributionMapping(ba)
    ebf = blockamr.make_eb_factory(geom, ba, dm)

    # --- Mesh: same class as non-EB; eb_factory flips has_eb=True ---
    mesh = Mesh(ba, dm, geom, eb_factory=ebf)
    assert mesh.has_eb

    # --- BCs: uniform inflow at xlo, zero-grad outflow at xhi,
    #     free-slip walls in y, periodic in z. The y walls must use a
    #     proper slip BC (U_y = 0, ∂U_x/∂y = 0); a pure NeumannBC()
    #     on a vector field allows U_y to leak through the walls,
    #     breaking mass conservation and pulling outflow velocity
    #     toward zero. See bc.SlipWallBC.
    U_bc = VectorBC(
        xlo=fixedValue([U_inf, 0.0, 0.0]),
        xhi=NeumannBC(),
        ylo=slipWall(),
        yhi=slipWall(),
    )

    # --- solver: same class as non-EB ---
    schemes_p = {"rtol": 1e-10, "atol": 1e-10, "max_iter": 200, "verbose": 0}
    solver = DSLIncompressibleSolver(
        mesh, nu, dt,
        fill_patch=FillPatchWithBC(U_bc),
        schemes_p=schemes_p,
        div_scheme=VanLeer(),
        cfl=cfl,
    )

    omega = CellField(mesh, ncomp=1, ngrow=0, name="vorticity",
                      fill_patch=FillPatchWithBC(U_bc))

    # --- IC: uniform flow; CellField.fill_patch zeros covered cells ---
    init_uniform_flow(solver.U.mf[0], geom, U_inf=U_inf)
    solver.U.fill_patch(0, 0.0)

    # --- volfrac diagnostics ---
    vf_per_box = mesh.vol_frac(0)
    fluid_cells = sum(int((vf > 0).sum()) for vf in vf_per_box)
    cut_cells = sum(int(((vf > 0) & (vf < 1)).sum()) for vf in vf_per_box)
    total = sum(int(vf.size) for vf in vf_per_box)
    print(f"Flow past cylinder: Nx×Ny×Nz = {Nx}×{Ny}×{Nz}, "
          f"Re={Re}, D={D}, U_inf={U_inf}, dt={dt:.5f}")
    print(f"  cylinder centre = ({cyl_x}, 0.5, 0.0); radius = {D/2}")
    print(f"  has_eb={mesh.has_eb}, fluid_cells={fluid_cells:,} "
          f"({100*fluid_cells/total:.1f}%), cut_cells={cut_cells:,}")
    print(f"  nu = U_inf*D/Re = {nu:.6f}")

    plot_count = 0

    def write_plot():
        nonlocal plot_count
        pdir = f"plt_cyl_{plot_count:04d}"
        for lev in range(mesh.n_levels()):
            solver.U.fill_patch(lev, solver.time)
        compute_vorticity(solver.U, omega)
        solver.write_plotfile(pdir, fields=[solver.U, omega])
        print(f"  -> wrote {pdir}")
        plot_count += 1

    if plotfile:
        write_plot()

    # Wake-deficit measurement station: 10 D downstream of cylinder
    x_outflow_meas = cyl_x + 10.0 * D

    cd_history = []
    cl_history = []
    t_history = []

    log_every = max(1, n_steps // 50)

    for step in range(1, n_steps + 1):
        cd, cl = wake_drag_coefficient(solver, x_outflow_meas,
                                        U_inf=U_inf, D=D)
        cd_history.append(cd)
        cl_history.append(cl)
        t_history.append(solver.time)
        if step % log_every == 0 or step <= 5:
            max_vel = solver._max_velocity()
            print(f"Step {step:6d}, t = {solver.time:.4f}, "
                  f"dt = {solver.dt:.5f}, max|U| = {max_vel:.4f}, "
                  f"Cd = {cd:+.4f}, Cl = {cl:+.4f}")
        solver.step()
        if plotfile and step % plot_interval == 0:
            write_plot()

    # Final running mean over the last 20% of steps (steady-state estimate)
    n_avg = max(1, len(cd_history) // 5)
    cd_mean = sum(cd_history[-n_avg:]) / n_avg
    cl_mean = sum(cl_history[-n_avg:]) / n_avg
    print(f"\nFinal Cd (mean over last {n_avg} samples): {cd_mean:+.4f}")
    print(f"Final Cl (mean over last {n_avg} samples): {cl_mean:+.4f}")

    # Per-x mass-flux profile diagnostic. For incompressibility with
    # a uniform inflow and slip walls in y, the streamwise integral
    # ∫U_x dy must be constant in x and equal U_inf · Ly. If it's
    # not, mass is leaking somewhere (or the projection is wrong).
    import numpy as np
    mf = solver.U.mf[0]
    ng = mf.n_grow()
    arr = np.asarray(mf.arrays()[0])
    nx_v = arr.shape[0] - 2 * ng
    ny_v = arr.shape[1] - 2 * ng
    nz_v = arr.shape[2] - 2 * ng
    Ly = float(mesh.geom(0).prob_hi()[1] - mesh.geom(0).prob_lo()[1])
    dy = Ly / ny_v
    u_valid = arr[ng:ng + nx_v, ng:ng + ny_v, ng:ng + nz_v, 0]
    flux_per_col = u_valid.sum(axis=(1, 2)) * dy / nz_v   # ∫U_x dy at each x
    expected = U_inf * Ly
    rel_err = np.abs(flux_per_col - expected) / expected
    print(f"\nPer-x mass-flux ∫U_x dy (expected {expected:.4f}):")
    print(f"  inflow column          = {float(flux_per_col[0]):.4f}")
    print(f"  cylinder vicinity (~mid)= {float(flux_per_col[nx_v // 4]):.4f}")
    print(f"  mid-domain             = {float(flux_per_col[nx_v // 2]):.4f}")
    print(f"  3/4 downstream         = {float(flux_per_col[3 * nx_v // 4]):.4f}")
    print(f"  outflow column         = {float(flux_per_col[-1]):.4f}")
    print(f"  max rel err over all x = {float(rel_err.max()):.4e}  "
          f"(at column {int(rel_err.argmax())} of {nx_v})")

    return {
        "t": t_history,
        "cd": cd_history,
        "cl": cl_history,
        "ncell": N_cells,
        "Re": Re,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flow past cylinder (EB)")
    parser.add_argument("--ncell", type=int, default=64,
                        help="Cells per unit length in y (default: 64)")
    parser.add_argument("--nz", type=int, default=4)
    parser.add_argument("--re", type=int, default=100)
    parser.add_argument("--cfl", type=float, default=0.2)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--plot-interval", type=int, default=20)
    parser.add_argument("--max-size", type=int, default=256,
                        help="Max box size (default 256, large enough to keep "
                             "single-box layout — see KNOWN LIMITATION in run())")
    parser.add_argument("--lx", type=float, default=2.0,
                        help="Domain length in x (default: 2.0)")
    parser.add_argument("--diameter", type=float, default=0.1,
                        help="Cylinder diameter (default: 0.1, blockage 10%%)")
    parser.add_argument("--cyl-x", type=float, default=0.4,
                        help="Cylinder x position (default: 0.6)")
    parser.add_argument("--no-plot", action="store_true", default=False)
    args = parser.parse_args()

    with blockamr.runtime():
        run(
            N_cells=args.ncell,
            Nz=args.nz,
            Re=args.re,
            cfl=args.cfl,
            n_steps=args.steps,
            plot_interval=args.plot_interval,
            max_size=args.max_size,
            Lx=args.lx,
            cyl_diameter=args.diameter,
            cyl_x=args.cyl_x,
            plotfile=not args.no_plot,
        )
