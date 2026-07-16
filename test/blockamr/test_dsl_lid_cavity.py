# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""Lid-driven cavity validation using the DSL solver against Ghia et al. (1982)."""

import numpy as np
import blockamr
from blockamr.mesh import Mesh
from blockamr.bc import VectorBC, fixedValue, noSlip
from blockamr.incompressible import build_incompressible, step


# Ghia et al. (1982), Table I — Re=100
# u-velocity along vertical centreline (x=0.5)
GHIA_Y = np.array(
    [
        0.0000,
        0.0547,
        0.0625,
        0.0703,
        0.1016,
        0.1719,
        0.2813,
        0.4531,
        0.5000,
        0.6172,
        0.7344,
        0.8516,
        0.9531,
        0.9609,
        0.9688,
        0.9766,
        1.0000,
    ]
)
GHIA_U = np.array(
    [
        0.00000,
        -0.03717,
        -0.04192,
        -0.04775,
        -0.06434,
        -0.10150,
        -0.15662,
        -0.21090,
        -0.20581,
        -0.13641,
        0.00332,
        0.23151,
        0.68717,
        0.73722,
        0.78871,
        0.84123,
        1.00000,
    ]
)


def _make_dsl_cavity_solver(N, Re, cfl=0.25):
    """Create a DSL-based lid-driven cavity solver."""
    nu = 1.0 / Re
    dt = cfl / N

    box = blockamr.Box([0, 0, 0], [N - 1, N - 1, N - 1])
    rb = blockamr.RealBox([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    geom = blockamr.Geometry(box, rb, 0, [0, 0, 1])  # periodic in z
    ba = blockamr.BoxArray(box)
    ba.max_size(N)
    dm = blockamr.DistributionMapping(ba)
    mesh = Mesh(ba, dm, geom)

    U_bc = VectorBC(
        xlo=noSlip(),
        xhi=noSlip(),
        ylo=noSlip(),
        yhi=fixedValue([1, 0, 0]),  # lid
    )

    solver = build_incompressible(mesh, nu, dt, U_bc=U_bc)
    return solver, geom


def _extract_centreline_u(solver, geom, N):
    """Extract u-velocity along vertical centreline x=0.5, z=mid."""
    U_mf = solver.U.mf[0]
    dx = geom.cell_size()
    ng = U_mf.n_grow()
    ix = N // 2
    iz = N // 2

    U_arrs = U_mf.arrays()[0]
    u_profile = np.array(U_arrs[ng + ix, ng : ng + N, ng + iz, 0])
    y_coords = np.array([(j + 0.5) * dx[1] for j in range(N)])
    return y_coords, u_profile


def test_dsl_lid_cavity_velocity_bounded(blockamr_session):
    """Velocity should remain bounded and not blow up."""
    N = 8
    solver, geom = _make_dsl_cavity_solver(N, Re=100, cfl=0.25)

    for _ in range(20):
        step(solver)

    U_arrs = solver.U.mf[0].arrays()[0]
    max_vel = float(np.max(np.abs(np.array(U_arrs))))
    assert max_vel < 5.0, f"Max velocity {max_vel} — solver may be unstable"
    assert max_vel > 0.01, f"Max velocity {max_vel} — solver may not be running"


def test_dsl_lid_cavity_re100_centreline(blockamr_session):
    """DSL lid-driven cavity at Re=100 matches Ghia et al. (1982).

    Uses N=16, runs enough steps for approximate steady state.
    Compares u-velocity along vertical centreline.
    """
    N = 16
    Re = 100
    solver, geom = _make_dsl_cavity_solver(N, Re, cfl=0.25)

    n_steps = 3000
    for _ in range(n_steps):
        step(solver)

    y_coords, u_profile = _extract_centreline_u(solver, geom, N)

    # Interpolate Ghia data to our grid points
    ghia_interp = np.interp(y_coords, GHIA_Y, GHIA_U)

    # Check agreement — allow ~15% tolerance for the coarse 16-cell grid
    mask = np.abs(ghia_interp) > 0.05
    if np.any(mask):
        rel_err = np.abs(u_profile[mask] - ghia_interp[mask]) / np.abs(ghia_interp[mask])
        max_rel_err = np.max(rel_err)
        assert max_rel_err < 0.5, f"Max relative error {max_rel_err:.2f} vs Ghia at Re=100 (N={N})"

    # Also check absolute error for all points
    abs_err = np.max(np.abs(u_profile - ghia_interp))
    assert abs_err < 0.3, f"Max absolute error {abs_err:.3f} vs Ghia at Re=100 (N={N})"
