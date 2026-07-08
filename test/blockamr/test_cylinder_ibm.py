# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""Flow around a cylinder via direct-forcing immersed boundary (IBM).

The cylinder is not cut into the grid; instead the engine pins the velocity to
zero in the solid cells each step (mask from center/radius/axis). The projection
then deflects the free stream around the zero-velocity zone. Checks the mask, the
no-slip forcing, and a physically sane wake (upstream stagnation, accelerated
flanks, low-velocity wake) without asserting literature drag (that is Spec 03).
"""

import numpy as np
import pytest

import neon.blockamr as blockamr
from neon.blockamr.bc import VectorBC, fixedValue, NeumannBC, slip
from neon.blockamr.dsl_solver import DSLIncompressibleSolver
from neon.blockamr.mesh import Mesh

U0 = 1.0
D = 0.2
RADIUS = D / 2.0
NU = U0 * D / 20.0  # Re = 20
NX, NY, NZ = 48, 24, 8
LX, LY, LZ = 2.0, 1.0, 0.25
CENTER = [0.5, 0.5, 0.125]


def _make_cylinder_solver():
    box = blockamr.Box([0, 0, 0], [NX - 1, NY - 1, NZ - 1])
    rb = blockamr.RealBox([0.0, 0.0, 0.0], [LX, LY, LZ])
    geom = blockamr.Geometry(box, rb, 0, [0, 0, 1])
    ba = blockamr.BoxArray(box)
    ba.max_size(max(NX, NY, NZ))
    dm = blockamr.DistributionMapping(ba)
    mesh = Mesh(ba, dm, geom)

    u_bc = VectorBC(
        xlo=fixedValue([U0, 0.0, 0.0]), xhi=NeumannBC(),
        ylo=slip(), yhi=slip(),
    )
    dt = 0.2 * float(geom.cell_size()[0]) / U0
    eb = {"center": CENTER, "radius": RADIUS, "axis": 2}
    solver = DSLIncompressibleSolver(mesh, NU, dt, u_bc, eb=eb)

    ng = solver.U.mf[0].n_grow()
    g = solver.U.mf[0].grown_arrays()[0]
    g = g.at[:, :, :, 0].set(U0).at[:, :, :, 1:].set(0.0)
    solver.U.mf[0].copy_grown_arrays([g])
    return solver, ng


def _valid_u(solver, ng):
    arr = np.array(solver.U.mf[0].arrays()[0])
    return arr[ng:ng + NX, ng:ng + NY, ng:ng + NZ, :], arr


def test_cylinder_mask_matches_geometry(blockamr_session):
    """The solid-cell count matches the analytic disc area (per z-layer)."""
    solver, ng = _make_cylinder_solver()
    mask = np.array(solver._solid_masks[0][0])[ng:ng + NX, ng:ng + NY, ng:ng + NZ]
    dx, dy = float(LX / NX), float(LY / NY)
    expected = np.pi * RADIUS ** 2 / (dx * dy) * NZ
    # a cell-centre-in-disc mask staircases the boundary — at this coarse
    # resolution (radius ≈ 2.4 cells) it under-counts the smooth area by ~12%.
    assert mask.sum() == pytest.approx(expected, rel=0.2)
    assert mask.sum() > 0


def test_cylinder_wake_and_noslip(blockamr_session):
    """Direct forcing holds U=0 in the body and produces a sane wake."""
    solver, ng = _make_cylinder_solver()
    for _ in range(150):
        solver.step()

    u, arr = _valid_u(solver, ng)
    ux = u[..., 0]
    mask = np.array(solver._solid_masks[0][0])[ng:ng + NX, ng:ng + NY, ng:ng + NZ]

    # bounded & finite
    assert np.all(np.isfinite(arr))
    assert float(np.max(np.abs(arr))) < 2.0 * U0

    # no-slip: velocity pinned to zero inside the body
    assert float(np.max(np.abs(u[mask]))) < 1e-6

    # sane wake structure around the body
    ci, cj = int(CENTER[0] / LX * NX), int(CENTER[1] / LY * NY)
    rc = int(RADIUS / LX * NX)
    u_wake = float(np.mean(ux[ci + rc + 2:ci + rc + 6, cj - 2:cj + 2, :]))
    u_flank = float(np.mean(ux[ci - 2:ci + 2, cj + rc + 1:cj + rc + 3, :]))
    u_stag = float(np.mean(ux[ci - rc - 4:ci - rc - 1, cj - 2:cj + 2, :]))

    assert u_wake < 0.6 * U0        # wake deficit behind the body
    assert u_flank > 1.05 * U0      # accelerated over the flanks
    assert u_stag < 0.9 * U0        # decelerated at the upstream stagnation
