# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""Inlet -> outlet channel through-flow for the DSL incompressible solver.

Exercises a non-periodic (open) domain: fixed-value inlet, zero-gradient
("streamed" Neumann) outlet, free-stream walls, periodic in the thin
direction. The outlet is only well-posed because the pressure projection gives
the outflow face a Dirichlet pressure (paired with its Neumann velocity) — the
closed/all-Neumann projection cannot let flow leave the domain.

Proves the outlet works: mass is conserved (inflow == outflow), the field stays
bounded, and an injected disturbance relaxes back toward the analytic plug-flow
steady state.
"""

import numpy as np
import pytest

import neon.blockamr as blockamr
from neon.blockamr.bc import VectorBC, fixedValue, NeumannBC
from neon.blockamr.dsl_solver import DSLIncompressibleSolver
from neon.blockamr.mesh import Mesh

U0 = 1.0
NU = 0.01
NX, NY, NZ = 32, 16, 8
LX, LY, LZ = 2.0, 1.0, 0.25


def _make_channel():
    box = blockamr.Box([0, 0, 0], [NX - 1, NY - 1, NZ - 1])
    rb = blockamr.RealBox([0.0, 0.0, 0.0], [LX, LY, LZ])
    geom = blockamr.Geometry(box, rb, 0, [0, 0, 1])  # x,y non-periodic; z periodic
    ba = blockamr.BoxArray(box)
    ba.max_size(max(NX, NY, NZ))
    dm = blockamr.DistributionMapping(ba)
    mesh = Mesh(ba, dm, geom)

    u_bc = VectorBC(
        xlo=fixedValue([U0, 0.0, 0.0]),  # inlet
        xhi=NeumannBC(),  # outlet (streamed / zeroGradient)
        ylo=fixedValue([U0, 0.0, 0.0]),  # free-stream walls
        yhi=fixedValue([U0, 0.0, 0.0]),
    )
    dt = 0.25 * float(geom.cell_size()[0]) / U0
    # This free-stream-wall channel is one of the outflow cases where the Krylov
    # nodal bottom solvers diverge; it needs the relaxation smoother. (The slip-
    # wall immersed-cylinder outflow, by contrast, converges in ~5 V-cycles with
    # the default Krylov bottom — hence the bottom solver is a per-case scheme,
    # not a hardcoded default.)
    sol_p = {"rtol": 1e-10, "atol": 1e-12, "maxIter": 200, "bottomSolver": "smoother"}
    return DSLIncompressibleSolver(mesh, NU, dt, U_bc=u_bc, sol_p=sol_p)


def _seed_plug_flow_with_blob(solver, amp):
    """Free-stream U=(U0,0,0) plus a localized streamwise disturbance."""
    ng = solver.U.mf[0].n_grow()
    g = solver.U.mf[0].grown_arrays()[0]
    g = g.at[:, :, :, 0].set(U0).at[:, :, :, 1:].set(0.0)
    cx, cy = ng + NX // 3, ng + NY // 2
    g = g.at[cx - 2 : cx + 2, cy - 2 : cy + 2, :, 0].set((1.0 + amp) * U0)
    solver.U.mf[0].copy_grown_arrays([g])
    return ng


def _interior_u(solver, ng):
    arr = np.array(solver.U.mf[0].arrays()[0])
    return arr[ng : ng + NX, ng : ng + NY, ng : ng + NZ, 0]


def test_channel_outflow_mass_conserved_and_stable(blockamr_session):
    """Through-flow stays bounded, conserves mass, and relaxes to plug flow."""
    solver = _make_channel()
    ng = _seed_plug_flow_with_blob(solver, amp=0.2)

    u0 = _interior_u(solver, ng)
    initial_disturbance = float(np.max(np.abs(u0 - U0)))
    assert initial_disturbance > 0.1  # the blob is really there

    for _ in range(80):
        solver.step()

    u = _interior_u(solver, ng)
    arr = np.array(solver.U.mf[0].arrays()[0])

    # (1) bounded and finite — the outflow projection did not blow up
    assert np.all(np.isfinite(arr))
    assert float(np.max(np.abs(arr))) < 2.0 * U0

    # (2) mass conservation: streamwise flux in == out
    inlet_flux = float(np.sum(u[0, :, :]))
    outlet_flux = float(np.sum(u[-1, :, :]))
    rel_diff = abs(inlet_flux - outlet_flux) / abs(inlet_flux)
    assert rel_diff < 1e-3, f"mass imbalance {rel_diff:.2e}"

    # (3) the disturbance decays toward the analytic plug-flow steady state
    final_disturbance = float(np.max(np.abs(u - U0)))
    assert final_disturbance < 0.1 * initial_disturbance
    assert float(np.mean(u)) == pytest.approx(U0, abs=1e-2)
