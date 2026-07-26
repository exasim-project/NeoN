# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

import numpy as np
import blockamr
from blockamr.bc import DirichletBC, NeumannBC, BoundaryCondition
from blockamr.field import CellField
from blockamr.fillpatch import FillPatchWithBC
from blockamr.mesh import Mesh


def _make_nonperiodic_mesh(n=8, max_size=8):
    """Create a non-periodic single-level mesh on [0,1]^3."""
    box = blockamr.Box([0, 0, 0], [n - 1, n - 1, n - 1])
    rb = blockamr.RealBox([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    geom = blockamr.Geometry(box, rb, 0, [0, 0, 0])
    ba = blockamr.BoxArray(box)
    ba.max_size(max_size)
    dm = blockamr.DistributionMapping(ba)
    return Mesh(ba, dm, geom), geom


def _init_constant(phi, value):
    """Set all cells (including ghosts) to a constant value."""
    mf = phi.mf[0]
    for mfi in blockamr.MFIterator(mf):
        arr = np.array(mf.copy_grown_to_host(mfi), order='F')
        arr[:] = value
        mf.copy_grown_from(mfi, np.asfortranarray(arr))


def _read_grown(phi):
    """Read back the full grown array (ghosts included) as numpy."""
    mf = phi.mf[0]
    for mfi in blockamr.MFIterator(mf):
        return np.array(mf.copy_grown_to_host(mfi), order='F')


def test_geometry_domain(blockamr_session):
    """Geometry.domain() returns the correct domain box."""
    box = blockamr.Box([0, 0, 0], [15, 15, 15])
    rb = blockamr.RealBox([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    geom = blockamr.Geometry(box, rb, 0, [0, 0, 1])
    dom = geom.domain()
    assert list(dom.small_end()) == [0, 0, 0]
    assert list(dom.big_end()) == [15, 15, 15]
    is_per = geom.is_periodic()
    assert list(is_per) == [0, 0, 1]


def test_dirichlet_bc(blockamr_session):
    """Dirichlet BC: ghost = 2*bc_value - interior."""
    mesh, geom = _make_nonperiodic_mesh(n=8)
    bc = BoundaryCondition(
        lo=[DirichletBC(0.0)] * 3,
        hi=[DirichletBC(1.0), DirichletBC(2.0), DirichletBC(3.0)],
    )
    phi = CellField(mesh, ncomp=1, ngrow=1, name="phi",
                     fill_patch=FillPatchWithBC(bc))

    _init_constant(phi, 0.5)
    phi.fill_patch(0, 0.0)

    arr = _read_grown(phi)
    ng = 1
    # x_lo ghost: 2*0.0 - 0.5 = -0.5
    assert np.isclose(arr[0, ng, ng, 0], -0.5), f"x_lo ghost: {arr[0, ng, ng, 0]}"
    # x_hi ghost: 2*1.0 - 0.5 = 1.5
    assert np.isclose(arr[-1, ng, ng, 0], 1.5), f"x_hi ghost: {arr[-1, ng, ng, 0]}"
    # y_lo ghost: 2*0.0 - 0.5 = -0.5
    assert np.isclose(arr[ng, 0, ng, 0], -0.5), f"y_lo ghost: {arr[ng, 0, ng, 0]}"
    # y_hi ghost: 2*2.0 - 0.5 = 3.5
    assert np.isclose(arr[ng, -1, ng, 0], 3.5), f"y_hi ghost: {arr[ng, -1, ng, 0]}"
    # z_lo ghost: 2*0.0 - 0.5 = -0.5
    assert np.isclose(arr[ng, ng, 0, 0], -0.5), f"z_lo ghost: {arr[ng, ng, 0, 0]}"
    # z_hi ghost: 2*3.0 - 0.5 = 5.5
    assert np.isclose(arr[ng, ng, -1, 0], 5.5), f"z_hi ghost: {arr[ng, ng, -1, 0]}"


def test_neumann_bc(blockamr_session):
    """Neumann (zero gradient) BC: ghost = interior."""
    mesh, geom = _make_nonperiodic_mesh(n=8)
    bc = BoundaryCondition(
        lo=[NeumannBC()] * 3,
        hi=[NeumannBC()] * 3,
    )
    phi = CellField(mesh, ncomp=1, ngrow=1, name="phi",
                     fill_patch=FillPatchWithBC(bc))

    # Set interior to a linear ramp in x: phi(i) = i + 0.5
    mf = phi.mf[0]
    for mfi in blockamr.MFIterator(mf):
        arr = mf.copy_to_host(mfi)
        ng = 1
        n = arr.shape[0]
        for i in range(n):
            arr[i, :, :, :] = float(i) + 0.5
        mf.copy_from(mfi, arr)

    phi.fill_patch(0, 0.0)

    arr = _read_grown(phi)
    ng = 1
    # x_lo ghost should equal first interior cell (0.5)
    assert np.isclose(arr[0, ng, ng, 0], 0.5), f"x_lo ghost: {arr[0, ng, ng, 0]}"
    # x_hi ghost should equal last interior cell (7.5)
    assert np.isclose(arr[-1, ng, ng, 0], 7.5), f"x_hi ghost: {arr[-1, ng, ng, 0]}"


def test_mixed_bcs(blockamr_session):
    """Mix Dirichlet and Neumann on different faces."""
    mesh, geom = _make_nonperiodic_mesh(n=8)
    bc = BoundaryCondition(
        lo=[DirichletBC(0.0), NeumannBC(), DirichletBC(0.0)],
        hi=[DirichletBC(1.0), NeumannBC(), DirichletBC(0.0)],
    )
    phi = CellField(mesh, ncomp=1, ngrow=1, name="phi",
                     fill_patch=FillPatchWithBC(bc))

    _init_constant(phi, 0.5)
    phi.fill_patch(0, 0.0)

    arr = _read_grown(phi)
    ng = 1
    # x_lo: Dirichlet(0.0) -> 2*0 - 0.5 = -0.5
    assert np.isclose(arr[0, ng, ng, 0], -0.5)
    # x_hi: Dirichlet(1.0) -> 2*1 - 0.5 = 1.5
    assert np.isclose(arr[-1, ng, ng, 0], 1.5)
    # y_lo: Neumann -> 0.5
    assert np.isclose(arr[ng, 0, ng, 0], 0.5)
    # y_hi: Neumann -> 0.5
    assert np.isclose(arr[ng, -1, ng, 0], 0.5)


def test_pressure_domain_bc_inlet_outlet(blockamr_session):
    """pressure_domain_bc pairs a velocity VectorBC with the matching pressure
    LinOpBCType: velocity-Dirichlet face (inlet/wall) -> pressure Neumann,
    velocity-Neumann face (outflow) -> pressure Dirichlet, periodic -> Periodic.
    """
    from blockamr.bc import (
        pressure_domain_bc, VectorBC, fixedValue, noSlip, NeumannBC,
    )

    box = blockamr.Box([0, 0, 0], [15, 15, 7])
    rb = blockamr.RealBox([0.0, 0.0, 0.0], [2.0, 1.0, 0.25])
    geom = blockamr.Geometry(box, rb, 0, [0, 0, 1])  # x,y walls; z periodic

    u_bc = VectorBC(
        xlo=fixedValue([1.0, 0.0, 0.0]),  # inlet
        xhi=NeumannBC(),                  # outflow
        ylo=noSlip(),                     # wall
        yhi=noSlip(),                     # wall
    )
    lo_bc, hi_bc = pressure_domain_bc(u_bc, geom)

    BC = blockamr.LinOpBCType
    # x: inlet -> Neumann (lo), outflow -> Dirichlet (hi)
    assert lo_bc[0] == BC.Neumann
    assert hi_bc[0] == BC.Dirichlet
    # y walls -> Neumann both sides
    assert lo_bc[1] == BC.Neumann
    assert hi_bc[1] == BC.Neumann
    # z periodic -> Periodic both sides
    assert lo_bc[2] == BC.Periodic
    assert hi_bc[2] == BC.Periodic


def test_pressure_domain_bc_all_walls_all_neumann(blockamr_session):
    """A fully-walled (lid-cavity-style) domain has no outflow, so every
    non-periodic pressure face is Neumann (the closed/singular case)."""
    from blockamr.bc import pressure_domain_bc, VectorBC, fixedValue, noSlip

    box = blockamr.Box([0, 0, 0], [7, 7, 7])
    rb = blockamr.RealBox([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    geom = blockamr.Geometry(box, rb, 0, [0, 0, 1])

    u_bc = VectorBC(xlo=noSlip(), xhi=noSlip(),
                    ylo=noSlip(), yhi=fixedValue([1.0, 0.0, 0.0]))
    lo_bc, hi_bc = pressure_domain_bc(u_bc, geom)

    BC = blockamr.LinOpBCType
    assert lo_bc[0] == BC.Neumann and hi_bc[0] == BC.Neumann
    assert lo_bc[1] == BC.Neumann and hi_bc[1] == BC.Neumann
    assert lo_bc[2] == BC.Periodic and hi_bc[2] == BC.Periodic
    # no Dirichlet anywhere -> the closed all-Neumann pressure system
    assert not any(bc == BC.Dirichlet for bc in (*lo_bc, *hi_bc))
