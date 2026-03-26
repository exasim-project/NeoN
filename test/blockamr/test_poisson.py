# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""Tests for MLPoisson / MLABecLaplacian / MLMG bindings."""

import math
import numpy as np
import neon.blockamr as blockamr
from neon.blockamr.mesh import Mesh


def _make_mesh(n, is_per=None):
    """Create a single-level mesh on [0,1]^3."""
    if is_per is None:
        is_per = [0, 0, 0]
    box = blockamr.Box([0, 0, 0], [n - 1, n - 1, n - 1])
    rb = blockamr.RealBox([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    geom = blockamr.Geometry(box, rb, 0, is_per)
    ba = blockamr.BoxArray(box)
    ba.max_size(n)
    dm = blockamr.DistributionMapping(ba)
    return Mesh(ba, dm, geom), geom, ba, dm


def test_linop_bctype_enum(blockamr_session):
    """LinOpBCType enum values are accessible."""
    assert blockamr.LinOpBCType.Dirichlet is not None
    assert blockamr.LinOpBCType.Neumann is not None
    assert blockamr.LinOpBCType.Periodic is not None
    assert blockamr.LinOpBCType.interior is not None


def test_lpinfo(blockamr_session):
    """LPInfo can be created and configured."""
    info = blockamr.LPInfo()
    ret = info.set_max_coarsening_level(5)
    assert ret is info  # chaining


def test_mlpoisson_trivial(blockamr_session):
    """Solve del^2(phi) = 0 with Neumann BCs — trivial solution phi=0."""
    N = 16
    _, geom, ba, dm = _make_mesh(N)

    lp = blockamr.MLPoisson(geom, ba, dm)
    lp.set_domain_bc(
        [blockamr.LinOpBCType.Neumann] * 3,
        [blockamr.LinOpBCType.Neumann] * 3,
    )
    lp.set_level_bc(0, None)

    sol = blockamr.MultiFab(ba, dm, 1, 1)
    rhs = blockamr.MultiFab(ba, dm, 1, 0)

    mlmg = blockamr.MLMG(lp)
    mlmg.set_verbose(0)
    mlmg.set_max_iter(100)
    res = mlmg.solve(sol, rhs, 1e-10, 1e-12)
    assert res < 1e-10


def test_mlpoisson_dirichlet_sine(blockamr_session):
    """Solve del^2(phi) = f with known analytical solution on [0,1]^3.

    f = -12*pi^2 * sin(2*pi*x) * sin(2*pi*y) * sin(2*pi*z)
    phi_exact = sin(2*pi*x) * sin(2*pi*y) * sin(2*pi*z)
    BC: Dirichlet phi=0 on all faces (sin vanishes at x,y,z = 0 and 1).
    """
    N = 32
    _, geom, ba, dm = _make_mesh(N, is_per=[0, 0, 0])

    lp = blockamr.MLPoisson(geom, ba, dm)
    lp.set_domain_bc(
        [blockamr.LinOpBCType.Dirichlet] * 3,
        [blockamr.LinOpBCType.Dirichlet] * 3,
    )
    lp.set_level_bc(0, None)

    sol = blockamr.MultiFab(ba, dm, 1, 1)  # initial guess = 0
    rhs = blockamr.MultiFab(ba, dm, 1, 0)

    # Fill RHS
    pi = math.pi
    dx = geom.cell_size()
    for mfi in blockamr.MFIterator(rhs):
        arr = rhs.copy_to_host(mfi)
        bx = mfi.valid_box()
        lo = bx.small_end()
        nx, ny, nz = arr.shape[:3]
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    x = (lo[0] + i + 0.5) * dx[0]
                    y = (lo[1] + j + 0.5) * dx[1]
                    z = (lo[2] + k + 0.5) * dx[2]
                    arr[i, j, k, 0] = -12.0 * pi**2 * math.sin(2*pi*x) * math.sin(2*pi*y) * math.sin(2*pi*z)
        rhs.copy_from(mfi, arr)

    mlmg = blockamr.MLMG(lp)
    mlmg.set_verbose(0)
    mlmg.set_max_iter(200)
    mlmg.set_bottom_verbose(0)
    mlmg.solve(sol, rhs, 1e-10, 1e-12)

    # Check solution vs analytical
    max_err = 0.0
    for mfi in blockamr.MFIterator(sol):
        arr = sol.copy_to_host(mfi)
        bx = mfi.valid_box()
        lo = bx.small_end()
        nx, ny, nz = arr.shape[:3]
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    x = (lo[0] + i + 0.5) * dx[0]
                    y = (lo[1] + j + 0.5) * dx[1]
                    z = (lo[2] + k + 0.5) * dx[2]
                    exact = math.sin(2*pi*x) * math.sin(2*pi*y) * math.sin(2*pi*z)
                    max_err = max(max_err, abs(arr[i, j, k, 0] - exact))

    # Second-order scheme on 32 cells: error ~ (pi*dx)^2 ≈ 0.01
    assert max_err < 0.02, f"Max error {max_err} exceeds tolerance"


def test_mlabeclaplacian_helmholtz(blockamr_session):
    """Solve (alpha - beta*del^2) phi = rhs with MLABecLaplacian.

    Verify it can be constructed and solved without errors.
    Uses constant coefficients a=1, b=1 (uniform diffusion).
    """
    N = 16
    _, geom, ba, dm = _make_mesh(N, is_per=[0, 0, 0])

    abec = blockamr.MLABecLaplacian(geom, ba, dm)
    abec.set_domain_bc(
        [blockamr.LinOpBCType.Dirichlet] * 3,
        [blockamr.LinOpBCType.Dirichlet] * 3,
    )
    abec.set_level_bc(0, None)
    abec.set_scalars(1.0, 1.0)  # alpha=1, beta=1

    # Set a_coeffs = 1.0 everywhere
    a_coeff = blockamr.MultiFab(ba, dm, 1, 0)
    for mfi in blockamr.MFIterator(a_coeff):
        arr = a_coeff.copy_to_host(mfi)
        arr[:] = 1.0
        a_coeff.copy_from(mfi, arr)
    abec.set_a_coeffs(0, a_coeff)

    # Set b_coeffs = 1.0 on faces (need face-centred MultiFabs)
    # Create face-centred boxes by converting the cell-centred domain box
    dom = geom.domain()
    face_mfs = []
    for d in range(3):
        face_box = blockamr.Box(dom.small_end(), dom.big_end())
        face_box.surrounding_nodes(d)
        face_ba = blockamr.BoxArray(face_box)
        face_ba.max_size(N)
        face_mfs.append(blockamr.MultiFab(face_ba, dm, 1, 0))
    bx, by, bz = face_mfs
    for mf in (bx, by, bz):
        for mfi in blockamr.MFIterator(mf):
            arr = mf.copy_to_host(mfi)
            arr[:] = 1.0
            mf.copy_from(mfi, arr)
    abec.set_b_coeffs(0, bx, by, bz)

    sol = blockamr.MultiFab(ba, dm, 1, 1)
    rhs = blockamr.MultiFab(ba, dm, 1, 0)
    for mfi in blockamr.MFIterator(rhs):
        arr = rhs.copy_to_host(mfi)
        arr[:] = 1.0
        rhs.copy_from(mfi, arr)

    mlmg = blockamr.MLMG(abec)
    mlmg.set_verbose(0)
    mlmg.set_max_iter(200)
    mlmg.solve(sol, rhs, 1e-10, 1e-12)

    # Solution should be non-trivial and bounded
    for mfi in blockamr.MFIterator(sol):
        arr = sol.copy_to_host(mfi)
        assert np.all(np.isfinite(arr)), "Solution contains non-finite values"
        assert np.max(np.abs(arr)) > 0.0, "Solution is trivially zero"


def test_mlmg_get_grad_solution(blockamr_session):
    """getGradSolution returns face-centred gradient after a solve."""
    N = 16
    _, geom, ba, dm = _make_mesh(N, is_per=[0, 0, 0])

    lp = blockamr.MLPoisson(geom, ba, dm)
    lp.set_domain_bc(
        [blockamr.LinOpBCType.Dirichlet] * 3,
        [blockamr.LinOpBCType.Dirichlet] * 3,
    )
    lp.set_level_bc(0, None)

    sol = blockamr.MultiFab(ba, dm, 1, 1)
    rhs = blockamr.MultiFab(ba, dm, 1, 0)

    # Set rhs = 1.0 so solution is non-trivial
    for mfi in blockamr.MFIterator(rhs):
        arr = rhs.copy_to_host(mfi)
        arr[:] = 1.0
        rhs.copy_from(mfi, arr)

    mlmg = blockamr.MLMG(lp)
    mlmg.set_verbose(0)
    mlmg.solve(sol, rhs, 1e-10, 1e-12)

    # Allocate face-centred MultiFabs for gradient
    dom = geom.domain()
    grad_mfs = []
    for d in range(3):
        face_box = blockamr.Box(dom.small_end(), dom.big_end())
        face_box.surrounding_nodes(d)
        face_ba = blockamr.BoxArray(face_box)
        face_ba.max_size(N)
        grad_mfs.append(blockamr.MultiFab(face_ba, dm, 1, 0))
    gx, gy, gz = grad_mfs

    mlmg.get_grad_solution(gx, gy, gz)

    # Gradient should be non-trivial
    has_nonzero = False
    for mfi in blockamr.MFIterator(gx):
        arr = gx.copy_to_host(mfi)
        if np.max(np.abs(arr)) > 0.0:
            has_nonzero = True
    assert has_nonzero, "Gradient is trivially zero"
