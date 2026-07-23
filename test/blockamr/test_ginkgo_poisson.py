# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""Matrix-free Ginkgo CG solve of an MLPoisson system, validated against MLMG."""

import math
import numpy as np
import pytest
import blockamr
from blockamr.mesh import Mesh


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


def test_ginkgo_poisson_matches_mlmg(blockamr_session):
    """Matrix-free Ginkgo CG matches MLMG on del^2(phi) = f with known solution.

    f = -12*pi^2 * sin(2*pi*x) * sin(2*pi*y) * sin(2*pi*z)
    phi_exact = sin(2*pi*x) * sin(2*pi*y) * sin(2*pi*z)
    BC: Dirichlet phi=0 on all faces (sin vanishes at x,y,z = 0 and 1).
    """
    if not hasattr(blockamr, "ginkgo_solve"):
        pytest.skip("blockamr.ginkgo_solve binding not available")

    N = 32
    _, geom, ba, dm = _make_mesh(N, is_per=[0, 0, 0])

    lp = blockamr.MLPoisson(geom, ba, dm)
    lp.set_domain_bc(
        [blockamr.LinOpBCType.Dirichlet] * 3,
        [blockamr.LinOpBCType.Dirichlet] * 3,
    )
    lp.set_level_bc(0, None)

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

    # Reference: MLMG solve
    sol_ref = blockamr.MultiFab(ba, dm, 1, 1)  # initial guess = 0
    mlmg = blockamr.MLMG(lp)
    mlmg.set_verbose(0)
    mlmg.set_max_iter(200)
    mlmg.set_bottom_verbose(0)
    mlmg.solve(sol_ref, rhs, 1e-10, 1e-12)

    # Matrix-free Ginkgo CG solve (in place into sol_gko)
    sol_gko = blockamr.MultiFab(ba, dm, 1, 1)  # initial guess = 0
    try:
        stats = blockamr.ginkgo_solve(lp, sol_gko, rhs, max_iter=2000, rtol=1e-10)
    except RuntimeError as exc:
        if "without Ginkgo" in str(exc):
            pytest.skip("blockamr built without Ginkgo")
        raise
    assert stats["num_iters"] > 0
    assert math.isfinite(stats["res_norm"])
    assert stats["res_norm"] < 1e-4, f"Residual norm {stats['res_norm']} too large"

    # Primary: Ginkgo solution matches MLMG reference (max-norm over all cells).
    # Only one MFIterator may be active at a time, so collect boxes in two passes.
    gko_boxes = [sol_gko.copy_to_host(mfi) for mfi in blockamr.MFIterator(sol_gko)]
    ref_boxes = [sol_ref.copy_to_host(mfi) for mfi in blockamr.MFIterator(sol_ref)]
    max_diff = max(
        float(np.max(np.abs(g - r))) for g, r in zip(gko_boxes, ref_boxes)
    )
    assert max_diff < 1e-6, f"Max |sol_gko - sol_ref| = {max_diff} exceeds 1e-6"

    # Secondary: Ginkgo solution matches the analytical solution
    max_err = 0.0
    for mfi in blockamr.MFIterator(sol_gko):
        arr = sol_gko.copy_to_host(mfi)
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
