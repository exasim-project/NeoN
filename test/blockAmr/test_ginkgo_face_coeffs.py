# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""Matrix-free Ginkgo solve of a general structured face-coefficient system.

The face-coefficient operator carries the matrix as OpenFOAM-style AMReX fields:
a cell-centred diagonal source ``alpha`` plus face-centred upper/lower
off-diagonal coefficients, with the full diagonal assembled on the fly as
``alpha - negSumDiag(faces)``. Two checks:

* symmetric Helmholtz (a - laplacian) vs an identically-defined MLABecLaplacian
  solved by MLMG — proves the face-coeff apply + CG reproduce the trusted operator;
* asymmetric convection-diffusion-reaction (upper != lower on the x-faces) vs a
  manufactured analytic solution, with 2nd-order convergence.
"""

import math

import numpy as np
import pytest

import blockamr


def _make_periodic_mesh(n):
    """Single-box periodic mesh on [0,1]^3 with n cells per side."""
    box = blockamr.Box([0, 0, 0], [n - 1, n - 1, n - 1])
    rb = blockamr.RealBox([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    geom = blockamr.Geometry(box, rb, 0, [1, 1, 1])
    ba = blockamr.BoxArray(box)
    ba.max_size(n)  # single box -> face fabs align 1:1 with the cell fab
    dm = blockamr.DistributionMapping(ba)
    return geom, ba, dm


def _cell_centers(lo, shape, dx):
    """Fortran-ordered cell-centre coordinate meshgrids for a host array."""
    nx, ny, nz = shape[:3]
    x = (lo[0] + np.arange(nx) + 0.5) * dx[0]
    y = (lo[1] + np.arange(ny) + 0.5) * dx[1]
    z = (lo[2] + np.arange(nz) + 0.5) * dx[2]
    return np.meshgrid(x, y, z, indexing="ij")


def _fill_cell(mf, dx, fn):
    """Fill the valid region of a cell-centred MultiFab with fn(x, y, z)."""
    for mfi in blockamr.MFIterator(mf):
        arr = mf.copy_to_host(mfi)
        lo = mfi.valid_box().small_end()
        xg, yg, zg = _cell_centers(lo, arr.shape, dx)
        arr[:, :, :, 0] = fn(xg, yg, zg)
        mf.copy_from(mfi, arr)


def _const_cell(ba, dm, value):
    """Cell-centred MultiFab (no ghost) filled with a constant."""
    mf = blockamr.MultiFab(ba, dm, 1, 0)
    mf.set_val(value)
    return mf


def _const_face(geom, dm, d, n, value):
    """Face-centred MultiFab in direction d filled with a constant."""
    dom = geom.domain()
    face_box = blockamr.Box(dom.small_end(), dom.big_end())
    face_box.surrounding_nodes(d)
    face_ba = blockamr.BoxArray(face_box)
    face_ba.max_size(n)
    mf = blockamr.MultiFab(face_ba, dm, 1, 0)
    mf.set_val(value)
    return mf


def _max_abs_diff(a, b):
    """Max-norm difference between the valid regions of two cell MultiFabs."""
    a_boxes = [a.copy_to_host(mfi) for mfi in blockamr.MFIterator(a)]
    b_boxes = [b.copy_to_host(mfi) for mfi in blockamr.MFIterator(b)]
    return max(float(np.max(np.abs(x - y))) for x, y in zip(a_boxes, b_boxes))


def _max_err_vs_analytic(mf, dx, fn):
    """Max-norm error of a cell MultiFab against fn(x, y, z) at cell centres."""
    max_err = 0.0
    for mfi in blockamr.MFIterator(mf):
        arr = mf.copy_to_host(mfi)
        lo = mfi.valid_box().small_end()
        xg, yg, zg = _cell_centers(lo, arr.shape, dx)
        max_err = max(max_err, float(np.max(np.abs(arr[:, :, :, 0] - fn(xg, yg, zg)))))
    return max_err


def _face_coeffs_solve_or_skip(**kwargs):
    """Call blockamr.ginkgo_solve_face_coeffs, skipping if the build has no Ginkgo."""
    try:
        return blockamr.ginkgo_solve_face_coeffs(**kwargs)
    except RuntimeError as exc:
        if "without Ginkgo" in str(exc):
            pytest.skip("blockamr built without Ginkgo")
        raise


def test_face_coeffs_symmetric_matches_mlmg(blockamr_session):
    """Face-coeff CG matches MLMG on a periodic Helmholtz (a - laplacian) problem.

    MLABecLaplacian with alpha=beta=a=b=1 is the operator (phi - laplacian phi);
    its face-coefficient form is a constant diagonal source alpha=1 with symmetric
    face coefficients -1/dx^2. Both solve the SAME matrix, so the solutions agree.
    """
    if not hasattr(blockamr, "ginkgo_solve_face_coeffs"):
        pytest.skip("blockamr.ginkgo_solve_face_coeffs binding not available")

    N = 32
    geom, ba, dm = _make_periodic_mesh(N)
    dx = geom.cell_size()
    pi = math.pi

    def rhs_fn(x, y, z):
        # Sum of several periodic modes (not a single eigenvector) so CG must
        # iterate across the spectrum — the MLMG match then exercises the whole
        # operator, not just one eigenmode.
        return (
            np.sin(2 * pi * x) * np.sin(2 * pi * y) * np.sin(2 * pi * z)
            + np.sin(4 * pi * x) * np.sin(2 * pi * y)
            + np.cos(2 * pi * x) * np.cos(4 * pi * z)
            + 0.5
        )

    rhs = blockamr.MultiFab(ba, dm, 1, 0)
    _fill_cell(rhs, dx, rhs_fn)

    # Reference: MLABecLaplacian (phi - laplacian phi), periodic, solved by MLMG.
    abec = blockamr.MLABecLaplacian(geom, ba, dm)
    abec.set_domain_bc(
        [blockamr.LinOpBCType.Periodic] * 3,
        [blockamr.LinOpBCType.Periodic] * 3,
    )
    abec.set_level_bc(0, None)
    abec.set_scalars(1.0, 1.0)  # alpha_scalar, beta_scalar
    abec.set_a_coeffs(0, _const_cell(ba, dm, 1.0))
    bx = _const_face(geom, dm, 0, N, 1.0)
    by = _const_face(geom, dm, 1, N, 1.0)
    bz = _const_face(geom, dm, 2, N, 1.0)
    abec.set_b_coeffs(0, bx, by, bz)

    sol_ref = blockamr.MultiFab(ba, dm, 1, 1)
    sol_ref.set_val(0.0)  # MultiFabs are not zero-initialized (arena recycling)
    mlmg = blockamr.MLMG(abec)
    mlmg.set_verbose(0)
    mlmg.set_max_iter(200)
    mlmg.solve(sol_ref, rhs, 1e-11, 1e-13)

    # Face-coeff: diag source alpha=1, symmetric face coeffs -1/dx^2 (cube -> equal dx).
    inv_dx2 = 1.0 / dx[0] ** 2
    alpha = _const_cell(ba, dm, 1.0)
    fx = _const_face(geom, dm, 0, N, -inv_dx2)
    fy = _const_face(geom, dm, 1, N, -inv_dx2)
    fz = _const_face(geom, dm, 2, N, -inv_dx2)

    sol_fc = blockamr.MultiFab(ba, dm, 1, 1)
    sol_fc.set_val(0.0)
    stats = _face_coeffs_solve_or_skip(
        alpha=alpha,
        ux=fx,
        lx=fx,
        uy=fy,
        ly=fy,
        uz=fz,
        lz=fz,
        sol=sol_fc,
        rhs=rhs,
        geom=geom,
        solver="cg",
        max_iter=2000,
        rtol=1e-11,
    )
    assert stats["num_iters"] > 1, "multi-mode rhs should take more than one CG iteration"
    assert stats["res_norm"] < 1e-6, f"Residual norm {stats['res_norm']} too large"

    max_diff = _max_abs_diff(sol_fc, sol_ref)
    assert max_diff < 1e-6, f"Max |sol_fc - sol_mlmg| = {max_diff} exceeds 1e-6"


def _solve_conv_diff(n, u0, gamma, c):
    """Solve c*phi + u0*d/dx(phi) - gamma*laplacian(phi) = rhs on periodic [0,1]^3.

    Central differencing makes the x-faces asymmetric (upper != lower). The
    manufactured solution is phi = sin(2*pi*x); the rhs is the analytic operator
    applied to it. Returns the max-norm error of the solve vs phi.
    """
    geom, ba, dm = _make_periodic_mesh(n)
    dx = geom.cell_size()
    pi = math.pi

    def phi(x, y, z):
        return np.sin(2 * pi * x)

    def rhs_fn(x, y, z):
        return (
            c * np.sin(2 * pi * x)
            + u0 * 2 * pi * np.cos(2 * pi * x)
            + gamma * (2 * pi) ** 2 * np.sin(2 * pi * x)
        )

    rhs = blockamr.MultiFab(ba, dm, 1, 0)
    _fill_cell(rhs, dx, rhs_fn)

    inv_dx2 = 1.0 / dx[0] ** 2
    inv_dy2 = 1.0 / dx[1] ** 2
    inv_dz2 = 1.0 / dx[2] ** 2

    # x-faces: central convection makes upper != lower.
    ux_val = -gamma * inv_dx2 + u0 / (2.0 * dx[0])
    lx_val = -gamma * inv_dx2 - u0 / (2.0 * dx[0])
    assert abs(ux_val - lx_val) > 1e-12, "x-faces are symmetric — not an asymmetric test"

    alpha = _const_cell(ba, dm, c)
    ux = _const_face(geom, dm, 0, n, ux_val)
    lx = _const_face(geom, dm, 0, n, lx_val)
    fy = _const_face(geom, dm, 1, n, -gamma * inv_dy2)
    fz = _const_face(geom, dm, 2, n, -gamma * inv_dz2)

    sol = blockamr.MultiFab(ba, dm, 1, 1)
    sol.set_val(0.0)
    stats = _face_coeffs_solve_or_skip(
        alpha=alpha,
        ux=ux,
        lx=lx,
        uy=fy,
        ly=fy,
        uz=fz,
        lz=fz,
        sol=sol,
        rhs=rhs,
        geom=geom,
        solver="bicgstab",
        max_iter=4000,
        rtol=1e-10,
    )
    assert stats["num_iters"] > 0
    assert stats["res_norm"] < 1e-5, f"Residual norm {stats['res_norm']} too large"

    return _max_err_vs_analytic(sol, dx, phi)


def test_face_coeffs_asymmetric_convection_diffusion(blockamr_session):
    """BiCGStab on an asymmetric convection-diffusion-reaction matrix vs analytic.

    c*phi + u0*d(phi)/dx - gamma*laplacian(phi) = rhs, periodic, manufactured
    phi = sin(2*pi*x). Central differencing gives a genuinely non-symmetric
    matrix (verified inside _solve_conv_diff) and 2nd-order accuracy, so the
    max-norm error is small and drops ~4x under grid halving.
    """
    if not hasattr(blockamr, "ginkgo_solve_face_coeffs"):
        pytest.skip("blockamr.ginkgo_solve_face_coeffs binding not available")

    u0, gamma, c = 1.0, 0.02, 1.0  # cell-Peclet u0*dx/gamma < 2 -> stable central

    err_coarse = _solve_conv_diff(64, u0, gamma, c)
    err_fine = _solve_conv_diff(128, u0, gamma, c)

    assert err_coarse < 5e-2, f"Coarse error {err_coarse} too large"
    assert err_fine < 1.5e-2, f"Fine error {err_fine} too large"

    ratio = err_coarse / err_fine
    assert 3.2 < ratio < 4.8, f"Convergence ratio {ratio} not ~2nd order (expected ~4)"
