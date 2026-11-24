# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""Matrix-free Ginkgo CG on MLABecLaplacian and MLPoisson with inhomogeneous BCs."""

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


def _cell_centers(lo, shape, dx):
    """Fortran-ordered meshgrids of cell-centre coordinates for a host array."""
    nx, ny, nz = shape[:3]
    x = (lo[0] + np.arange(nx) + 0.5) * dx[0]
    y = (lo[1] + np.arange(ny) + 0.5) * dx[1]
    z = (lo[2] + np.arange(nz) + 0.5) * dx[2]
    return np.meshgrid(x, y, z, indexing="ij")


def _fill_mf(mf, dx, fn):
    """Fill the valid region of a cell-centred MultiFab with fn(x, y, z)."""
    for mfi in blockamr.MFIterator(mf):
        arr = mf.copy_to_host(mfi)
        lo = mfi.valid_box().small_end()
        xg, yg, zg = _cell_centers(lo, arr.shape, dx)
        arr[:, :, :, 0] = fn(xg, yg, zg)
        mf.copy_from(mfi, arr)


def _make_levdata(ba, dm, dx, fn):
    """Cell-centred MultiFab (1 ghost) carrying Dirichlet values for set_level_bc.

    AMReX reads the ghost cells only, interpreting them as the value ON the domain
    face — so ghost cell-centre coordinates are clamped onto [0,1] per axis before
    evaluating fn (the valid cells are ignored, filling them too is harmless).
    """
    levdata = blockamr.MultiFab(ba, dm, 1, 1)
    ng = levdata.n_grow()
    for mfi in blockamr.MFIterator(levdata):
        arr = levdata.copy_grown_to_host(mfi)
        vlo = mfi.valid_box().small_end()
        glo = [vlo[d] - ng for d in range(3)]
        xg, yg, zg = _cell_centers(glo, arr.shape, dx)
        xg = np.clip(xg, 0.0, 1.0)
        yg = np.clip(yg, 0.0, 1.0)
        zg = np.clip(zg, 0.0, 1.0)
        arr[:, :, :, 0] = fn(xg, yg, zg)
        levdata.copy_grown_from(mfi, arr)
    return levdata


def _ginkgo_solve_or_skip(lp, sol, rhs, **kwargs):
    """Call blockamr.ginkgo_solve, skipping if the build has no Ginkgo support."""
    try:
        return blockamr.ginkgo_solve(lp, sol, rhs, **kwargs)
    except RuntimeError as exc:
        if "without Ginkgo" in str(exc):
            pytest.skip("blockamr built without Ginkgo")
        raise


def _max_abs_diff(a, b):
    """Max-norm difference between the valid regions of two MultiFabs.

    Only one MFIterator may be active at a time, so collect boxes in two passes.
    """
    a_boxes = [a.copy_to_host(mfi) for mfi in blockamr.MFIterator(a)]
    b_boxes = [b.copy_to_host(mfi) for mfi in blockamr.MFIterator(b)]
    return max(float(np.max(np.abs(x - y))) for x, y in zip(a_boxes, b_boxes))


def _max_err_vs_analytic(mf, dx, fn):
    """Max-norm error of a MultiFab against fn(x, y, z) at cell centres."""
    max_err = 0.0
    for mfi in blockamr.MFIterator(mf):
        arr = mf.copy_to_host(mfi)
        lo = mfi.valid_box().small_end()
        xg, yg, zg = _cell_centers(lo, arr.shape, dx)
        max_err = max(max_err, float(np.max(np.abs(arr[:, :, :, 0] - fn(xg, yg, zg)))))
    return max_err


def _build_abec_helmholtz(geom, ba, dm, dx, n):
    """MLABecLaplacian with alpha=1, beta=1, varying a-coeff and unit face b-coeffs."""
    abec = blockamr.MLABecLaplacian(geom, ba, dm)
    abec.set_domain_bc(
        [blockamr.LinOpBCType.Dirichlet] * 3,
        [blockamr.LinOpBCType.Dirichlet] * 3,
    )
    abec.set_level_bc(0, None)
    abec.set_scalars(1.0, 1.0)  # alpha=1, beta=1

    # Spatially varying a-coefficient: a = 1 + 0.5*sin(2*pi*x)*sin(2*pi*y)*sin(2*pi*z)
    pi = math.pi
    a_coeff = blockamr.MultiFab(ba, dm, 1, 0)
    _fill_mf(
        a_coeff,
        dx,
        lambda x, y, z: 1.0 + 0.5 * np.sin(2 * pi * x) * np.sin(2 * pi * y) * np.sin(2 * pi * z),
    )
    abec.set_a_coeffs(0, a_coeff)

    # b_coeffs = 1.0 on faces (face-centred MultiFabs)
    dom = geom.domain()
    face_mfs = []
    for d in range(3):
        face_box = blockamr.Box(dom.small_end(), dom.big_end())
        face_box.surrounding_nodes(d)
        face_ba = blockamr.BoxArray(face_box)
        face_ba.max_size(n)
        face_mfs.append(blockamr.MultiFab(face_ba, dm, 1, 0))
    bx, by, bz = face_mfs
    for mf in (bx, by, bz):
        for mfi in blockamr.MFIterator(mf):
            arr = mf.copy_to_host(mfi)
            arr[:] = 1.0
            mf.copy_from(mfi, arr)
    abec.set_b_coeffs(0, bx, by, bz)
    return abec


def test_ginkgo_abeclap_helmholtz(blockamr_session):
    """Ginkgo CG (sign=+1) matches MLMG on a Helmholtz problem with varying a-coeff.

    (a(x,y,z) - del.(del)) phi = rhs with a = 1 + 0.5*sin(2*pi*x)*sin(2*pi*y)*sin(2*pi*z),
    rhs = sin(2*pi*x)*sin(2*pi*y)*sin(2*pi*z), homogeneous Dirichlet on all faces.
    MLABecLaplacian is positive-definite, hence sign=+1.0.
    """
    if not hasattr(blockamr, "ginkgo_solve"):
        pytest.skip("blockamr.ginkgo_solve binding not available")

    N = 16
    _, geom, ba, dm = _make_mesh(N)
    dx = geom.cell_size()

    pi = math.pi
    rhs = blockamr.MultiFab(ba, dm, 1, 0)
    _fill_mf(
        rhs,
        dx,
        lambda x, y, z: np.sin(2 * pi * x) * np.sin(2 * pi * y) * np.sin(2 * pi * z),
    )

    # Reference: MLMG solve
    abec_ref = _build_abec_helmholtz(geom, ba, dm, dx, N)
    sol_ref = blockamr.MultiFab(ba, dm, 1, 1)
    sol_ref.set_val(0.0)  # MultiFabs are not zero-initialized (arena recycling)
    mlmg = blockamr.MLMG(abec_ref)
    mlmg.set_verbose(0)
    mlmg.set_max_iter(200)
    mlmg.solve(sol_ref, rhs, 1e-10, 1e-12)

    # Matrix-free Ginkgo CG on an identically-built second linop
    abec_gko = _build_abec_helmholtz(geom, ba, dm, dx, N)
    sol_gko = blockamr.MultiFab(ba, dm, 1, 1)
    sol_gko.set_val(0.0)
    stats = _ginkgo_solve_or_skip(abec_gko, sol_gko, rhs, max_iter=2000, rtol=1e-10, sign=+1.0)
    assert stats["num_iters"] > 0
    assert stats["res_norm"] < 1e-4, f"Residual norm {stats['res_norm']} too large"

    max_diff = _max_abs_diff(sol_gko, sol_ref)
    assert max_diff < 1e-6, f"Max |sol_gko - sol_ref| = {max_diff} exceeds 1e-6"


def _build_inhom_dirichlet_poisson(n):
    """MLPoisson for phi = x^2+y^2+z^2 with inhomogeneous Dirichlet BCs.

    del^2(phi) = 6 everywhere (true-sign Laplacian), Dirichlet values from the
    manufactured solution via a levdata MultiFab.
    Returns (geom, ba, dm, dx, lp, rhs, phi, levdata) — levdata must stay
    referenced, or its freed arena memory (which holds phi) gets recycled
    into later MultiFab allocations.
    """
    _, geom, ba, dm = _make_mesh(n)
    dx = geom.cell_size()

    def phi(x, y, z):
        return x**2 + y**2 + z**2

    lp = blockamr.MLPoisson(geom, ba, dm)
    lp.set_domain_bc(
        [blockamr.LinOpBCType.Dirichlet] * 3,
        [blockamr.LinOpBCType.Dirichlet] * 3,
    )
    levdata = _make_levdata(ba, dm, dx, phi)
    lp.set_level_bc(0, levdata)

    rhs = blockamr.MultiFab(ba, dm, 1, 0)
    for mfi in blockamr.MFIterator(rhs):
        arr = rhs.copy_to_host(mfi)
        arr[:] = 6.0
        rhs.copy_from(mfi, arr)

    return geom, ba, dm, dx, lp, rhs, phi, levdata


def test_ginkgo_inhomogeneous_dirichlet(blockamr_session):
    """Ginkgo CG matches MLMG on MLPoisson with inhomogeneous Dirichlet BCs.

    Manufactured phi = x^2 + y^2 + z^2 on [0,1]^3, del^2(phi) = 6, Dirichlet
    values on all faces supplied via set_level_bc.
    """
    if not hasattr(blockamr, "ginkgo_solve"):
        pytest.skip("blockamr.ginkgo_solve binding not available")

    N = 32
    _, ba, dm, dx, lp, rhs, phi, _levdata = _build_inhom_dirichlet_poisson(N)

    # Reference: MLMG solve
    sol_ref = blockamr.MultiFab(ba, dm, 1, 1)
    sol_ref.set_val(0.0)  # MultiFabs are not zero-initialized (arena recycling)
    mlmg = blockamr.MLMG(lp)
    mlmg.set_verbose(0)
    mlmg.set_max_iter(200)
    mlmg.solve(sol_ref, rhs, 1e-10, 1e-12)

    # Matrix-free Ginkgo CG (default sign = -1.0 for MLPoisson)
    sol_gko = blockamr.MultiFab(ba, dm, 1, 1)
    sol_gko.set_val(0.0)
    stats = _ginkgo_solve_or_skip(lp, sol_gko, rhs, max_iter=2000, rtol=1e-10)
    assert stats["num_iters"] > 0
    assert stats["res_norm"] < 1e-4, f"Residual norm {stats['res_norm']} too large"

    max_diff = _max_abs_diff(sol_gko, sol_ref)
    assert max_diff < 1e-6, f"Max |sol_gko - sol_ref| = {max_diff} exceeds 1e-6"

    # Secondary: match the manufactured solution at cell centres
    max_err = _max_err_vs_analytic(sol_gko, dx, phi)
    assert max_err < 2e-3, f"Max error {max_err} vs analytic exceeds 2e-3"


def test_ginkgo_mixed_dirichlet_neumann(blockamr_session):
    """Ginkgo CG matches MLMG on MLPoisson with mixed Neumann/Dirichlet BCs.

    Manufactured phi = cos(pi*x)*(y^2 + z^2): d(phi)/dx = 0 at x=0 and x=1, so
    Neumann on both x-faces and Dirichlet on y/z faces.
    del^2(phi) = cos(pi*x)*(4 - pi^2*(y^2 + z^2)).
    """
    if not hasattr(blockamr, "ginkgo_solve"):
        pytest.skip("blockamr.ginkgo_solve binding not available")

    N = 32
    _, geom, ba, dm = _make_mesh(N)
    dx = geom.cell_size()
    pi = math.pi

    def phi(x, y, z):
        return np.cos(pi * x) * (y**2 + z**2)

    lp = blockamr.MLPoisson(geom, ba, dm)
    lp.set_domain_bc(
        [
            blockamr.LinOpBCType.Neumann,
            blockamr.LinOpBCType.Dirichlet,
            blockamr.LinOpBCType.Dirichlet,
        ],
        [
            blockamr.LinOpBCType.Neumann,
            blockamr.LinOpBCType.Dirichlet,
            blockamr.LinOpBCType.Dirichlet,
        ],
    )
    # Ghosts filled with clamped phi everywhere — harmless on the Neumann x-faces.
    levdata = _make_levdata(ba, dm, dx, phi)
    lp.set_level_bc(0, levdata)

    rhs = blockamr.MultiFab(ba, dm, 1, 0)
    _fill_mf(rhs, dx, lambda x, y, z: np.cos(pi * x) * (4.0 - pi**2 * (y**2 + z**2)))

    # Reference: MLMG solve
    sol_ref = blockamr.MultiFab(ba, dm, 1, 1)
    sol_ref.set_val(0.0)  # MultiFabs are not zero-initialized (arena recycling)
    mlmg = blockamr.MLMG(lp)
    mlmg.set_verbose(0)
    mlmg.set_max_iter(200)
    mlmg.solve(sol_ref, rhs, 1e-10, 1e-12)

    # Matrix-free Ginkgo CG — Neumann faces slow CG down, allow a generous budget
    sol_gko = blockamr.MultiFab(ba, dm, 1, 1)
    sol_gko.set_val(0.0)
    stats = _ginkgo_solve_or_skip(lp, sol_gko, rhs, max_iter=4000, rtol=1e-10)
    assert stats["num_iters"] > 0
    assert stats["res_norm"] < 1e-4, f"Residual norm {stats['res_norm']} too large"

    max_diff = _max_abs_diff(sol_gko, sol_ref)
    assert max_diff < 1e-6, f"Max |sol_gko - sol_ref| = {max_diff} exceeds 1e-6"

    # Secondary: 2nd-order discretization on 32 cells with Neumann faces — loose bound
    max_err = _max_err_vs_analytic(sol_gko, dx, phi)
    assert max_err < 5e-3, f"Max error {max_err} vs analytic exceeds 5e-3"


def test_ginkgo_warm_start(blockamr_session):
    """A second ginkgo_solve on an already-converged sol takes (almost) no iterations.

    Proves the incoming values of sol are used as the initial guess (the solver
    works in residual-correction form internally).
    """
    if not hasattr(blockamr, "ginkgo_solve"):
        pytest.skip("blockamr.ginkgo_solve binding not available")

    N = 32
    _, ba, dm, _, lp, rhs, _, _levdata = _build_inhom_dirichlet_poisson(N)

    sol = blockamr.MultiFab(ba, dm, 1, 1)
    sol.set_val(0.0)
    stats_cold = _ginkgo_solve_or_skip(lp, sol, rhs, max_iter=2000, rtol=1e-10)
    assert stats_cold["num_iters"] > 5, "Cold start converged suspiciously fast"
    assert stats_cold["res_norm"] < 1e-4

    # Warm start: sol is already converged, so the solver should stop immediately
    stats_warm = _ginkgo_solve_or_skip(lp, sol, rhs, max_iter=2000, rtol=1e-10)
    assert stats_warm["num_iters"] <= 5, (
        f"Warm start took {stats_warm['num_iters']} iterations — initial guess ignored?"
    )
