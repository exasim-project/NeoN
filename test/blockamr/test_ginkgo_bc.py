# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""Non-periodic boundary conditions of the matrix-free FaceCoeffSolver.

The ``bc`` constructor kwarg (6 entries: xlo, xhi, ylo, yhi, zlo, zhi) folds
homogeneous Dirichlet (u = 0 on the face, ghost = -interior) and homogeneous
Neumann (du/dn = 0, ghost = interior) domain BCs into the face-coefficient
stencil via a ghost-cell reflect fill — the matrix itself is untouched, and the
caller supplies face coefficients on ALL faces including boundary ones. Checks:

* manufactured solutions with 2nd-order grid convergence for all-Dirichlet,
  all-Neumann and mixed BCs;
* agreement with an identically-discretised MLABecLaplacian solved by MLMG
  (``set_max_order(2)`` makes MLMG's Dirichlet ghost fill linear = ours);
* validation errors (bc vs geometry periodicity, CSR solver periodic-only);
* the singular all-Neumann pure Poisson composes with project_nullspace.
"""

import math

import numpy as np
import pytest

import blockamr


def _make_mesh(n, periodic):
    """Single-box mesh on [0,1]^3 with n cells per side and given periodicity."""
    box = blockamr.Box([0, 0, 0], [n - 1, n - 1, n - 1])
    rb = blockamr.RealBox([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    geom = blockamr.Geometry(box, rb, 0, periodic)
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


def _random_rhs(ba, dm, seed=42):
    """Cell MultiFab with seeded random values — full spectrum, so CG must iterate."""
    rng = np.random.default_rng(seed)
    rhs = blockamr.MultiFab(ba, dm, 1, 0)
    for mfi in blockamr.MFIterator(rhs):
        arr = rhs.copy_to_host(mfi)
        arr[:, :, :, 0] = rng.standard_normal(arr.shape[:3])
        rhs.copy_from(mfi, arr)
    return rhs


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


def _mean(mf):
    """Arithmetic mean of a cell MultiFab's valid values (uniform cells)."""
    parts = [mf.copy_to_host(mfi)[:, :, :, 0] for mfi in blockamr.MFIterator(mf)]
    return float(np.sum([np.sum(p) for p in parts]) / np.sum([p.size for p in parts]))


def _poisson_coeffs(geom, ba, dm, n, alpha_val):
    """alpha=alpha_val cell source + symmetric -1/dx^2 face coeffs on ALL faces."""
    dx = geom.cell_size()
    inv_dx2 = 1.0 / dx[0] ** 2
    alpha = _const_cell(ba, dm, alpha_val)
    fx = _const_face(geom, dm, 0, n, -inv_dx2)
    fy = _const_face(geom, dm, 1, n, -inv_dx2)
    fz = _const_face(geom, dm, 2, n, -inv_dx2)
    return alpha, fx, fy, fz


def _make_solver_or_skip(cls, coeffs, geom, executor, **kwargs):
    """Construct a persistent solver, skipping if Ginkgo/CUDA are unavailable."""
    if not hasattr(blockamr, cls):
        pytest.skip(f"blockamr.{cls} binding not available")
    alpha, fx, fy, fz = coeffs
    try:
        return getattr(blockamr, cls)(
            alpha, fx, fx, fy, fy, fz, fz, geom, executor=executor, **kwargs
        )
    except RuntimeError as exc:
        if "without Ginkgo" in str(exc):
            pytest.skip("blockamr built without Ginkgo")
        if executor == "cuda":
            pytest.skip(f"cuda executor unavailable: {exc}")
        raise


def _zero_sol(ba, dm):
    sol = blockamr.MultiFab(ba, dm, 1, 1)
    sol.set_val(0.0)  # MultiFabs are not zero-initialized (arena recycling)
    return sol


def _solve_manufactured(n, executor, bc, alpha_val, u_fn, f_fn):
    """Solve alpha*u - lap u = f on the non-periodic unit cube; max-norm error vs u."""
    geom, ba, dm = _make_mesh(n, [0, 0, 0])
    dx = geom.cell_size()
    coeffs = _poisson_coeffs(geom, ba, dm, n, alpha_val)
    rhs = blockamr.MultiFab(ba, dm, 1, 0)
    _fill_cell(rhs, dx, f_fn)
    sol = _zero_sol(ba, dm)
    s = _make_solver_or_skip(
        "FaceCoeffSolver",
        coeffs,
        geom,
        executor,
        solver="cg",
        max_iter=5000,
        rtol=1e-11,
        bc=bc,
    )
    stats = s.solve(rhs, sol)
    assert stats["converged"] is True
    return _max_err_vs_analytic(sol, dx, u_fn)


@pytest.mark.parametrize("executor", ["reference", "cuda"])
def test_dirichlet_manufactured_second_order(blockamr_session, executor):
    """-lap u = f, u = sin(pi x)sin(pi y)sin(pi z) (u=0 on all boundaries).

    All-Dirichlet bc via ghost reflect-odd: 2nd order, so the max-norm error
    drops ~4x from N=16 to N=32.
    """
    pi = math.pi

    def u_fn(x, y, z):
        return np.sin(pi * x) * np.sin(pi * y) * np.sin(pi * z)

    def f_fn(x, y, z):
        return 3 * pi**2 * u_fn(x, y, z)

    bc = ["dirichlet"] * 6
    err_16 = _solve_manufactured(16, executor, bc, 0.0, u_fn, f_fn)
    err_32 = _solve_manufactured(32, executor, bc, 0.0, u_fn, f_fn)

    assert err_16 < 5e-3, f"N=16 error {err_16} too large"
    assert err_32 < 1.5e-3, f"N=32 error {err_32} too large"
    ratio = err_16 / err_32
    assert ratio > 3, f"Convergence ratio {ratio} not ~2nd order (expected ~4)"


@pytest.mark.parametrize("executor", ["reference", "cuda"])
def test_neumann_manufactured_second_order(blockamr_session, executor):
    """u - lap u = f (Helmholtz, nonsingular), u = cos(pi x)cos(pi y)cos(pi z).

    du/dn = 0 exactly on all boundaries, so the reflect-even (ghost = interior)
    Neumann fill is 2nd order for this u.
    """
    pi = math.pi

    def u_fn(x, y, z):
        return np.cos(pi * x) * np.cos(pi * y) * np.cos(pi * z)

    def f_fn(x, y, z):
        return (1 + 3 * pi**2) * u_fn(x, y, z)

    bc = ["neumann"] * 6
    err_16 = _solve_manufactured(16, executor, bc, 1.0, u_fn, f_fn)
    err_32 = _solve_manufactured(32, executor, bc, 1.0, u_fn, f_fn)

    assert err_16 < 5e-3, f"N=16 error {err_16} too large"
    assert err_32 < 1.5e-3, f"N=32 error {err_32} too large"
    ratio = err_16 / err_32
    assert ratio > 3, f"Convergence ratio {ratio} not ~2nd order (expected ~4)"


@pytest.mark.parametrize("executor", ["reference", "cuda"])
def test_mixed_bc_manufactured_second_order(blockamr_session, executor):
    """Mixed bc: Dirichlet on x faces, Neumann on y/z faces (order check).

    Helmholtz u - lap u = f with u = sin(pi x)cos(pi y)cos(pi z): u = 0 at
    x = 0, 1 and du/dn = 0 at y, z = 0, 1, so both BC kinds are exact.
    """
    pi = math.pi

    def u_fn(x, y, z):
        return np.sin(pi * x) * np.cos(pi * y) * np.cos(pi * z)

    def f_fn(x, y, z):
        return (1 + 3 * pi**2) * u_fn(x, y, z)

    bc = ["dirichlet", "dirichlet", "neumann", "neumann", "neumann", "neumann"]
    err_16 = _solve_manufactured(16, executor, bc, 1.0, u_fn, f_fn)
    err_32 = _solve_manufactured(32, executor, bc, 1.0, u_fn, f_fn)

    assert err_16 < 5e-3, f"N=16 error {err_16} too large"
    assert err_32 < 1.5e-3, f"N=32 error {err_32} too large"
    ratio = err_16 / err_32
    assert ratio > 3, f"Convergence ratio {ratio} not ~2nd order (expected ~4)"


@pytest.mark.parametrize("executor", ["reference", "cuda"])
def test_dirichlet_matches_mlmg(blockamr_session, executor):
    """All-Dirichlet Poisson with a random rhs matches MLMG on the same matrix.

    MLABecLaplacian (a=0, b=1 -> -laplacian) with homogeneous Dirichlet domain
    BCs and set_max_order(2): MLMG's boundary ghost fill is then linear
    (ghost = -interior), identical to the FaceCoeffSolver reflect-odd fill, so
    the two solve the SAME matrix and agree to solver tolerance. The random
    rhs exercises every row (boundary rows included), not just one eigenmode.
    """
    N = 32
    geom, ba, dm = _make_mesh(N, [0, 0, 0])
    coeffs = _poisson_coeffs(geom, ba, dm, N, 0.0)
    rhs = _random_rhs(ba, dm)

    # Reference: MLABecLaplacian -laplacian, Dirichlet, linear ghost fill.
    abec = blockamr.MLABecLaplacian(geom, ba, dm)
    abec.set_domain_bc(
        [blockamr.LinOpBCType.Dirichlet] * 3,
        [blockamr.LinOpBCType.Dirichlet] * 3,
    )
    abec.set_level_bc(0, None)  # homogeneous
    abec.set_max_order(2)  # linear boundary interpolation = the reflect fill
    abec.set_scalars(0.0, 1.0)
    abec.set_a_coeffs(0, _const_cell(ba, dm, 0.0))
    bx = _const_face(geom, dm, 0, N, 1.0)
    by = _const_face(geom, dm, 1, N, 1.0)
    bz = _const_face(geom, dm, 2, N, 1.0)
    abec.set_b_coeffs(0, bx, by, bz)

    sol_ref = _zero_sol(ba, dm)
    mlmg = blockamr.MLMG(abec)
    mlmg.set_verbose(0)
    mlmg.set_max_iter(200)
    mlmg.solve(sol_ref, rhs, 1e-12, 0.0)

    sol_fc = _zero_sol(ba, dm)
    s = _make_solver_or_skip(
        "FaceCoeffSolver",
        coeffs,
        geom,
        executor,
        solver="cg",
        max_iter=5000,
        rtol=1e-12,
        bc=["dirichlet"] * 6,
    )
    stats = s.solve(rhs, sol_fc)
    assert stats["converged"] is True
    assert stats["num_iters"] > 1, "random rhs should take more than one CG iteration"

    max_diff = _max_abs_diff(sol_fc, sol_ref)
    assert max_diff < 1e-6, f"Max |sol_fc - sol_mlmg| = {max_diff} exceeds 1e-6"


def test_bc_validation_errors(blockamr_session):
    """bc entries must match the geometry's periodicity; CSR stays periodic-only."""
    if not hasattr(blockamr, "FaceCoeffSolver"):
        pytest.skip("blockamr.FaceCoeffSolver binding not available")

    N = 8
    geom_np, ba, dm = _make_mesh(N, [0, 0, 0])
    coeffs = _poisson_coeffs(geom_np, ba, dm, N, 1.0)
    alpha, fx, fy, fz = coeffs

    def build(cls, geom, bc):
        return getattr(blockamr, cls)(
            alpha, fx, fx, fy, fy, fz, fz, geom, executor="reference", bc=bc
        )

    try:
        # bc='periodic' (also the default) on a non-periodic geometry direction.
        with pytest.raises(RuntimeError, match="periodic"):
            build("FaceCoeffSolver", geom_np, ["periodic"] * 6)
        # Non-periodic bc entry on a periodic geometry direction.
        geom_p, _, _ = _make_mesh(N, [1, 1, 1])
        with pytest.raises(RuntimeError, match="periodic"):
            build("FaceCoeffSolver", geom_p, ["dirichlet"] * 6)
        # Unknown bc value.
        with pytest.raises(RuntimeError, match="unknown bc"):
            build("FaceCoeffSolver", geom_np, ["wall"] * 6)
        # Wrong length.
        with pytest.raises(RuntimeError, match="6 entries"):
            build("FaceCoeffSolver", geom_np, ["dirichlet"] * 4)
        # The assembled-CSR solver stays periodic-only.
        with pytest.raises(RuntimeError, match="periodic boundaries only"):
            build("FaceCoeffCsrSolver", geom_np, ["dirichlet"] * 6)
    except RuntimeError as exc:  # pragma: no cover - gating only
        if "without Ginkgo" in str(exc):
            pytest.skip("blockamr built without Ginkgo")
        raise


@pytest.mark.parametrize("executor", ["reference", "cuda"])
def test_all_neumann_singular_projected(blockamr_session, executor):
    """All-Neumann pure Poisson (alpha=0) is singular; project_nullspace composes.

    -lap u = f with u = cos(pi x)cos(pi y)cos(pi z) (du/dn = 0, mean-zero) and
    f = 3 pi^2 u (mean-zero, i.e. consistent): with project_nullspace=True the
    solve converges and returns the mean-zero representative, which is u.
    """
    pi = math.pi

    def u_fn(x, y, z):
        return np.cos(pi * x) * np.cos(pi * y) * np.cos(pi * z)

    def f_fn(x, y, z):
        return 3 * pi**2 * u_fn(x, y, z)

    N = 16
    geom, ba, dm = _make_mesh(N, [0, 0, 0])
    dx = geom.cell_size()
    coeffs = _poisson_coeffs(geom, ba, dm, N, 0.0)
    rhs = blockamr.MultiFab(ba, dm, 1, 0)
    _fill_cell(rhs, dx, f_fn)
    sol = _zero_sol(ba, dm)

    s = _make_solver_or_skip(
        "FaceCoeffSolver",
        coeffs,
        geom,
        executor,
        solver="cg",
        max_iter=5000,
        rtol=1e-11,
        bc=["neumann"] * 6,
        project_nullspace=True,
    )
    stats = s.solve(rhs, sol)
    assert stats["converged"] is True
    assert abs(_mean(sol)) < 1e-10, f"solution mean {_mean(sol)} not ~0"
    err = _max_err_vs_analytic(sol, dx, u_fn)
    assert err < 2e-2, f"error vs analytic {err} too large"
