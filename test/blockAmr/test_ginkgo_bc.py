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
* validation errors (bc vs geometry periodicity, bc that CSR cannot express);
* the singular all-Neumann pure Poisson composes with project_nullspace.

The ``bc_data`` constructor kwarg makes those same BCs INHOMOGENEOUS: a ghosted
cell MultiFab whose ghost layer carries u on the face (dirichlet sides) or the
outward du/dn (neumann sides) — MLMG's ``set_level_bc`` contract, so one fab
drives both solvers. Section 2 below covers it.

Section 3 covers the assembled-CSR twin, ``FaceCoeffCsrSolver``, which folds the
same HOMOGENEOUS conditions into the matrix entries instead of into a ghost
layer, and must therefore land on the matrix-free path's answer.
"""

import math

import numpy as np
import pytest

import blockamr

from ._executors import gko_executor


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
            alpha, fx, fx, fy, fy, fz, fz, geom, executor=gko_executor(executor), **kwargs
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


def _solve_manufactured(n, executor, bc, alpha_val, u_fn, f_fn, cls="FaceCoeffSolver"):
    """Solve alpha*u - lap u = f on the non-periodic unit cube; max-norm error vs u."""
    geom, ba, dm = _make_mesh(n, [0, 0, 0])
    dx = geom.cell_size()
    coeffs = _poisson_coeffs(geom, ba, dm, n, alpha_val)
    rhs = blockamr.MultiFab(ba, dm, 1, 0)
    _fill_cell(rhs, dx, f_fn)
    sol = _zero_sol(ba, dm)
    s = _make_solver_or_skip(
        cls,
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
    """bc entries must match the geometry's periodicity, on BOTH solver paths.

    The assembled-CSR solver used to reject every non-periodic bc outright; it
    now folds homogeneous dirichlet/neumann into the matrix (section 3), so the
    only thing left to assert about it here is that it accepts them and still
    applies the SAME bc-vs-geometry validation as the matrix-free path — that
    check lives in parseBc, which both paths call.
    """
    if not hasattr(blockamr, "FaceCoeffSolver"):
        pytest.skip("blockamr.FaceCoeffSolver binding not available")

    N = 8
    geom_np, ba, dm = _make_mesh(N, [0, 0, 0])
    coeffs = _poisson_coeffs(geom_np, ba, dm, N, 1.0)
    alpha, fx, fy, fz = coeffs

    def build(cls, geom, bc):
        return getattr(blockamr, cls)(
            alpha, fx, fx, fy, fy, fz, fz, geom, executor=gko_executor("reference"), bc=bc
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
        # The assembled-CSR solver now CONSTRUCTS on a non-periodic bc: the
        # boundary faces are folded onto the diagonal instead of wrapping around.
        assert build("FaceCoeffCsrSolver", geom_np, ["dirichlet"] * 6) is not None
        assert build("FaceCoeffCsrSolver", geom_np, ["neumann"] * 6) is not None
        # ... and it validates bc against the geometry exactly as FaceCoeffSolver
        # does, in both directions.
        with pytest.raises(RuntimeError, match="periodic"):
            build("FaceCoeffCsrSolver", geom_np, ["periodic"] * 6)
        with pytest.raises(RuntimeError, match="periodic"):
            build("FaceCoeffCsrSolver", geom_p, ["dirichlet"] * 6)
        with pytest.raises(RuntimeError, match="unknown bc"):
            build("FaceCoeffCsrSolver", geom_np, ["wall"] * 6)
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


# ---------------------------------------------------------------------------
# 2. Inhomogeneous BCs (bc_data)
#
# The homogeneous fills above are ghost = sign*interior; bc_data adds the affine
# term, ghost = sign*interior + scale*g, with scale = 2 for dirichlet (so
# (interior+ghost)/2 = g at the face) and scale = dx for neumann (so
# (ghost-interior)/dx = g across it). Same face placement, same order — only the
# constant moves off zero.
#
# That constant makes the boundary operator AFFINE, L(x) = A x + c0, and the two
# solver paths deal with it differently, which is why every test below runs on
# both. The Krylov path keeps `apply` linear (Ginkgo requires it) and folds
# c0 = L(0) into the right-hand side, one extra apply per solve. solver="gmg"
# instead lets its OUTER residual be rhs - L(x) directly; the V-cycle underneath
# still solves for a correction, whose BC is homogeneous whatever the solution's
# is. test_inhomogeneous_bc_paths_agree pins the two against each other.
# ---------------------------------------------------------------------------

# One manufactured solution for the whole section, chosen so that NOTHING about
# it is zero on the boundary: a homogeneous fill cannot accidentally reproduce
# it, which is what test_bc_data_is_not_ignored turns into an assertion.
_PHASE = (0.4, 0.7, 1.1)


def _u_inhom(x, y, z):
    pi = math.pi
    return np.sin(pi * x + _PHASE[0]) * np.sin(pi * y + _PHASE[1]) * np.sin(pi * z + _PHASE[2])


def _f_inhom(x, y, z):
    """rhs of u - lap u = f (alpha=1 Helmholtz, nonsingular under any BC mix)."""
    return (1.0 + 3.0 * math.pi**2) * _u_inhom(x, y, z)


def _grad_u_inhom(d, x, y, z):
    """d-th partial of _u_inhom."""
    pi = math.pi
    fac = [np.sin, np.sin, np.sin]
    fac[d] = np.cos
    return pi * fac[0](pi * x + _PHASE[0]) * fac[1](pi * y + _PHASE[1]) * fac[2](pi * z + _PHASE[2])


def _bc_datum(side, x, y, z, bc):
    """The datum bc_data must carry on `side` for the exact solution _u_inhom.

    dirichlet -> u on the face; neumann -> du/dn with n the OUTWARD normal, so
    the low sides carry -d(u)/dx_d and the high sides +d(u)/dx_d.
    """
    if bc[side] == "dirichlet":
        return _u_inhom(x, y, z)
    d = side // 2
    sign = -1.0 if side % 2 == 0 else 1.0
    return sign * _grad_u_inhom(d, x, y, z)


def _bc_data(ba, dm, geom, bc):
    """MLMG-style carrier: cell MultiFab, 1 ghost, datum in the GHOST layer.

    Each ghost cell outside a non-periodic domain side holds the datum for the
    boundary face it looks across — evaluated AT that face, not at the ghost
    cell centre, since that is where both fills place it. Everything else (valid
    region, interior/periodic ghosts) stays zero and is never read.
    """
    dx = geom.cell_size()
    dom = geom.domain()
    dlo, dhi = dom.small_end(), dom.big_end()
    mf = blockamr.MultiFab(ba, dm, 1, 1)
    mf.set_val(0.0)
    for mfi in blockamr.MFIterator(mf):
        arr = mf.copy_grown_to_host(mfi)
        vb = mfi.valid_box()
        glo = [vb.small_end()[d] - 1 for d in range(3)]
        grids = list(_cell_centers(glo, arr.shape, dx))
        for side in range(6):
            if bc[side] == "periodic":
                continue
            d = side // 2
            low = side % 2 == 0
            ghost = (dlo[d] - 1) if low else (dhi[d] + 1)
            local = ghost - glo[d]
            if not 0 <= local < arr.shape[d]:
                continue  # this box does not touch that domain face
            sl = [slice(None)] * 3
            sl[d] = local
            sl = tuple(sl)
            coords = [g[sl] for g in grids]
            # Move the normal coordinate off the ghost centre and onto the face.
            face = (dlo[d] if low else dhi[d] + 1) * dx[d]
            coords[d] = np.full_like(coords[d], face)
            arr[sl + (0,)] = _bc_datum(side, *coords, bc)
        mf.copy_grown_from(mfi, arr)
    return mf


def _multibox_mesh(n, max_size):
    """Non-periodic mesh chopped into boxes of `max_size`, plus matching coeffs.

    convert_ba keeps the box ORDER, so the cell DistributionMapping still applies
    to the face fabs — which is what lets this file use more than one box
    (surrounding_nodes on the whole domain would not line up).
    """
    box = blockamr.Box([0, 0, 0], [n - 1, n - 1, n - 1])
    rb = blockamr.RealBox([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    geom = blockamr.Geometry(box, rb, 0, [0, 0, 0])
    ba = blockamr.BoxArray(box)
    ba.max_size(max_size)
    dm = blockamr.DistributionMapping(ba)
    inv_dx2 = 1.0 / geom.cell_size()[0] ** 2
    faces = []
    for d in range(3):
        typ = [0, 0, 0]
        typ[d] = 1
        mf = blockamr.MultiFab(blockamr.convert_ba(ba, blockamr.IntVect(*typ)), dm, 1, 0)
        mf.set_val(-inv_dx2)
        faces.append(mf)
    return geom, ba, dm, (_const_cell(ba, dm, 1.0), *faces)


def _solve_inhom(n, executor, bc, max_size=None, **solver_kw):
    """Solve u - lap u = f with the inhomogeneous BCs of `bc`; return (err, sol)."""
    if max_size is None:
        geom, ba, dm = _make_mesh(n, [0, 0, 0])
        coeffs = _poisson_coeffs(geom, ba, dm, n, 1.0)
    else:
        geom, ba, dm, coeffs = _multibox_mesh(n, max_size)
    dx = geom.cell_size()
    rhs = blockamr.MultiFab(ba, dm, 1, 0)
    _fill_cell(rhs, dx, _f_inhom)
    sol = _zero_sol(ba, dm)
    s = _make_solver_or_skip(
        "FaceCoeffSolver",
        coeffs,
        geom,
        executor,
        bc=bc,
        bc_data=_bc_data(ba, dm, geom, bc),
        **solver_kw,
    )
    stats = s.solve(rhs, sol)
    assert stats["converged"] is True, f"did not converge: {dict(stats)}"
    return _max_err_vs_analytic(sol, dx, _u_inhom), sol


_CG = dict(solver="cg", max_iter=5000, rtol=1e-11)


@pytest.mark.parametrize("executor", ["reference", "cuda"])
@pytest.mark.parametrize(
    "kind, bc",
    [
        ("dirichlet", ["dirichlet"] * 6),
        ("neumann", ["neumann"] * 6),
        ("mixed", ["dirichlet", "neumann", "neumann", "dirichlet", "dirichlet", "neumann"]),
    ],
)
def test_inhomogeneous_manufactured_second_order(blockamr_session, executor, kind, bc):
    """u - lap u = f with u nonzero AND du/dn nonzero on every face: 2nd order.

    The point of running all three mixes off ONE exact solution: the dirichlet
    and neumann branches of the fill are independent (different sign, different
    scale), and the mixed row is the only one that catches a per-side indexing
    slip — a fill that used the wrong side's datum still passes both uniform
    rows.
    """
    err_16, _ = _solve_inhom(16, executor, bc, **_CG)
    err_32, _ = _solve_inhom(32, executor, bc, **_CG)

    assert err_16 < 2e-2, f"{kind}: N=16 error {err_16} too large"
    assert err_32 < 6e-3, f"{kind}: N=32 error {err_32} too large"
    ratio = err_16 / err_32
    assert ratio > 3, f"{kind}: convergence ratio {ratio} not ~2nd order (expected ~4)"


@pytest.mark.parametrize("executor", ["reference", "cuda"])
def test_inhomogeneous_bc_survives_decomposition(blockamr_session, executor):
    """8 boxes must give the single box's answer — the ghost fill is PER BOX.

    Every fill in bc_geom.hpp walks boxes and asks whether each one touches a
    given domain face; an interior box touches none, and its ghost layer is
    FillBoundary's business instead. A fill that wrote the boundary datum into
    every box's ghost layer, or skipped a box that does touch, passes the
    single-box tests above and fails here.
    """
    bc = ["dirichlet", "neumann", "dirichlet", "neumann", "dirichlet", "neumann"]
    err_1, _ = _solve_inhom(16, executor, bc, **_CG)
    err_8, _ = _solve_inhom(16, executor, bc, max_size=8, **_CG)

    assert abs(err_8 - err_1) < 1e-7, f"decomposition changed the answer: {err_8} vs {err_1}"
    assert err_1 < 2e-2, f"error vs analytic {err_1} too large"


@pytest.mark.parametrize("executor", ["reference", "cuda"])
def test_bc_data_is_not_ignored(blockamr_session, executor):
    """Dropping bc_data must change the answer — the anti-vacuity check.

    Every assertion in this section is a tolerance on the solution, and a
    silently-ignored bc_data would still produce a converged solve of the
    HOMOGENEOUS problem. So solve the same system both ways and require the
    homogeneous one to be badly wrong: the exact solution is ~0.3-1 in
    magnitude on the boundary, so anything under O(0.1) here would mean the
    inhomogeneous term is not reaching the operator.
    """
    bc = ["dirichlet"] * 6
    err_inhom, _ = _solve_inhom(16, executor, bc, **_CG)

    geom, ba, dm = _make_mesh(16, [0, 0, 0])
    dx = geom.cell_size()
    coeffs = _poisson_coeffs(geom, ba, dm, 16, 1.0)
    rhs = blockamr.MultiFab(ba, dm, 1, 0)
    _fill_cell(rhs, dx, _f_inhom)
    sol = _zero_sol(ba, dm)
    s = _make_solver_or_skip("FaceCoeffSolver", coeffs, geom, executor, bc=bc, **_CG)
    assert s.solve(rhs, sol)["converged"] is True
    err_home = _max_err_vs_analytic(sol, dx, _u_inhom)

    assert err_home > 0.1, f"homogeneous BCs gave error {err_home} — bc_data may be a no-op"
    assert err_inhom < 0.1 * err_home, f"bc_data barely helped: {err_inhom} vs {err_home}"


@pytest.mark.parametrize("executor", ["reference", "cuda"])
def test_inhomogeneous_bc_paths_agree(blockamr_session, executor):
    """The Krylov rhs-fold and the stationary V-cycle solve the SAME system.

    Three configurations that handle the affine term by different routes —
    unpreconditioned CG (fold, no multigrid), CG preconditioned by the V-cycle
    (fold, and a hierarchy built on homogeneous fills), and the native
    stationary GMG solver (no fold at all, the offset lives in the outer
    residual) — must land on one solution. Cross-checking them is what makes the
    "a correction has homogeneous BCs" argument an assertion rather than a claim.
    """
    bc = ["dirichlet", "dirichlet", "neumann", "neumann", "dirichlet", "neumann"]
    err_cg, sol_cg = _solve_inhom(16, executor, bc, **_CG)
    _, sol_pc = _solve_inhom(16, executor, bc, precond="gmg", **_CG)
    _, sol_gmg = _solve_inhom(
        16,
        executor,
        bc,
        solver="gmg",
        max_iter=200,
        rtol=1e-11,
        gmg_coarsest_sweeps=100,
    )

    assert _max_abs_diff(sol_cg, sol_pc) < 1e-8, "precond='gmg' disagrees with plain CG"
    assert _max_abs_diff(sol_cg, sol_gmg) < 1e-8, "solver='gmg' disagrees with CG"
    assert err_cg < 2e-2, f"error vs analytic {err_cg} too large"


@pytest.mark.parametrize("executor", ["reference", "cuda"])
def test_inhomogeneous_bc_composes_with_kokkos_precond(blockamr_session, executor):
    """precond='gmg_kokkos' composes with bc_data, for the reason precond='gmg' does.

    The ported Kokkos V-cycle carries only the HOMOGENEOUS reflection and has no
    bc_data of its own. That is not a gap: as a preconditioner it is handed a
    residual and returns a correction, and a correction's boundary condition is
    homogeneous. Asserted rather than argued, because the failure it rules out —
    a preconditioner quietly solving a different boundary problem — surfaces only
    as a worse iteration count, never as a wrong answer.
    """
    bc = ["dirichlet", "dirichlet", "neumann", "neumann", "dirichlet", "neumann"]
    _, sol_cg = _solve_inhom(16, executor, bc, **_CG)
    try:
        _, sol_k = _solve_inhom(16, executor, bc, precond="gmg_kokkos", **_CG)
    except RuntimeError as exc:  # device-only path
        pytest.skip(f"precond='gmg_kokkos' unavailable on {executor}: {exc}")
    assert _max_abs_diff(sol_cg, sol_k) < 1e-8, "gmg_kokkos-preconditioned solve disagrees"


@pytest.mark.parametrize("executor", ["reference", "cuda"])
def test_inhomogeneous_dirichlet_matches_mlmg(blockamr_session, executor):
    """The same bc_data fab drives MLMG's set_level_bc and ours to one answer.

    MLABecLaplacian (a=b=1 -> u - lap u) with Dirichlet domain BCs, the boundary
    values handed to set_level_bc in the ghost layer, and set_max_order(2) so
    MLMG's boundary interpolation is linear = ghost = 2g - interior. That is
    exactly the inhomogeneous fill, so the two solve the SAME matrix with the
    SAME right-hand side and must agree to solver tolerance — the external
    referee for the convention (value at the FACE, not at the ghost centre).
    """
    N = 32
    bc = ["dirichlet"] * 6
    geom, ba, dm = _make_mesh(N, [0, 0, 0])
    dx = geom.cell_size()
    coeffs = _poisson_coeffs(geom, ba, dm, N, 1.0)
    rhs = blockamr.MultiFab(ba, dm, 1, 0)
    _fill_cell(rhs, dx, _f_inhom)
    bc_data = _bc_data(ba, dm, geom, bc)

    abec = blockamr.MLABecLaplacian(geom, ba, dm)
    abec.set_domain_bc(
        [blockamr.LinOpBCType.Dirichlet] * 3,
        [blockamr.LinOpBCType.Dirichlet] * 3,
    )
    abec.set_level_bc(0, bc_data)
    abec.set_max_order(2)
    abec.set_scalars(1.0, 1.0)
    abec.set_a_coeffs(0, _const_cell(ba, dm, 1.0))
    abec.set_b_coeffs(
        0,
        _const_face(geom, dm, 0, N, 1.0),
        _const_face(geom, dm, 1, N, 1.0),
        _const_face(geom, dm, 2, N, 1.0),
    )
    sol_ref = _zero_sol(ba, dm)
    mlmg = blockamr.MLMG(abec)
    mlmg.set_verbose(0)
    mlmg.set_max_iter(200)
    mlmg.solve(sol_ref, rhs, 1e-12, 0.0)

    sol_fc = _zero_sol(ba, dm)
    s = _make_solver_or_skip(
        "FaceCoeffSolver", coeffs, geom, executor, bc=bc, bc_data=bc_data, **_CG
    )
    assert s.solve(rhs, sol_fc)["converged"] is True

    max_diff = _max_abs_diff(sol_fc, sol_ref)
    assert max_diff < 1e-6, f"Max |sol_fc - sol_mlmg| = {max_diff} exceeds 1e-6"


def test_bc_data_validation_errors(blockamr_session):
    """bc_data is refused, not ignored, when nothing would read it correctly."""
    if not hasattr(blockamr, "FaceCoeffSolver"):
        pytest.skip("blockamr.FaceCoeffSolver binding not available")

    N = 8
    geom, ba, dm = _make_mesh(N, [0, 0, 0])
    alpha, fx, fy, fz = _poisson_coeffs(geom, ba, dm, N, 1.0)
    bc = ["dirichlet"] * 6

    def build(geom_, bc_, data, cls="FaceCoeffSolver"):
        return getattr(blockamr, cls)(
            alpha,
            fx,
            fx,
            fy,
            fy,
            fz,
            fz,
            geom_,
            executor=gko_executor("reference"),
            bc=bc_,
            bc_data=data,
        )

    try:
        # No ghost layer to carry the datum.
        no_ghost = blockamr.MultiFab(ba, dm, 1, 0)
        no_ghost.set_val(0.0)
        with pytest.raises(RuntimeError, match="ghost cell"):
            build(geom, bc, no_ghost)
        # Different BoxArray than the coefficients.
        other_ba = blockamr.BoxArray(blockamr.Box([0, 0, 0], [N - 1, N - 1, N - 1]))
        other_ba.max_size(N // 2)
        other = blockamr.MultiFab(other_ba, blockamr.DistributionMapping(other_ba), 1, 1)
        other.set_val(0.0)
        with pytest.raises(RuntimeError, match="BoxArray"):
            build(geom, bc, other)
        # All-periodic: no side would ever read it.
        geom_p, ba_p, dm_p = _make_mesh(N, [1, 1, 1])
        data_p = blockamr.MultiFab(ba_p, dm_p, 1, 1)
        data_p.set_val(0.0)
        with pytest.raises(RuntimeError, match="nothing would read it"):
            getattr(blockamr, "FaceCoeffSolver")(
                alpha,
                fx,
                fx,
                fy,
                fy,
                fz,
                fz,
                geom_p,
                executor=gko_executor("reference"),
                bc=["periodic"] * 6,
                bc_data=data_p,
            )
        # The assembled-CSR solver has no inhomogeneous path.
        with pytest.raises(RuntimeError, match="periodic boundaries only"):
            build(geom, bc, _bc_data(ba, dm, geom, bc), cls="FaceCoeffCsrSolver")
    except RuntimeError as exc:  # pragma: no cover - gating only
        if "without Ginkgo" in str(exc):
            pytest.skip("blockamr built without Ginkgo")
        raise


# ---------------------------------------------------------------------------
# 3. The assembled-CSR path (FaceCoeffCsrSolver) under the same homogeneous BCs
#
# Everything above folds BCs by REFLECTING a ghost cell and leaves the matrix
# alone. FaceCoeffCsrSolver has no ghost layer to reflect — it hands Ginkgo an
# explicit CSR — so the identical fold has to be spelled into the entries:
#
#   periodic  side: keep the modular-wraparound neighbour column;
#   dirichlet side: DROP that column, and diag += (-1) * aFace;
#   neumann   side: DROP that column, and diag += (+1) * aFace,
#
# because the reflection makes the outside neighbour's value sign*pC, which turns
# the stencil term aFace*pNeighbour into the diagonal term sign*aFace*pC. Rows on
# a non-periodic boundary therefore carry fewer than 7 entries.
#
# That is a re-derivation of the same operator by a different route, so the check
# that matters is not a tolerance on a solution but AGREEMENT: same mesh, same
# coefficients, same bc, both solvers, one answer. A sign slip, a side-indexing
# slip, or a dropped fold all show up there and only there — the manufactured
# rows below would still converge at 2nd order with the wrong sign on one face.
#
# Homogeneous only: bc_data is an rhs fold (the affine c0 of section 2) and the
# assembled path has none, so it is refused.
# ---------------------------------------------------------------------------

# Fully non-periodic for the uniform mixes; the "mixed" row is periodic in z so
# that ONE bc array exercises all three per-side branches of the fold at once.
_CSR_BC_CASES = [
    ("dirichlet", [0, 0, 0], ["dirichlet"] * 6),
    ("neumann", [0, 0, 0], ["neumann"] * 6),
    (
        "mixed",
        [0, 0, 1],
        ["dirichlet", "dirichlet", "neumann", "neumann", "periodic", "periodic"],
    ),
]


@pytest.mark.parametrize("executor", ["reference", "cuda"])
@pytest.mark.parametrize("kind, periodic, bc", _CSR_BC_CASES)
def test_csr_matches_matrix_free(blockamr_session, executor, kind, periodic, bc):
    """The assembled matrix IS the matrix-free operator, on every bc kind.

    alpha=1 (Helmholtz) so the system is nonsingular under every mix here,
    all-Neumann included, and a seeded random rhs so every row is exercised —
    a boundary row assembled wrongly cannot hide in one smooth eigenmode. Both
    solves run the same unpreconditioned CG to rtol=1e-12, so the 1e-8 bar is
    solver tolerance, not discretisation error: the two answers differ only by
    the two mat-vec implementations' roundoff.
    """
    N = 16
    geom, ba, dm = _make_mesh(N, periodic)
    coeffs = _poisson_coeffs(geom, ba, dm, N, 1.0)
    rhs = _random_rhs(ba, dm)

    sols = []
    for cls in ("FaceCoeffSolver", "FaceCoeffCsrSolver"):
        sol = _zero_sol(ba, dm)
        s = _make_solver_or_skip(
            cls, coeffs, geom, executor, solver="cg", max_iter=5000, rtol=1e-12, bc=bc
        )
        stats = s.solve(rhs, sol)
        assert stats["converged"] is True, f"{kind}/{cls} did not converge: {dict(stats)}"
        assert stats["num_iters"] > 1, f"{kind}/{cls}: random rhs took one CG iteration"
        sols.append(sol)

    max_diff = _max_abs_diff(sols[0], sols[1])
    assert max_diff < 1e-8, f"{kind}: |matrix-free - CSR| = {max_diff} exceeds 1e-8"


# The three manufactured solutions of section 1, restated for the CSR path. All
# three are eigenfunctions of -lap with eigenvalue 3 pi^2, so one rhs formula
# covers them; alpha=1 keeps every mix nonsingular.
def _u_csr_dirichlet(x, y, z):
    pi = math.pi
    return np.sin(pi * x) * np.sin(pi * y) * np.sin(pi * z)


def _u_csr_neumann(x, y, z):
    pi = math.pi
    return np.cos(pi * x) * np.cos(pi * y) * np.cos(pi * z)


def _u_csr_mixed(x, y, z):
    pi = math.pi
    return np.sin(pi * x) * np.cos(pi * y) * np.cos(pi * z)


@pytest.mark.parametrize("executor", ["reference", "cuda"])
@pytest.mark.parametrize(
    "kind, bc, u_fn",
    [
        ("dirichlet", ["dirichlet"] * 6, _u_csr_dirichlet),
        ("neumann", ["neumann"] * 6, _u_csr_neumann),
        (
            "mixed",
            ["dirichlet", "dirichlet", "neumann", "neumann", "neumann", "neumann"],
            _u_csr_mixed,
        ),
    ],
)
def test_csr_manufactured_second_order(blockamr_session, executor, kind, bc, u_fn):
    """u - lap u = f through the assembled matrix: 2nd order, same as matrix-free.

    Agreement (above) proves the two paths build one matrix; this proves that
    matrix is the RIGHT one — an identical sign error in both would satisfy
    agreement and fail here. Same u, same thresholds as the section-1 rows, so
    the two are directly comparable.
    """

    def f_fn(x, y, z):
        return (1.0 + 3.0 * math.pi**2) * u_fn(x, y, z)

    err_16 = _solve_manufactured(16, executor, bc, 1.0, u_fn, f_fn, cls="FaceCoeffCsrSolver")
    err_32 = _solve_manufactured(32, executor, bc, 1.0, u_fn, f_fn, cls="FaceCoeffCsrSolver")

    assert err_16 < 5e-3, f"{kind}: N=16 error {err_16} too large"
    assert err_32 < 1.5e-3, f"{kind}: N=32 error {err_32} too large"
    ratio = err_16 / err_32
    assert ratio > 3, f"{kind}: convergence ratio {ratio} not ~2nd order (expected ~4)"


def test_csr_refuses_bc_data(blockamr_session):
    """bc_data on the CSR path is refused, and that refusal is now REACHABLE.

    It used to sit behind the periodic-only rejection and could never fire: a
    bc_data carrier needs a non-periodic side, and a non-periodic bc was already
    out. Folding the homogeneous boundaries into the matrix turned it into a live
    user-facing path, so it gets its own case rather than riding on a dead
    branch. It stays a refusal because an inhomogeneous BC is the affine term
    L(x) = A x + c0 of section 2 — an rhs fold, which the assembled path does not
    have — and silently dropping the datum would read as a wrong answer.
    """
    if not hasattr(blockamr, "FaceCoeffCsrSolver"):
        pytest.skip("blockamr.FaceCoeffCsrSolver binding not available")

    N = 8
    bc = ["dirichlet"] * 6
    geom, ba, dm = _make_mesh(N, [0, 0, 0])
    alpha, fx, fy, fz = _poisson_coeffs(geom, ba, dm, N, 1.0)

    try:
        with pytest.raises(RuntimeError, match="bc_data"):
            blockamr.FaceCoeffCsrSolver(
                alpha,
                fx,
                fx,
                fy,
                fy,
                fz,
                fz,
                geom,
                executor=gko_executor("reference"),
                bc=bc,
                bc_data=_bc_data(ba, dm, geom, bc),
            )
    except RuntimeError as exc:  # pragma: no cover - gating only
        if "without Ginkgo" in str(exc):
            pytest.skip("blockamr built without Ginkgo")
        raise
