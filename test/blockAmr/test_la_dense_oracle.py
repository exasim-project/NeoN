# SPDX-FileCopyrightText: 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""The ``blockamr::la`` matrix against an INDEPENDENT reference: a dense numpy operator.

Every other agreement test in this suite compares two C++ paths, and all of them
reach the same ``stencil.hpp``: ``FaceCoeffOp`` and the GMG smoothers all call
``stencilDiag`` / ``loadFaceCoeffs``.
That sharing is deliberate -- the twelve sites must agree bit-for-bit -- but it
means none of those tests can see a wrong diagonal FORMULA: both sides would be
wrong identically, by construction. What they actually check is indexing, row
ordering and the BC column drop.

So this file writes the 7-point matrix out longhand in numpy, from the
coefficient fields alone, and solves it densely. It reuses nothing from the C++
side, which is exactly what makes it able to object to::

    diag = alpha - (aE + aW + aN + aS + aT + aB)

being anything else. Both halves of the BC contract are in the oracle too: a
periodic side keeps its modular wraparound column, a non-periodic side has no
such column at all and carries ``sign*aF`` on the diagonal instead (``-1``
Dirichlet, ``+1`` Neumann) -- the reflect ghost, folded. A mistake in either half
moves the dense solution and the comparison below fails.

n = 8 (512 rows) rather than the 16 the rest of the suite uses: the reference is
a dense matrix and a plain triple loop, and 512 rows keeps both instantaneous
while still giving every cell class -- interior, face, edge, corner -- many
representatives.

The entry point is ``blockamr._blockamr._la_system_solve``, the same test-facing
binding ``test_la_linear_system.py`` and ``test_la_boundary_conditions.py`` use.
"""

import numpy as np
import pytest

import blockamr

from ._executors import gko_executor

_ext = getattr(blockamr, "_blockamr", None)

N = 8
_SOLVE_KWARGS = dict(solver="cg", max_iter=5000, rtol=1e-14, atol=0.0)

# The four BC kinds test_la_boundary_conditions.py parametrises over: the periodic
# control that folds nothing, both signs of the fold, and one array that mixes all
# three so no row is only ever seen next to its own kind.
_BC_CASES = [
    ("periodic", [1, 1, 1], ["periodic"] * 6),
    ("dirichlet", [0, 0, 0], ["dirichlet"] * 6),
    ("neumann", [0, 0, 0], ["neumann"] * 6),
    (
        "mixed",
        [0, 0, 1],
        ["dirichlet", "dirichlet", "neumann", "neumann", "periodic", "periodic"],
    ),
]

# Two solutions of one matrix: a CG run stopped at rtol=1e-14 and a dense LU. The
# solutions peak between 1.1e-2 and 5.0e-2 across the four cases, and the worst
# measured disagreement is 1.6e-16 -- three orders of magnitude of headroom. Small
# enough to still catch the mistakes this file exists for: a wrong diagonal
# formula, a wrong fold sign or a kept wraparound column each move the answer by
# 1.2e-2 to 1.1e-1, i.e. by more than 10**13 times this bar. Absolute rather than
# relative because what has to be small is the gap between two roundings of one
# solve, not a ratio at cells where the solution passes through zero.
_ORACLE_TOL = 1e-13


def _require_bindings():
    if _ext is None or not hasattr(_ext, "_la_system_solve"):
        pytest.skip("blockamr._la_system_solve binding not available")


def _make_mesh(periodic):
    """Single-box mesh on [0,1]^3, N cells per side, with the given periodicity."""
    box = blockamr.Box([0, 0, 0], [N - 1, N - 1, N - 1])
    rb = blockamr.RealBox([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    geom = blockamr.Geometry(box, rb, 0, periodic)
    ba = blockamr.BoxArray(box)
    ba.max_size(N)  # single box -> face fabs align 1:1 with the cell fab
    dm = blockamr.DistributionMapping(ba)
    return geom, ba, dm


def _const_cell(ba, dm, value):
    mf = blockamr.MultiFab(ba, dm, 1, 0)
    mf.set_val(value)
    return mf


def _face_field(geom, dm, d, value):
    dom = geom.domain()
    face_box = blockamr.Box(dom.small_end(), dom.big_end())
    face_box.surrounding_nodes(d)
    face_ba = blockamr.BoxArray(face_box)
    face_ba.max_size(N)
    mf = blockamr.MultiFab(face_ba, dm, 1, 0)
    mf.set_val(value)
    return mf


def _out_fields(geom, ba, dm):
    """Zeroed receivers for the assembled (alpha, ux, uy, uz)."""
    return _const_cell(ba, dm, 0.0), *(_face_field(geom, dm, d, 0.0) for d in range(3))


def _one_box(mf):
    """The single box of a single-box MultiFab, as (i, j, k) without the component."""
    boxes = [mf.copy_to_host(mfi) for mfi in blockamr.MFIterator(mf)]
    assert len(boxes) == 1
    return boxes[0][:, :, :, 0]


def _random_rhs(ba, dm, seed=42):
    """Seeded random rhs -- the full spectrum, so no direction of the matrix is unprobed.

    A smooth rhs is nearly an eigenvector of this operator, and a comparison
    against it would barely depend on the off-diagonals at all.
    """
    mf = blockamr.MultiFab(ba, dm, 1, 0)
    rng = np.random.default_rng(seed)
    for mfi in blockamr.MFIterator(mf):
        arr = mf.copy_to_host(mfi)
        arr[:, :, :, 0] = rng.standard_normal(arr.shape[:3])
        mf.copy_from(mfi, arr)
    return mf


def _zero_sol(ba, dm):
    sol = blockamr.MultiFab(ba, dm, 1, 1)
    sol.set_val(0.0)  # MultiFabs are not zero-initialized (arena recycling)
    return sol


def _dense_operator(alpha, upper, lower, bc, shape):
    """The explicit 7-point matrix, in numpy, from the coefficient fields alone.

    Written INDEPENDENTLY of ``linearAlgebra/stencil.hpp``: this is the reference
    the C++ paths are checked against, so the diagonal formula and the boundary
    fold are spelled out here rather than borrowed from the code under test. Keep
    it that way -- the moment it imports an answer from the C++ side it stops
    being able to disagree with it.

    ``alpha`` is the cell-centred diagonal source; ``upper[d]`` / ``lower[d]`` are
    the face fields of direction ``d``, the high face of cell ``i`` living at face
    index ``i + 1`` of ``upper[d]`` and its low face at index ``i`` of ``lower[d]``.
    ``bc`` is the six-string side array in order (xlo, xhi, ylo, yhi, zlo, zhi).

    Row and column index is ``(k*nj + j)*ni + i``. The choice is arbitrary -- the
    test flattens its vectors the same way -- but it is the flattening the
    assembled C++ route uses, so a row of this matrix can be read against one of
    that route's directly.

    A plain triple loop, at the speed of the mathematical statement rather than of
    numpy: at 8**3 = 512 rows this is instantaneous, and clarity is what a
    reference is for.
    """
    ni, nj, nk = shape
    n = ni * nj * nk

    def idx(i, j, k):
        return (k * nj + j) * ni + i

    a = np.zeros((n, n))
    for k in range(nk):
        for j in range(nj):
            for i in range(ni):
                aE = upper[0][i + 1, j, k]
                aW = lower[0][i, j, k]
                aN = upper[1][i, j + 1, k]
                aS = lower[1][i, j, k]
                aT = upper[2][i, j, k + 1]
                aB = lower[2][i, j, k]

                diag = alpha[i, j, k] - (aE + aW + aN + aS + aT + aB)

                # Order (xlo, xhi, ylo, yhi, zlo, zhi), matching `bc`.
                sides = [
                    (i == 0, aW, ((i - 1) % ni, j, k)),
                    (i == ni - 1, aE, ((i + 1) % ni, j, k)),
                    (j == 0, aS, (i, (j - 1) % nj, k)),
                    (j == nj - 1, aN, (i, (j + 1) % nj, k)),
                    (k == 0, aB, (i, j, (k - 1) % nk)),
                    (k == nk - 1, aT, (i, j, (k + 1) % nk)),
                ]
                for s, (on_domain_face, a_face, neighbour) in enumerate(sides):
                    if on_domain_face and bc[s] != "periodic":
                        # No such neighbour: the reflect ghost is sign * u(C), so the
                        # term is sign*aF ON THE DIAGONAL and the column is not there
                        # at all -- not there carrying an explicit zero.
                        diag += (-1.0 if bc[s] == "dirichlet" else 1.0) * a_face
                    else:
                        a[idx(i, j, k), idx(*neighbour)] += a_face

                # Last, so every side's fold is already in `diag`. At ni,nj,nk >= 3 no
                # wraparound neighbour is the cell itself, so this never clobbers one.
                a[idx(i, j, k), idx(i, j, k)] = diag
    return a


def _flat(arr):
    """(i, j, k) array as the vector this file's matrix is indexed by: i fastest."""
    return arr.ravel(order="F")


def _system_solve(gamma, alpha, geom, rhs, sol, bc, out):
    try:
        return _ext._la_system_solve(
            gamma,
            alpha,
            geom,
            rhs,
            sol,
            *out,
            executor=gko_executor("reference"),
            bc=bc,
            **_SOLVE_KWARGS,
        )
    except RuntimeError as exc:
        if "without Ginkgo" in str(exc):
            pytest.skip("blockamr built without Ginkgo")
        raise


@pytest.mark.parametrize("case, periodic, bc", _BC_CASES)
def test_matrix_free_solve_matches_the_dense_operator(blockamr_session, case, periodic, bc):
    """The matrix-free route solves the matrix the coefficients say it should.

    ``system += ops::Laplacian(...)`` writes the coefficients; the oracle turns
    those same coefficients into an explicit matrix by a formula written here, and
    a dense LU of it has to land on what CG through ``FaceCoeffOp`` landed on. The
    coefficients are the only thing shared, so a wrong ``stencilDiag`` -- which no
    C++-versus-C++ comparison in this suite can see, since every one of them calls
    it -- shows up here as a different solution.

    The four BC rows are what make it a test of the boundary contract as well: the
    periodic one must KEEP its wraparound column and the other three must drop it
    and fold ``sign*aF`` onto the diagonal, with Dirichlet and Neumann disagreeing
    on the sign. Getting either half wrong changes the dense matrix and moves this
    answer.
    """
    _require_bindings()
    geom, ba, dm = _make_mesh(periodic)
    gamma = _const_cell(ba, dm, 1.0)
    alpha = _const_cell(ba, dm, 1.0)
    rhs = _random_rhs(ba, dm)
    sol = _zero_sol(ba, dm)
    out = _out_fields(geom, ba, dm)

    stats = _system_solve(gamma, alpha, geom, rhs, sol, bc, out)
    assert stats["converged"] is True, f"{case} did not converge: {dict(stats)}"

    # A symmetric matrix aliases lower[d] onto upper[d] -- there is one field per
    # direction and it carries both roles, which is why the same array goes in twice.
    upper = [_one_box(out[1 + d]) for d in range(3)]
    a = _dense_operator(_one_box(out[0]), upper, upper, bc, (N, N, N))
    want = np.linalg.solve(a, _flat(_one_box(rhs)))

    np.testing.assert_allclose(
        _flat(_one_box(sol)),
        want,
        rtol=0.0,
        atol=_ORACLE_TOL,
        err_msg=f"{case}: la solve disagrees with the dense numpy operator",
    )
