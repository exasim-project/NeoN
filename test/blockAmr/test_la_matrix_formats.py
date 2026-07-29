# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""The ``blockamr::la`` matrix formats behind the ``Matrix`` type erasure.

``Matrix`` holds any type satisfying ``IsMatrix``; two do. ``MFFaceCoeffs`` is
matrix-free (``op()`` is a ``FaceCoeffOp`` over the coefficient fields, nothing is
assembled), ``CsrMatrix`` is assembled (``op()`` is the explicit Ginkgo Csr that
``assembleFaceCoeffCsr`` builds from the SAME fields). Both allocate their own
``alpha`` plus the six face fields and hand out the same ``CellView``/``FaceView``,
which is the whole point: one write face, one view type, no second erasure.

What is checked here:

* both formats, filled only through ``Matrix::coefficients()``, reproduce a
  hand-built ``FaceCoeffSolver`` on the same problem — periodic and non-periodic;
* the erasure's ``clone()``: a copy of a ``Matrix`` solves the same problem;
* the assembly-freshness rule — ``CsrMatrix::op()`` reassembles after a write
  through ``coefficients()`` or ``zero()`` and reuses its matrix when nothing was
  written;
* ``localRows()`` is the rank-local count and the operator's row count is global.

The entry points are ``blockamr._blockamr._la_matrix_solve`` / ``_la_matrix_probe``
— test-facing bindings, since blockAmr has no C++ test target that builds. They
are not a solver interface; S5 brings that.
"""

import numpy as np
import pytest

import blockamr

from ._executors import gko_executor

# The private extension module: the la bindings are underscore-prefixed, so
# `from ._blockamr import *` in blockamr/__init__.py does not re-export them.
_ext = getattr(blockamr, "_blockamr", None)


def _require_bindings():
    if _ext is None or not hasattr(_ext, "_la_matrix_solve"):
        pytest.skip("blockamr._la_matrix_solve binding not available")


def _make_mesh(n, periodic):
    """Single-box mesh on [0,1]^3 with n cells per side and given periodicity."""
    box = blockamr.Box([0, 0, 0], [n - 1, n - 1, n - 1])
    rb = blockamr.RealBox([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    geom = blockamr.Geometry(box, rb, 0, periodic)
    ba = blockamr.BoxArray(box)
    ba.max_size(n)  # single box -> face fabs align 1:1 with the cell fab
    dm = blockamr.DistributionMapping(ba)
    return geom, ba, dm


def _const_cell(ba, dm, value):
    mf = blockamr.MultiFab(ba, dm, 1, 0)
    mf.set_val(value)
    return mf


def _const_face(geom, dm, d, n, value):
    dom = geom.domain()
    face_box = blockamr.Box(dom.small_end(), dom.big_end())
    face_box.surrounding_nodes(d)
    face_ba = blockamr.BoxArray(face_box)
    face_ba.max_size(n)
    mf = blockamr.MultiFab(face_ba, dm, 1, 0)
    mf.set_val(value)
    return mf


def _poisson_coeffs(geom, ba, dm, n, alpha_val):
    """alpha=alpha_val cell source + symmetric -1/dx^2 face coeffs on ALL faces."""
    dx = geom.cell_size()
    inv_dx2 = 1.0 / dx[0] ** 2
    alpha = _const_cell(ba, dm, alpha_val)
    fx = _const_face(geom, dm, 0, n, -inv_dx2)
    fy = _const_face(geom, dm, 1, n, -inv_dx2)
    fz = _const_face(geom, dm, 2, n, -inv_dx2)
    return alpha, fx, fy, fz


def _random_rhs(ba, dm, seed=42):
    """Seeded random rhs — the full spectrum, so CG has to iterate across it.

    A smooth rhs here is nearly an eigenvector and converges in three iterations,
    which barely exercises the mat-vec at all; this one takes 70-90, so the
    agreement bar below measures two mat-vec implementations that were each run
    many times, not two solves that stopped before they diverged.
    """
    rng = np.random.default_rng(seed)
    rhs = blockamr.MultiFab(ba, dm, 1, 0)
    for mfi in blockamr.MFIterator(rhs):
        arr = rhs.copy_to_host(mfi)
        arr[:, :, :, 0] = rng.standard_normal(arr.shape[:3])
        rhs.copy_from(mfi, arr)
    return rhs


def _zero_sol(ba, dm):
    sol = blockamr.MultiFab(ba, dm, 1, 1)
    sol.set_val(0.0)  # MultiFabs are not zero-initialized (arena recycling)
    return sol


def _max_abs_diff(a, b):
    a_boxes = [a.copy_to_host(mfi) for mfi in blockamr.MFIterator(a)]
    b_boxes = [b.copy_to_host(mfi) for mfi in blockamr.MFIterator(b)]
    return max(float(np.max(np.abs(x - y))) for x, y in zip(a_boxes, b_boxes))


_SOLVE_KWARGS = dict(solver="cg", max_iter=5000, rtol=1e-14, atol=0.0)


def _reference_solve(coeffs, geom, ba, dm, rhs, executor, bc):
    """The hand-built FaceCoeffSolver every format below has to reproduce."""
    alpha, fx, fy, fz = coeffs
    sol = _zero_sol(ba, dm)
    try:
        s = blockamr.FaceCoeffSolver(
            alpha,
            fx,
            fx,
            fy,
            fy,
            fz,
            fz,
            geom,
            executor=gko_executor(executor),
            bc=bc,
            **_SOLVE_KWARGS,
        )
    except RuntimeError as exc:
        if "without Ginkgo" in str(exc):
            pytest.skip("blockamr built without Ginkgo")
        if executor == "cuda":
            pytest.skip(f"cuda executor unavailable: {exc}")
        raise
    stats = s.solve(rhs, sol)
    assert stats["converged"] is True, f"reference did not converge: {dict(stats)}"
    return sol, stats


def _matrix_solve(fmt, coeffs, geom, ba, dm, rhs, executor, bc, **kwargs):
    """Solve through a blockamr::la::Matrix holding the named format."""
    alpha, fx, fy, fz = coeffs
    sol = _zero_sol(ba, dm)
    try:
        stats = _ext._la_matrix_solve(
            fmt,
            alpha,
            fx,
            fx,
            fy,
            fy,
            fz,
            fz,
            geom,
            rhs,
            sol,
            executor=gko_executor(executor),
            bc=bc,
            **_SOLVE_KWARGS,
            **kwargs,
        )
    except RuntimeError as exc:
        if "without Ginkgo" in str(exc):
            pytest.skip("blockamr built without Ginkgo")
        if executor == "cuda":
            pytest.skip(f"cuda executor unavailable: {exc}")
        raise
    return sol, stats


# The two boundary cases the formats must agree on. Periodic is what S4's
# acceptance asks for; the non-periodic row is only reachable because S6a taught
# assembleFaceCoeffCsr about boundaries, and it is the direct rehearsal for S6b's
# format-agreement check. alpha=1 (Helmholtz) keeps both nonsingular, so no
# nullspace projection enters the comparison.
_BC_CASES = [
    ("periodic", [1, 1, 1], ["periodic"] * 6),
    ("dirichlet", [0, 0, 0], ["dirichlet"] * 6),
]

_AGREE_TOL = 1e-12


@pytest.mark.parametrize("executor", ["reference", "cuda"])
@pytest.mark.parametrize("fmt", ["mf", "csr"])
@pytest.mark.parametrize("case, periodic, bc", _BC_CASES)
def test_format_matches_hand_built_solver(blockamr_session, executor, fmt, case, periodic, bc):
    """Both formats, filled through Matrix::coefficients(), reproduce FaceCoeffSolver.

    Same mesh, same coefficients, same CG, one answer. This is what makes the
    erasure worth having: nothing above Matrix knows whether it drove a
    matrix-free operator or an assembled Csr, and the answer does not either.
    """
    _require_bindings()
    n = 16
    geom, ba, dm = _make_mesh(n, periodic)
    coeffs = _poisson_coeffs(geom, ba, dm, n, 1.0)
    rhs = _random_rhs(ba, dm)

    ref, ref_stats = _reference_solve(coeffs, geom, ba, dm, rhs, executor, bc)
    got, stats = _matrix_solve(fmt, coeffs, geom, ba, dm, rhs, executor, bc)

    assert stats["converged"] is True, f"{fmt}/{case} did not converge: {dict(stats)}"
    assert stats["num_iters"] > 10, f"{fmt}/{case}: too few CG iterations to mean anything"
    assert stats["is_assembled"] is (fmt == "csr")
    assert stats["local_rows"] == n**3
    assert stats["symmetric"] is True

    diff = _max_abs_diff(ref, got)
    assert diff < _AGREE_TOL, (
        f"{fmt}/{case}: |FaceCoeffSolver - Matrix({fmt})| = {diff} exceeds {_AGREE_TOL} "
        f"(ref {dict(ref_stats)}, got {dict(stats)})"
    )


@pytest.mark.parametrize("fmt", ["mf", "csr"])
def test_matrix_copy_solves_identically(blockamr_session, fmt):
    """A copied Matrix is the same matrix — the erasure's clone() path.

    The copy shares the coefficient fields (amrex::MultiFab cannot be copied) and,
    for the assembled format, the assembly-freshness state with them, so it must
    land on exactly the original's answer, not merely near it.
    """
    _require_bindings()
    n = 16
    geom, ba, dm = _make_mesh(n, [1, 1, 1])
    coeffs = _poisson_coeffs(geom, ba, dm, n, 1.0)
    rhs = _random_rhs(ba, dm)
    bc = ["periodic"] * 6

    direct, _ = _matrix_solve(fmt, coeffs, geom, ba, dm, rhs, "reference", bc)
    copied, _ = _matrix_solve(fmt, coeffs, geom, ba, dm, rhs, "reference", bc, via_copy=True)
    assert _max_abs_diff(direct, copied) == 0.0


@pytest.mark.parametrize("fmt", ["mf", "csr"])
def test_op_before_write_does_not_freeze_the_matrix(blockamr_session, fmt):
    """op() called BEFORE the coefficients are written still solves the right system.

    The formats allocate their fields zeroed, so an op() taken at that point is an
    operator for the zero matrix. A CsrMatrix that cached it and did not notice the
    subsequent write through coefficients() would then solve with that zero matrix
    — this is the assembly-freshness rule under test. Since S7 the matrix-free
    format is no longer merely the control: op() also computes and caches the
    stored diagonal, so an MFFaceCoeffs missing the same flag fails here too.
    """
    _require_bindings()
    n = 16
    geom, ba, dm = _make_mesh(n, [1, 1, 1])
    coeffs = _poisson_coeffs(geom, ba, dm, n, 1.0)
    rhs = _random_rhs(ba, dm)
    bc = ["periodic"] * 6

    fresh, _ = _matrix_solve(fmt, coeffs, geom, ba, dm, rhs, "reference", bc)
    stale, stats = _matrix_solve(
        fmt, coeffs, geom, ba, dm, rhs, "reference", bc, assemble_before_write=True
    )
    assert stats["converged"] is True, dict(stats)
    assert _max_abs_diff(fresh, stale) == 0.0


def test_asymmetric_factory_matches_symmetric_one(blockamr_session):
    """asymmetric() with lower == upper is the symmetric matrix.

    The two factories differ in storage, not in mathematics: symmetric aliases the
    low-side fields onto the high-side ones and reports an empty `lower` view, so
    writing the same coefficients through the asymmetric form's three extra fields
    has to give the identical operator.
    """
    _require_bindings()
    n = 16
    geom, ba, dm = _make_mesh(n, [1, 1, 1])
    coeffs = _poisson_coeffs(geom, ba, dm, n, 1.0)
    rhs = _random_rhs(ba, dm)
    bc = ["periodic"] * 6

    sym, _ = _matrix_solve("mf", coeffs, geom, ba, dm, rhs, "reference", bc, symmetry="symmetric")
    asym, stats = _matrix_solve(
        "mf", coeffs, geom, ba, dm, rhs, "reference", bc, symmetry="asymmetric"
    )
    assert stats["symmetric"] is False
    assert _max_abs_diff(sym, asym) == 0.0


def _probe(fmt, symmetry="symmetric"):
    n = 16
    geom, ba, dm = _make_mesh(n, [1, 1, 1])
    alpha, fx, fy, fz = _poisson_coeffs(geom, ba, dm, n, 1.0)
    try:
        return n, _ext._la_matrix_probe(
            fmt,
            alpha,
            fx,
            fx,
            fy,
            fy,
            fz,
            fz,
            geom,
            executor=gko_executor("reference"),
            symmetry=symmetry,
        )
    except RuntimeError as exc:
        if "without Ginkgo" in str(exc):
            pytest.skip("blockamr built without Ginkgo")
        raise


@pytest.mark.parametrize("fmt, assembled", [("mf", False), ("csr", True)])
def test_matrix_reports_its_shape(blockamr_session, fmt, assembled):
    """localRows() is the rank-local count; op()'s row count is the global one.

    They coincide on one rank, which is exactly why the distinction is worth
    asserting somewhere: nothing in this suite would catch localRows() returning
    boxArray().numPts() instead, and on more than one rank that is a wrong size
    handed to every vector built over the matrix.
    """
    _require_bindings()
    n, d = _probe(fmt)
    assert d["is_assembled"] is assembled
    assert d["symmetric"] is True
    assert d["local_rows"] == n**3
    assert d["op_rows"] == n**3


@pytest.mark.parametrize("symmetry, lower_empty", [("symmetric", True), ("asymmetric", False)])
def test_coefficient_views_report_symmetry(blockamr_session, symmetry, lower_empty):
    """A symmetric matrix reports an EMPTY `lower` view — that is how the interface
    says "there is no low side to write", and MatrixCoefficients::symmetric() is
    defined as exactly that."""
    _require_bindings()
    _, d = _probe("mf", symmetry=symmetry)
    assert d["diag_empty"] is False
    assert d["upper_empty"] is False
    assert d["lower_empty"] is lower_empty
    assert d["reports_symmetric"] is lower_empty
    assert d["symmetric"] is (symmetry == "symmetric")


def test_csr_assembles_once_until_written(blockamr_session):
    """The dirty flag: assemble on demand, once, and again after every write.

    Two op() calls with nothing between return the SAME matrix; a write through
    coefficients() or zero() forces the next one to reassemble. The flag is set
    when the handles are handed out, not when a value actually changes — there is
    no "done writing" call to hook, so a caller that takes the handles and writes
    nothing pays one redundant assembly.
    """
    _require_bindings()
    _, d = _probe("csr")
    assert d["op_stable_without_write"] is True
    assert d["op_rebuilt_after_coefficients"] is True
    assert d["op_rebuilt_after_zero"] is True


def test_matrix_free_op_is_rebuilt_every_call(blockamr_session):
    """MFFaceCoeffs builds a fresh operator on every op() call, which is why its
    freshness rule is about the stored DIAGONAL and not about the operator: on the
    host path the operator stages PINNED COPIES of the coefficients at
    construction, so a cached operator would go stale after a write on that path
    and not on the device path. What is cached across calls is the diagonal
    (test_la_stored_diagonal.py), and this test pins that the operator is not."""
    _require_bindings()
    _, d = _probe("mf")
    assert d["op_stable_without_write"] is False
    assert d["op_rebuilt_after_coefficients"] is True
    assert d["op_rebuilt_after_zero"] is True
