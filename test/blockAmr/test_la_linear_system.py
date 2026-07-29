# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""``blockamr::la::LinearSystem`` and the first operator, ``ops::Laplacian``.

A ``LinearSystem`` is a ``Matrix`` and an rhs, and its ``operator+=`` is the only
way an operator ever runs: ``Coefficients`` (what an operator writes through) has
a private constructor whose sole friend is ``LinearSystem``, and
``Operator::assemble`` is private with the same friend. Neither gate is testable
from here -- code that violated one would not compile -- so both are asserted in
``linearAlgebra/coefficientsConcepts.cpp`` instead.

What IS testable, and is what this file checks:

* ``system += ops::Laplacian(gamma, geom, bc)`` writes **bitwise** the face
  coefficients a caller writes by hand today, periodic and non-periodic, on both
  matrix formats -- and the two paths then solve to the same answer. Non-periodic
  is the same statement and not a special case: the boundary condition is applied
  by the matrix from the live coefficient, so the operator writes no diagonal
  source and drops no face (see ``operators/laplacian.hpp``);
* the face coefficient is the two-cell mean of ``gamma`` over ``dx**2``, with the
  right neighbour on every INTERIOR face (periodic wraparound included) and the
  boundary cell's own value on a non-periodic DOMAIN face, which has no second
  cell. Constant ``gamma`` cannot tell an off-by-one-cell mistake from a correct
  one, so one case varies it;
* operators ACCUMULATE -- applying the same one twice doubles the coefficients;
* ``zero()`` clears the coefficients *and* the rhs.

The boundary condition itself -- every kind, both formats, the inhomogeneous rhs
term -- lives in ``test_la_boundary_conditions.py``; what is pinned here is the
operator's coefficient arithmetic and the ``LinearSystem`` mechanics around it.

Bitwise, not ``allclose``: the claim is that the operator writes *exactly* what
the hand-built path writes, and a tolerance would hide an off-by-one-cell or a
factor that happens to be small on this mesh.

The entry points are ``blockamr._blockamr._la_system_solve`` / ``_la_system_probe``
-- test-facing bindings in the same style S4 added for ``Matrix``, since blockAmr
has no C++ test target that builds.
"""

import numpy as np
import pytest

import blockamr

from ._executors import gko_executor

_ext = getattr(blockamr, "_blockamr", None)


def _require_bindings():
    if _ext is None or not hasattr(_ext, "_la_system_solve"):
        pytest.skip("blockamr._la_system_solve binding not available")


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


def _face_field(geom, dm, d, n, value):
    dom = geom.domain()
    face_box = blockamr.Box(dom.small_end(), dom.big_end())
    face_box.surrounding_nodes(d)
    face_ba = blockamr.BoxArray(face_box)
    face_ba.max_size(n)
    mf = blockamr.MultiFab(face_ba, dm, 1, 0)
    mf.set_val(value)
    return mf


def _inv_dx2(geom, d):
    """1/dx**2 spelled as the operator spells it: 1.0 / (dx*dx), not dx**-2.

    The comparison below is bitwise, so the expression -- not just the value --
    has to match ops::Laplacian's `1.0 / (dx[d] * dx[d])`.
    """
    dx = geom.cell_size()
    return 1.0 / (dx[d] * dx[d])


def _hand_built_faces(geom, dm, n):
    """The three face fields a caller writes today for a unit-gamma Laplacian."""
    return [_face_field(geom, dm, d, n, -_inv_dx2(geom, d)) for d in range(3)]


def _expected_alpha(alpha_in):
    """alpha comes back exactly as it went in, under EVERY bc.

    The operator writes no diagonal SOURCE of its own (there is no ops::Ddt yet),
    and a non-periodic boundary does not make it write one: the boundary condition
    stays derivable from the live face coefficient, which is what lets the GMG
    hierarchy re-derive it per level (``operators/laplacian.hpp``).
    """
    return _one_box(alpha_in).copy()


def _out_fields(geom, ba, dm, n):
    """Zeroed receivers for the assembled (alpha, ux, uy, uz)."""
    return _const_cell(ba, dm, 0.0), *(_face_field(geom, dm, d, n, 0.0) for d in range(3))


def _boxes(mf):
    return [mf.copy_to_host(mfi) for mfi in blockamr.MFIterator(mf)]


def _one_box(mf):
    """The single box of a single-box MultiFab, as (i, j, k) with the component dropped."""
    boxes = _boxes(mf)
    assert len(boxes) == 1
    return boxes[0][:, :, :, 0]


def _random_cells(ba, dm, seed, low=0.0, high=1.0):
    mf = blockamr.MultiFab(ba, dm, 1, 0)
    rng = np.random.default_rng(seed)
    for mfi in blockamr.MFIterator(mf):
        arr = mf.copy_to_host(mfi)
        arr[:, :, :, 0] = rng.uniform(low, high, arr.shape[:3])
        mf.copy_from(mfi, arr)
    return mf


def _random_rhs(ba, dm, seed=42):
    """Seeded random rhs -- the full spectrum, so CG has to iterate across it.

    A smooth rhs on this problem is nearly an eigenvector and converges in three
    iterations, which barely exercises the mat-vec at all (S4 handoff §9).
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


def _max_abs_diff(a, b):
    return max(float(np.max(np.abs(x - y))) for x, y in zip(_boxes(a), _boxes(b)))


def _assert_bitwise(got, want, what):
    for i, (g, w) in enumerate(zip(_boxes(got), _boxes(want))):
        np.testing.assert_array_equal(g, w, err_msg=f"{what}: box {i} differs bitwise")


_SOLVE_KWARGS = dict(solver="cg", max_iter=5000, rtol=1e-14, atol=0.0)

# Periodic and one non-periodic case. The coefficients an operator writes are the
# same either way EXCEPT for which cells a domain face averages gamma over, which
# only the varying-gamma row below can see (`nonperiodic` selects its expectation).
# The non-periodic row is the one S6a made reachable through CsrMatrix.
_BC_CASES = [
    ("periodic", [1, 1, 1], ["periodic"] * 6, False),
    ("dirichlet", [0, 0, 0], ["dirichlet"] * 6, True),
]

_AGREE_TOL = 1e-12


def _skip_on_missing_ginkgo(exc, executor="reference"):
    if "without Ginkgo" in str(exc):
        pytest.skip("blockamr built without Ginkgo")
    if executor == "cuda":
        pytest.skip(f"cuda executor unavailable: {exc}")
    raise exc


def _probe(fmt, gamma, alpha, geom, ba, dm, n, bc, out, **kwargs):
    try:
        return _ext._la_system_probe(
            fmt,
            gamma,
            alpha,
            geom,
            _random_rhs(ba, dm),
            *out,
            executor=gko_executor("reference"),
            bc=bc,
            **kwargs,
        )
    except RuntimeError as exc:
        _skip_on_missing_ginkgo(exc)


def _system_solve(fmt, gamma, alpha, geom, ba, dm, rhs, sol, bc, out, executor):
    try:
        return _ext._la_system_solve(
            fmt,
            gamma,
            alpha,
            geom,
            rhs,
            sol,
            *out,
            executor=gko_executor(executor),
            bc=bc,
            **_SOLVE_KWARGS,
        )
    except RuntimeError as exc:
        _skip_on_missing_ginkgo(exc, executor)


def _reference_solve(geom, ba, dm, n, rhs, bc, executor):
    """The hand-built FaceCoeffSolver the operator path has to reproduce."""
    alpha = _const_cell(ba, dm, 1.0)
    fx, fy, fz = _hand_built_faces(geom, dm, n)
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
        _skip_on_missing_ginkgo(exc, executor)
    stats = s.solve(rhs, sol)
    assert stats["converged"] is True, f"reference did not converge: {dict(stats)}"
    return sol, stats


@pytest.mark.parametrize("fmt", ["mf", "csr"])
@pytest.mark.parametrize("case, periodic, bc, nonperiodic", _BC_CASES)
def test_laplacian_writes_the_hand_built_coefficients(
    blockamr_session, fmt, case, periodic, bc, nonperiodic
):
    """`system += ops::Laplacian(gamma=1, ...)` gives bitwise the hand-built faces.

    Bitwise is the claim: the operator is a replacement for the seven-MultiFab
    call every caller writes today, not an approximation of it. `nonperiodic` is
    deliberately NOT consulted -- at constant gamma both rows expect the identical
    coefficients, because the boundary condition is not in them: `alpha` comes back
    exactly as it went in and every face, domain ones included, carries the
    hand-built value.
    """
    _require_bindings()
    n = 16
    geom, ba, dm = _make_mesh(n, periodic)
    gamma = _const_cell(ba, dm, 1.0)
    alpha = _const_cell(ba, dm, 1.0)
    out = _out_fields(geom, ba, dm, n)

    _probe(fmt, gamma, alpha, geom, ba, dm, n, bc, out)

    want_faces = _hand_built_faces(geom, dm, n)
    np.testing.assert_array_equal(
        _one_box(out[0]), _expected_alpha(alpha), err_msg=f"{fmt}/{case} alpha"
    )
    for d, name in enumerate("xyz"):
        _assert_bitwise(out[1 + d], want_faces[d], f"{fmt}/{case} u{name}")


@pytest.mark.parametrize("fmt", ["mf", "csr"])
@pytest.mark.parametrize("case, periodic, bc, nonperiodic", _BC_CASES)
def test_system_solve_matches_hand_built_solver(
    blockamr_session, fmt, case, periodic, bc, nonperiodic
):
    """The assembled system solves to the hand-built solver's answer.

    Same mesh, same coefficients, same CG. The random rhs is deliberate: a smooth
    one converges in three iterations and would compare two solves that barely ran.
    """
    _require_bindings()
    n = 16
    geom, ba, dm = _make_mesh(n, periodic)
    gamma = _const_cell(ba, dm, 1.0)
    alpha = _const_cell(ba, dm, 1.0)
    rhs = _random_rhs(ba, dm)

    ref, ref_stats = _reference_solve(geom, ba, dm, n, rhs, bc, "reference")

    sol = _zero_sol(ba, dm)
    out = _out_fields(geom, ba, dm, n)
    stats = _system_solve(fmt, gamma, alpha, geom, ba, dm, rhs, sol, bc, out, "reference")

    assert stats["converged"] is True, f"{fmt}/{case} did not converge: {dict(stats)}"
    assert stats["num_iters"] > 10, f"{fmt}/{case}: too few CG iterations to mean anything"
    assert stats["is_assembled"] is (fmt == "csr")
    assert stats["local_rows"] == n**3

    diff = _max_abs_diff(ref, sol)
    assert diff < _AGREE_TOL, (
        f"{fmt}/{case}: |FaceCoeffSolver - LinearSystem({fmt})| = {diff} exceeds {_AGREE_TOL} "
        f"(ref {dict(ref_stats)}, got {dict(stats)})"
    )


@pytest.mark.parametrize("case, periodic, bc, nonperiodic", _BC_CASES)
def test_laplacian_face_gamma_is_the_two_cell_average(
    blockamr_session, case, periodic, bc, nonperiodic
):
    """A varying gamma pins WHICH two cells each face averages.

    With gamma constant every neighbour choice gives the same number, so a
    wraparound written as a clamp -- or a face indexed one cell off -- passes the
    test above unnoticed. Here the expectation is computed per face from the cell
    values: periodic faces wrap. A non-periodic DOMAIN face has only ONE adjacent
    cell, so it averages the boundary cell's gamma with itself; the ghost on the
    far side is never filled and reading it would be reading recycled memory. This
    is the only row in the suite that can tell that apart from reading the ghost,
    or from the zero the coefficient used to be given there.
    """
    _require_bindings()
    n = 8
    geom, ba, dm = _make_mesh(n, periodic)
    gamma = _random_cells(ba, dm, seed=7, low=0.5, high=1.5)
    alpha = _const_cell(ba, dm, 1.0)
    out = _out_fields(geom, ba, dm, n)

    _probe("mf", gamma, alpha, geom, ba, dm, n, bc, out)

    g = _one_box(gamma)
    wrap = case == "periodic"
    for d in range(3):
        inv_dx2 = _inv_dx2(geom, d)
        got = _one_box(out[1 + d])
        for f in range(n + 1):
            if nonperiodic and f in (0, n):
                # Both sides are the single interior cell: 0 at the low face,
                # n-1 at the high one.
                lo = hi = 0 if f == 0 else n - 1
            else:
                lo = (f - 1) % n if wrap else f - 1
                hi = f % n if wrap else f
            want = -0.5 * (np.take(g, lo, axis=d) + np.take(g, hi, axis=d)) * inv_dx2
            np.testing.assert_array_equal(
                np.take(got, f, axis=d), want, err_msg=f"{case}: dir {d}, face {f}"
            )


def test_operator_accumulates_rather_than_assigns(blockamr_session):
    """Applying the same operator twice doubles the coefficients.

    Several operators share one system, so an operator that assigned would silently
    discard whatever ran before it. Doubling is exact in binary, so this stays a
    bitwise check.
    """
    _require_bindings()
    n = 16
    geom, ba, dm = _make_mesh(n, [1, 1, 1])
    gamma = _const_cell(ba, dm, 1.0)
    alpha = _const_cell(ba, dm, 1.0)
    bc = ["periodic"] * 6

    once = _out_fields(geom, ba, dm, n)
    twice = _out_fields(geom, ba, dm, n)
    _probe("mf", gamma, alpha, geom, ba, dm, n, bc, once, n_apply=1)
    _probe("mf", gamma, alpha, geom, ba, dm, n, bc, twice, n_apply=2)

    for d, name in enumerate("xyz"):
        np.testing.assert_array_equal(
            _one_box(twice[1 + d]), 2.0 * _one_box(once[1 + d]), err_msg=f"u{name} did not double"
        )
    # alpha is written by the binding, not by the operator, so it does NOT double.
    np.testing.assert_array_equal(_one_box(twice[0]), _one_box(once[0]))


def test_system_zero_clears_coefficients_and_rhs(blockamr_session):
    """zero() clears the matrix and the rhs together -- they are one system."""
    _require_bindings()
    n = 8
    geom, ba, dm = _make_mesh(n, [1, 1, 1])
    gamma = _const_cell(ba, dm, 1.0)
    alpha = _const_cell(ba, dm, 1.0)
    bc = ["periodic"] * 6
    out = _out_fields(geom, ba, dm, n)

    kept = _probe("mf", gamma, alpha, geom, ba, dm, n, bc, out)
    assert kept["rhs_sum"] != 0.0  # the random rhs the probe built

    cleared = _probe("mf", gamma, alpha, geom, ba, dm, n, bc, out, zero_after=True)
    assert cleared["rhs_sum"] == 0.0
    for field in out:
        assert np.count_nonzero(_one_box(field)) == 0


@pytest.mark.parametrize("fmt", ["mf", "csr"])
def test_system_reports_the_matrix_shape_and_holds_the_caller_rhs(blockamr_session, fmt):
    """localRows() comes from the matrix, and the rhs is the caller's own MultiFab.

    localRows() is the RANK-LOCAL count; it coincides with numPts() on one rank,
    which is exactly why it is worth asserting somewhere that it is read off the
    matrix. The rhs check pins the non-owning contract: an operator's contribution
    to b is visible to the caller with no copy-back step.
    """
    _require_bindings()
    n = 8
    geom, ba, dm = _make_mesh(n, [1, 1, 1])
    gamma = _const_cell(ba, dm, 1.0)
    alpha = _const_cell(ba, dm, 1.0)
    out = _out_fields(geom, ba, dm, n)

    d = _probe(fmt, gamma, alpha, geom, ba, dm, n, ["periodic"] * 6, out)
    assert d["local_rows"] == n**3
    assert d["symmetric"] is True
    assert d["is_assembled"] is (fmt == "csr")
    assert d["rhs_aliases_input"] is True


def test_system_solve_stats_keys_match_face_coeff_solver(blockamr_session):
    """la::Solver returns the SAME dict keys every other solve entry point returns.

    S5 reuses la::SolveResult rather than adding the design's parallel
    SolverStats; the uniform key set is a contract a caller reads without branching
    on which solver produced it (test_gmg_solver_stats_keys_match_cg pins it for
    the older paths, this pins the new one against them).
    """
    _require_bindings()
    n = 8
    geom, ba, dm = _make_mesh(n, [1, 1, 1])
    gamma = _const_cell(ba, dm, 1.0)
    alpha = _const_cell(ba, dm, 1.0)
    rhs = _random_rhs(ba, dm)
    bc = ["periodic"] * 6

    _, ref_stats = _reference_solve(geom, ba, dm, n, rhs, bc, "reference")
    sol = _zero_sol(ba, dm)
    out = _out_fields(geom, ba, dm, n)
    stats = _system_solve("mf", gamma, alpha, geom, ba, dm, rhs, sol, bc, out, "reference")

    extra = {"is_assembled", "local_rows", "symmetric", "rhs_aliases_input"}
    assert set(stats.keys()) - extra == set(ref_stats.keys())
