# SPDX-FileCopyrightText: 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""``blockamr.linear_algebra`` -- the Python surface over ``blockamr::la``.

What this file is for: proving the new surface is WIRED TO THE SAME MACHINERY the
existing ``_la_system_solve`` test seam drives, rather than to a parallel path
that happens to give a similar answer. So the central test assembles the same
problem twice -- once through ``MFFaceCoeffs`` / ``LinearSystem`` / ``laplacian``
/ ``Solver``, once through ``_la_system_solve`` -- and requires the solution to
agree **bitwise**. A tolerance would let the two drift apart into two
implementations of the same idea, which is exactly what this refactor exists to
prevent.

Bitwise is a legitimate demand here and not a lucky one: both paths build the
same format, write the same diagonal source, accumulate the same
``ops::Laplacian`` and hand the result to the same ``la::Solver`` on the same
executor, so every floating-point operation is the same operation in the same
order. If it ever stops being bitwise, something moved.

The problem is a Helmholtz one, ``alpha*phi - div(gamma grad phi)`` with
``alpha = gamma = 1`` and homogeneous Dirichlet on all six sides. Dirichlet
rather than periodic on purpose: it makes the matrix non-singular without a
nullspace projection, and it is the case where ``laplacian()``'s ``bc`` argument
is load-bearing -- with periodic sides a dropped ``bc`` would be invisible.
``alpha`` is non-zero for the same reason: it is what makes
``Matrix.diagonal_source()`` observable.

The rhs is seeded random rather than smooth: a smooth rhs on this problem is
nearly an eigenvector and CG converges in a few iterations, which barely
exercises the mat-vec (S4 handoff §9).

What is NOT here, because it is already pinned elsewhere: the operator's
coefficients bitwise against a hand-built set, the BC fold per kind, accumulation
and ``zero()`` (``test_la_linear_system.py``, ``test_la_boundary_conditions.py``),
and the format-freshness rules (``test_la_matrix_formats.py``). Those go through
the underscore-prefixed seams, which this slice left working untouched.
"""

import numpy as np
import pytest

import blockamr
from blockamr.linear_algebra import (
    CsrMatrix,
    LinearSystem,
    MFFaceCoeffs,
    Solver,
    SolverConfig,
    laplacian,
)

from ._executors import gko_executor

_ext = getattr(blockamr, "_blockamr", None)

_N = 8
_BC = ["dirichlet"] * 6
_SOLVE = dict(solver="cg", max_iter=5000, rtol=1e-14, atol=0.0)
_FORMATS = {"mf": MFFaceCoeffs, "csr": CsrMatrix}


def _mesh():
    """Single box on [0,1]^3, non-periodic -- face fabs align 1:1 with the cells."""
    box = blockamr.Box([0, 0, 0], [_N - 1, _N - 1, _N - 1])
    rb = blockamr.RealBox([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    geom = blockamr.Geometry(box, rb, 0, [0, 0, 0])
    ba = blockamr.BoxArray(box)
    ba.max_size(_N)
    dm = blockamr.DistributionMapping(ba)
    return geom, ba, dm


def _const_cell(ba, dm, value):
    mf = blockamr.MultiFab(ba, dm, 1, 0)
    mf.set_val(value)
    return mf


def _random_rhs(ba, dm, seed=7):
    mf = blockamr.MultiFab(ba, dm, 1, 0)
    rng = np.random.default_rng(seed)
    for mfi in blockamr.MFIterator(mf):
        arr = mf.copy_to_host(mfi)
        arr[:, :, :, 0] = rng.standard_normal(arr.shape[:3])
        mf.copy_from(mfi, arr)
    return mf


def _zero_sol(ba, dm):
    sol = blockamr.MultiFab(ba, dm, 1, 1)
    sol.set_val(0.0)  # MultiFabs are not zero-initialised (arena recycling)
    return sol


def _boxes(mf):
    return [mf.copy_to_host(mfi) for mfi in blockamr.MFIterator(mf)]


def _solve_through_the_python_api(fmt, executor):
    """The surface under test, spelled the way the design's example spells it."""
    geom, ba, dm = _mesh()
    gamma = _const_cell(ba, dm, 1.0)
    alpha = _const_cell(ba, dm, 1.0)
    rhs = _random_rhs(ba, dm)
    sol = _zero_sol(ba, dm)

    matrix = _FORMATS[fmt].symmetric(ba, dm, geom, executor=gko_executor(executor), bc=_BC)
    matrix.diagonal_source(alpha)
    system = LinearSystem(matrix, rhs)
    system += laplacian(gamma, geom, bc=_BC)
    stats = Solver(SolverConfig(**_SOLVE)).solve(system, sol)
    return stats, sol, matrix


def _solve_through_the_system_binding(fmt, executor):
    """The S5 seam the S4/S5/S6b tests use, on an identical problem."""
    geom, ba, dm = _mesh()
    gamma = _const_cell(ba, dm, 1.0)
    alpha = _const_cell(ba, dm, 1.0)
    rhs = _random_rhs(ba, dm)
    sol = _zero_sol(ba, dm)
    out = (
        _const_cell(ba, dm, 0.0),
        *(_face_out(geom, dm, d) for d in range(3)),
    )
    stats = _ext._la_system_solve(
        fmt,
        gamma,
        alpha,
        geom,
        rhs,
        sol,
        *out,
        executor=gko_executor(executor),
        bc=_BC,
        **_SOLVE,
    )
    return stats, sol


def _face_out(geom, dm, d):
    dom = geom.domain()
    face_box = blockamr.Box(dom.small_end(), dom.big_end())
    face_box.surrounding_nodes(d)
    face_ba = blockamr.BoxArray(face_box)
    face_ba.max_size(_N)
    mf = blockamr.MultiFab(face_ba, dm, 1, 0)
    mf.set_val(0.0)
    return mf


@pytest.mark.parametrize("executor", ["reference", "cuda"])
@pytest.mark.parametrize("fmt", ["mf", "csr"])
def test_python_api_reproduces_the_system_binding_bitwise(fmt, executor):
    """The two spellings of one solve land on the same bits, format by format."""
    api_stats, api_sol, _ = _solve_through_the_python_api(fmt, executor)
    ref_stats, ref_sol = _solve_through_the_system_binding(fmt, executor)

    assert api_stats["num_iters"] == ref_stats["num_iters"]
    assert float(api_stats["res_norm"]).hex() == float(ref_stats["res_norm"]).hex()
    for i, (got, want) in enumerate(zip(_boxes(api_sol), _boxes(ref_sol))):
        np.testing.assert_array_equal(
            got, want, err_msg=f"{fmt}/{executor}: solution box {i} differs bitwise"
        )


@pytest.mark.parametrize("fmt,assembled", [("mf", False), ("csr", True)])
def test_matrix_reports_the_format_it_holds(fmt, assembled):
    """The erasure still answers the three questions a caller may ask of it."""
    geom, ba, dm = _mesh()
    matrix = _FORMATS[fmt].symmetric(ba, dm, geom, executor=gko_executor("reference"), bc=_BC)

    assert matrix.is_assembled() is assembled
    assert matrix.is_symmetric() is True
    assert matrix.local_rows() == _N**3


def test_precond_is_refused_with_the_gmg_hierarchy_explanation():
    """precond='gmg' is a real gap, and the message says why rather than falling back."""
    geom, ba, dm = _mesh()
    gamma = _const_cell(ba, dm, 1.0)
    rhs = _random_rhs(ba, dm)
    sol = _zero_sol(ba, dm)
    matrix = MFFaceCoeffs.symmetric(ba, dm, geom, executor=gko_executor("reference"), bc=_BC)
    system = LinearSystem(matrix, rhs)
    system += laplacian(gamma, geom, bc=_BC)

    with pytest.raises(RuntimeError, match="precond 'gmg' is not wired up yet"):
        Solver(SolverConfig(solver="cg", precond="gmg")).solve(system, sol)


@pytest.mark.parametrize("solver", ["gmg", "ir", "mpir"])
def test_gmg_family_solvers_are_refused_with_the_hierarchy_explanation(solver):
    """solver='gmg'/'ir'/'mpir' need the coefficient-field hierarchy, not a LinOp."""
    geom, ba, dm = _mesh()
    gamma = _const_cell(ba, dm, 1.0)
    rhs = _random_rhs(ba, dm)
    sol = _zero_sol(ba, dm)
    matrix = MFFaceCoeffs.symmetric(ba, dm, geom, executor=gko_executor("reference"), bc=_BC)
    system = LinearSystem(matrix, rhs)
    system += laplacian(gamma, geom, bc=_BC)

    with pytest.raises(RuntimeError, match="needs the GMG hierarchy"):
        Solver(SolverConfig(solver=solver)).solve(system, sol)


def test_an_unknown_solver_is_rejected_when_the_solver_is_built():
    """The spelling is parsed once, at construction -- not at the first solve."""
    with pytest.raises(RuntimeError, match="unknown solver 'not-a-solver'"):
        Solver(SolverConfig(solver="not-a-solver"))


def test_solver_config_rejects_a_nonpositive_max_iter():
    """A config is validated where it is built, like GmgConfig beside it."""
    with pytest.raises(ValueError):
        SolverConfig(max_iter=0)
