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
same matrix, write the same diagonal source, accumulate the same
``ops::Laplacian`` and hand the result to the same ``la::Solver`` on the same
executor, so every floating-point operation is the same operation in the same
order. If it ever stops being bitwise, something moved.

The problem is a Helmholtz one, ``alpha*phi - div(gamma grad phi)`` with
``alpha = gamma = 1`` and homogeneous Dirichlet on all six sides. Dirichlet
rather than periodic on purpose: it makes the matrix non-singular without a
nullspace projection, and it is the case where the matrix's ``bc`` is
load-bearing -- with periodic sides a dropped ``bc`` would be invisible.
``alpha`` is non-zero for the same reason: it is what makes
``MFFaceCoeffs.diagonal_source()`` observable.

The rhs is seeded random rather than smooth: a smooth rhs on this problem is
nearly an eigenvector and CG converges in a few iterations, which barely
exercises the mat-vec (S4 handoff §9).

The second half of the file is about PRECONDITIONERS, which are built from the
MATRIX (``la::makeHierarchy``) rather than by the solver, because the GMG
hierarchy is rediscretised from the coefficient FIELDS and the solver holds only
a ``gko::LinOp``. The gate there is CG's own behaviour: a preconditioner
cannot move the fixed point, only the number of iterations needed to reach it, so
"same answer" and "materially fewer iterations" together say the cycle is both
correct and actually running.

What is NOT here, because it is already pinned elsewhere: the operator's
coefficients bitwise against a hand-built set, the BC fold per kind, accumulation
and ``zero()`` (``test_la_linear_system.py``, ``test_la_boundary_conditions.py``),
and the format-freshness rules (``test_la_matrix_formats.py``). Those go through
the underscore-prefixed seams, which this slice left working untouched. The GMG
hierarchy's own correctness is pinned by ``test_ginkgo_gmg.py`` /
``test_ginkgo_gmg_kokkos.py`` against ``FaceCoeffSolver``; what is new here is
only that the same hierarchy is reachable through ``la``.
"""

import numpy as np
import pytest

import blockamr
from blockamr.linear_algebra import (
    GmgConfig,
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

# The preconditioner tests solve a bigger problem to a looser tolerance -- see
# _precond_system for the size, and note the tolerance: at rtol=1e-14 CG runs into
# the residual's own round-off floor, where an iteration count stops measuring the
# preconditioner and starts measuring noise.
_PRECOND_N = 32
_PRECOND_SOLVE = dict(solver="cg", max_iter=5000, rtol=1e-10, atol=0.0)


def _mesh(n=_N):
    """Single box on [0,1]^3, non-periodic -- face fabs align 1:1 with the cells."""
    box = blockamr.Box([0, 0, 0], [n - 1, n - 1, n - 1])
    rb = blockamr.RealBox([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    geom = blockamr.Geometry(box, rb, 0, [0, 0, 0])
    ba = blockamr.BoxArray(box)
    ba.max_size(n)
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


def _solve_through_the_python_api(executor):
    """The surface under test, spelled the way the design's example spells it."""
    geom, ba, dm = _mesh()
    gamma = _const_cell(ba, dm, 1.0)
    alpha = _const_cell(ba, dm, 1.0)
    rhs = _random_rhs(ba, dm)
    sol = _zero_sol(ba, dm)

    matrix = MFFaceCoeffs.symmetric(
        blockamr.MeshLevel(ba, dm, geom), executor=gko_executor(executor), bc=_BC
    )
    matrix.diagonal_source(alpha)
    system = LinearSystem(matrix, rhs)
    system += laplacian(gamma)
    stats = Solver(SolverConfig(**_SOLVE)).solve(system, sol)
    return stats, sol, matrix


def _solve_through_the_system_binding(executor):
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
def test_python_api_reproduces_the_system_binding_bitwise(executor):
    """The two spellings of one solve land on the same bits."""
    api_stats, api_sol, _ = _solve_through_the_python_api(executor)
    ref_stats, ref_sol = _solve_through_the_system_binding(executor)

    assert api_stats["num_iters"] == ref_stats["num_iters"]
    assert float(api_stats["res_norm"]).hex() == float(ref_stats["res_norm"]).hex()
    for i, (got, want) in enumerate(zip(_boxes(api_sol), _boxes(ref_sol))):
        np.testing.assert_array_equal(
            got, want, err_msg=f"{executor}: solution box {i} differs bitwise"
        )


def test_matrix_reports_its_symmetry_and_shape():
    """The two questions a caller may ask of the matrix without solving.

    (It used to be three: ``is_assembled()`` went with the format erasure, which
    had two formats to distinguish and now has none.)
    """
    geom, ba, dm = _mesh()
    matrix = MFFaceCoeffs.symmetric(
        blockamr.MeshLevel(ba, dm, geom), executor=gko_executor("reference"), bc=_BC
    )

    assert matrix.is_symmetric() is True
    assert matrix.local_rows() == _N**3


def _precond_system(executor, n=_PRECOND_N):
    """The Helmholtz problem of the module docstring, on a mesh deep enough to coarsen.

    Bigger than ``_N``: a GMG hierarchy on 8^3 bottoms out after two coarsenings,
    which is enough to *work* but not enough for the iteration-count gap below to
    mean anything. 32^3 gives the V-cycle four levels and unpreconditioned CG
    something to be slow at.
    """
    geom, ba, dm = _mesh(n)
    gamma = _const_cell(ba, dm, 1.0)
    alpha = _const_cell(ba, dm, 1.0)
    rhs = _random_rhs(ba, dm)
    sol = _zero_sol(ba, dm)
    matrix = MFFaceCoeffs.symmetric(
        blockamr.MeshLevel(ba, dm, geom), executor=gko_executor(executor), bc=_BC
    )
    matrix.diagonal_source(alpha)
    system = LinearSystem(matrix, rhs)
    system += laplacian(gamma)
    # gamma/alpha/rhs are held by pointer or non-owningly; returning them keeps
    # them alive for as long as the system is.
    return system, sol, (geom, ba, dm, gamma, alpha, rhs)


def _solve_with_precond(executor, precond, **cfg):
    system, sol, _keepalive = _precond_system(executor)
    stats = Solver(SolverConfig(precond=precond, **_PRECOND_SOLVE, **cfg)).solve(system, sol)
    assert stats["converged"] is True, f"precond={precond!r} did not converge"
    return stats, sol


@pytest.mark.parametrize("executor", ["reference", "cuda"])
def test_gmg_precond_solves_and_beats_unpreconditioned(executor):
    """precond='gmg' is REACHABLE, and it is a real preconditioner, not a no-op.

    This is the positive twin of what used to be
    ``test_precond_is_refused_with_the_gmg_hierarchy_explanation``: the surface
    refused every precond because the hierarchy is built from the coefficient
    FIELDS and a solver holds only a LinOp. It is now built by the MATRIX, which
    still has the fields, so this asks the same question the other way round.

    Two claims, and both are needed. That it CONVERGES says the V-cycle is wired
    up the right way round -- a preconditioner cannot move the fixed point, only
    the path to it, so a wrong one shows up as a wrong answer or as no
    convergence. That it converges in materially FEWER iterations is what says the
    preconditioner is doing something at all; a silently-dropped one would still
    pass the first claim.
    """
    gmg, _sol_gmg = _solve_with_precond(executor, "gmg")
    plain, _sol_plain = _solve_with_precond(executor, "none")

    assert gmg["num_iters"] * 2 <= plain["num_iters"], (
        f"precond='gmg' took {gmg['num_iters']} iterations against "
        f"{plain['num_iters']} unpreconditioned -- not a material speed-up"
    )


@pytest.mark.parametrize("executor", ["reference", "cuda"])
def test_gmg_precond_reaches_the_same_answer_as_unpreconditioned(executor):
    """A preconditioner changes the path, never the fixed point.

    Compared RELATIVE to the solution's own scale rather than with a bare atol:
    on this problem the diagonal is ~6/dx^2, so the solution sits around 1e-4 and
    an absolute tolerance would silently be either vacuous or impossible. Both
    runs stop at rtol=1e-10 on the residual, which leaves each a few 1e-9 of
    solution error, hence a fixed 1e-6 of the peak -- three decades of headroom
    over the solve's own noise, three below any real disagreement.
    """
    _gmg_stats, sol_gmg = _solve_with_precond(executor, "gmg")
    _plain_stats, sol_plain = _solve_with_precond(executor, "none")

    scale = max(np.abs(box).max() for box in _boxes(sol_plain))
    for i, (got, want) in enumerate(zip(_boxes(sol_gmg), _boxes(sol_plain))):
        np.testing.assert_allclose(
            got,
            want,
            rtol=0.0,
            atol=1e-6 * scale,
            err_msg=f"{executor}: solution box {i} differs by more than 1e-6 of its scale",
        )


def test_gmg_kokkos_precond_solves_and_beats_unpreconditioned():
    """precond='gmg_kokkos' likewise -- the optimised V-cycle, through the same seam.

    cuda only: the ported V-cycle is a device path and has no ReferenceExecutor
    implementation, which it rejects at construction rather than ignoring.
    """
    try:
        kokkos, _sol = _solve_with_precond("cuda", "gmg_kokkos")
    except RuntimeError as exc:
        if "cuda" in str(exc).lower() and "unavailable" in str(exc).lower():
            pytest.skip(f"cuda executor unavailable: {exc}")
        raise
    plain, _sol_plain = _solve_with_precond("cuda", "none")

    assert kokkos["num_iters"] * 2 <= plain["num_iters"], (
        f"precond='gmg_kokkos' took {kokkos['num_iters']} iterations against "
        f"{plain['num_iters']} unpreconditioned -- not a material speed-up"
    )


def test_the_gmg_vcycle_knobs_reach_the_hierarchy():
    """``gmg=GmgConfig(...)`` is not decoration: a weaker cycle costs iterations.

    One V-cycle of a single pre- and post-sweep is a strictly weaker
    preconditioner than the default 2+2, so CG must need at least as many
    iterations with it. If the nested GmgConfig were dropped on the way to C++
    both runs would build the same hierarchy and the counts would be equal --
    which is why the assertion is strict.
    """
    default, _s0 = _solve_with_precond("reference", "gmg")
    weak, _s1 = _solve_with_precond(
        "reference", "gmg", gmg=GmgConfig(pre_sweeps=1, post_sweeps=1, omega=1.0)
    )

    assert weak["num_iters"] > default["num_iters"], (
        f"gmg=GmgConfig(pre_sweeps=1, post_sweeps=1) took {weak['num_iters']} iterations, "
        f"the same or fewer than the 2+2 default's {default['num_iters']} -- the knobs "
        "are not reaching the hierarchy"
    )


@pytest.mark.parametrize("solver", ["gmg", "ir", "mpir"])
def test_gmg_family_solvers_are_refused_with_the_hierarchy_explanation(solver):
    """solver='gmg'/'ir'/'mpir' need the coefficient-field hierarchy, not a LinOp."""
    geom, ba, dm = _mesh()
    gamma = _const_cell(ba, dm, 1.0)
    rhs = _random_rhs(ba, dm)
    sol = _zero_sol(ba, dm)
    matrix = MFFaceCoeffs.symmetric(
        blockamr.MeshLevel(ba, dm, geom), executor=gko_executor("reference"), bc=_BC
    )
    system = LinearSystem(matrix, rhs)
    system += laplacian(gamma)

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
